"""
Attachment strategies used by ATLAS.

This module provides the same public strategy interface used by the ATLAS
experiments while implementing the selection logic independently around
NetworkX primitives.
"""

import random

import networkx as nx

from . import graph_io as gio
from . import utils


def _largest_strong_component(graph):
    """Return a copy of the largest strongly connected component."""
    component = max(nx.strongly_connected_components(graph), key=len)
    return graph.subgraph(component).copy()


def _eligible_nodes(graph, joining_node):
    """Nodes that are neither the joining node nor one of its current neighbours."""
    neighbours = set(graph.neighbors(joining_node))
    return list(set(graph.nodes()) - neighbours - {joining_node})


def _pick_from_ranked(ranked_nodes, k):
    """
    Select k node IDs from a descending (node, score) ranking.

    Equal-score nodes are randomly ordered when a tie crosses the selection
    boundary, matching the stochastic tie handling used by ATLAS.
    """
    ranked_nodes = list(ranked_nodes)
    selected = []

    while len(selected) < k and ranked_nodes:
        best_score = ranked_nodes[0][1]
        tied = [item for item in ranked_nodes if item[1] == best_score]

        remaining = k - len(selected)
        if len(tied) > remaining:
            random.shuffle(tied)
            selected.extend(tied[:remaining])
            break

        selected.extend(tied)
        ranked_nodes = ranked_nodes[len(tied):]

    return list(dict(selected).keys())


def select_attachment_nodes(graph, n, k, strategy='random', weight=None, weight_value=None,):
    """Return up to k attachment candidates for node n."""
    if k == 0:
        return []

    working_graph = _largest_strong_component(graph.copy())
    print( "Attachment strategy using largest strongly connected component  " "({} nodes)".format(working_graph.number_of_nodes()) )

    if n not in working_graph:
        working_graph.add_node( n, key='joining_node', alias='new_node', )

    nodes = _eligible_nodes(working_graph, n)
    if len(nodes) <= k:
        return nodes

    if strategy == 'random':
        return choose_random(k, nodes)

    if strategy == 'k-center_gon_deg':
        return k_center_gon_deg(working_graph, n, k)

    if strategy == 'highest_degree':
        ranked = sort_by_degree(working_graph, nodes)
    elif strategy == 'bc':
        ranked = sort_by_centrality( working_graph, n, nodes, weight=weight )
    elif strategy == 'bc_approx_10':
        ranked = sort_by_centrality_approx( working_graph, weight=weight, factor=0.1 )
    elif strategy == 'closeness':
        ranked = sort_by_closeness( working_graph, nodes)
    else:
        raise ValueError("Unknown attachment strategy: {}".format(strategy))

    return _pick_from_ranked(ranked, k)


def choose_random(k, nodes):
    """Uniformly sample k distinct candidates."""
    print("choosing", k, "random node(s)...")
    return random.sample(nodes, k)


def sort_by_degree(g, nodes):
    """Rank eligible nodes by total directed degree."""
    print("sorting nodes by degree...")

    eligible = set(nodes)
    return  sorted(
        ((node, g.degree(node)) for node in eligible),
        key=lambda item: item[1],
        reverse=True
    )


def sort_by_centrality_approx(g, weight=None, factor=0.5):
    """
    Rank nodes using sampled betweenness centrality.

    `factor` is the fraction of graph nodes used as NetworkX samples.
    """
    if factor < 0 or factor > 1:
        print("Factor has to be between 0 and 1!")
        return []

    print( "sorting by betweenness centrality " "(approximate with {} percent of all nodes)...".format(factor * 100) )

    sample_count = int(g.number_of_nodes() * factor)
    kwargs = {
        'normalized': False,
        'k': sample_count,
    }
    if weight is not None:
        kwargs['weight'] = weight

    centrality = nx.betweenness_centrality(g, **kwargs)
    return sorted( centrality.items(), key=lambda item: item[1], reverse=True, )


def sort_by_closeness(g, nodes):
    """Rank eligible nodes by NetworkX closeness centrality."""
    print("sorting nodes by closeness...")

    eligible = set(nodes)
    scores = nx.closeness_centrality(g)

    return sorted(
        ((node, score) for node, score in scores.items() if node in eligible),
        key=lambda item: item[1],
        reverse=True
    )


def gonzalez(g, n, k):
    """
    Gonzalez farthest-first attachment heuristic.

    Each selected candidate is connected to n in both directions before the
    next farthest node is chosen.
    """
    if nx.degree(g, n) < 1:
        print( "Cannot execute gonzalez' algorithm. " "Node does not have any neighbor yet!" )
        return []

    candidates = []

    while len(candidates) != k:
        distances = nx.single_source_shortest_path_length(g, n)
        farthest_distance = max(distances.values())
        farthest_nodes = [
            node
            for node, distance in distances.items()
            if distance == farthest_distance
        ]

        candidate = random.choice(farthest_nodes)
        g.add_edge(n, candidate)
        g.add_edge(candidate, n)
        candidates.append(candidate)

    return candidates


def k_center_gon_deg(g, n, k):
    """Run the unweighted Gonzalez-based k-center attachment strategy."""
    print("starting k-center_gon_deg algorithm...", end=" ")

    working_graph = g.copy()
    if g.is_multigraph():
        working_graph = gio.multi_to_largest_di_graph(g)

    working_graph = _largest_strong_component(working_graph)
    print( "using largest strongly connected component " "({} nodes)".format(working_graph.number_of_nodes()) )

    candidates = []

    if n not in working_graph:
        working_graph.add_node(n)

    if nx.degree(working_graph, n) == 0:
        initial = utils.create_initial_connection(working_graph, n, None, None, metric='degree')
        candidates.append(initial)
        k -= 1
        print( "Added (permanent) channel from", n, "to", initial, "(initial connection to the network)", )

    candidates.extend(gonzalez(working_graph, n, k))
    return candidates


def _simple_directed_graph(graph):
    """Convert a graph to the directed representation used for BC."""
    if graph.is_multigraph():
        return gio.multi_to_largest_di_graph(graph)
    return nx.DiGraph(graph)


def _component_betweenness(component_graph, weight=None, edges=False):
    """Compute unnormalised node or edge BC for one strongly connected component."""
    if edges:
        return nx.edge_betweenness_centrality( component_graph, normalized=False, weight=weight, )

    return nx.betweenness_centrality( component_graph, normalized=False, weight=weight, )


def sort_by_centrality(g, exclude_n=None, nodes=None, weight=None, max_elem=False, edges=False,):
    """
    Rank nodes or edges by betweenness centrality.

    Centrality is evaluated separately inside each strongly connected
    component, as in the ATLAS strategy definition.
    """
    print("sorting by betweenness centrality...")

    directed = _simple_directed_graph(g)
    if exclude_n is not None and exclude_n in directed:
        directed.remove_node(exclude_n)

    components = list(nx.strongly_connected_components(directed))
    print("{} strongly connected component(s)".format(len(components)))

    scores = {}
    for component in components:
        subgraph = directed.subgraph(component).copy()
        scores.update( _component_betweenness( subgraph, weight=weight, edges=edges, ) )

    if max_elem:
        return max(scores, key=scores.get)

    ranked = list(scores.items())

    if nodes is not None:
        eligible = set(nodes)
        ranked = [
            item
            for item in ranked
            if item[0] in eligible
        ]

    ranked.sort( key=lambda item: item[1], reverse=True, )
    return ranked
