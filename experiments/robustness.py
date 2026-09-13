"""
ATLAS targeted-node-removal robustness experiment.

This code performs robustness evaluation using ATLAS attack logic while restricting
the attack choices to the two targeted attacks reported in the paper:

    highest_degree
    highest_betweenness

The attachment strategy is inferred from each supplied long-term graph filename.
The long-term graphs themselves should correspond to one of the five ATLAS
attachment strategies:

    random, highest_degree, bc_approx_10, k-center_gon_deg, closeness

Metrics:
    all      transaction success, average fees, LCC, and C(G)
    lcc      largest connected component only
    cg       residual connectivity C(G) only

Internally, LCC is computed as the largest weakly connected component (LWCC)
because the ATLAS graph is directed. The CLI/output terminology remains "lcc".

Transactions are sampled from the entire remaining graph after attack.
Disconnected sender/receiver pairs therefore count as failed payments.

C(G) = sum_i p_i^2, where p_i is the proportion of remaining nodes in weakly
connected component i. This is the connectivity produced by the stated targeted
removal heuristic; it is not the globally optimised R_k(G).

Example:

    python -m experiments.robustness highest_degree all --seed-start 1 --seed-end 10

By default, the script evaluates the explicitly listed GRAPH_PATHS below.
Use --graph-path to override that list and evaluate one evolved graph only.
"""

import argparse
import os
import re
import time
from pathlib import Path
from random import seed

import networkx as nx

from atlas import graph_io as gio
from atlas import simulator as sim


ATTACHMENT_STRATEGIES = {
    "random",
    "highest_degree",
    "bc_approx_10",
    "k-center_gon_deg",
    "closeness",
}

REMOVAL_STRATEGIES = ( "highest_degree", "highest_betweenness", )


# ---------------------------------------------------------------------
# Long-term graphs used for robustness evaluation
# ---------------------------------------------------------------------

GRAPH_PATHS = [
    # Graph 1
    "./results/long_term/random/graph_1/seed_1/updated_graph1_random_1_10000.json",
    "./results/long_term/highest_degree/graph_1/seed_1/updated_graph1_highest_degree_1_10000.json",
    "./results/long_term/bc_approx_10/graph_1/seed_1/updated_graph1_bc_approx_10_1_10000.json",
    "./results/long_term/k-center_gon_deg/graph_1/seed_1/updated_graph1_k-center_gon_deg_1_10000.json",
    "./results/long_term/closeness/graph_1/seed_1/updated_graph1_closeness_1_10000.json",

    # Graph 2
    "./results/long_term/random/graph_2/seed_1/updated_graph2_random_1_10000.json",
    "./results/long_term/highest_degree/graph_2/seed_1/updated_graph2_highest_degree_1_10000.json",
    "./results/long_term/bc_approx_10/graph_2/seed_1/updated_graph2_bc_approx_10_1_10000.json",
    "./results/long_term/k-center_gon_deg/graph_2/seed_1/updated_graph2_k-center_gon_deg_1_10000.json",
    "./results/long_term/closeness/graph_2/seed_1/updated_graph2_closeness_1_10000.json",

    # Graph 3
    "./results/long_term/random/graph_3/seed_1/updated_graph3_random_1_10000.json",
    "./results/long_term/highest_degree/graph_3/seed_1/updated_graph3_highest_degree_1_10000.json",
    "./results/long_term/bc_approx_10/graph_3/seed_1/updated_graph3_bc_approx_10_1_10000.json",
    "./results/long_term/k-center_gon_deg/graph_3/seed_1/updated_graph3_k-center_gon_deg_1_10000.json",
    "./results/long_term/closeness/graph_3/seed_1/updated_graph3_closeness_1_10000.json",
]

def extract_attachment_strategy(filename: str) -> str:
    """Infer the attachment strategy from a saved graph filename."""
    base = os.path.basename(filename)

    for strategy in sorted( ATTACHMENT_STRATEGIES, key=len, reverse=True, ):
        if strategy in base:
            return strategy

    return "unknown"


def extract_graph_number(filename: str) -> str:
    """Extract ``graphN`` from a filename when present."""
    base = os.path.basename(filename)
    match = re.search(r"graph(\d+)", base)
    return match.group(1) if match else "1"


def rank_nodes( G: nx.MultiDiGraph, removal_strategy: str, seed_val: int, ):
    """Compute one fixed targeted-removal ranking from the intact graph."""
    del seed_val  # retained in the interface for compatibility

    if removal_strategy == "highest_degree":
        return [
            node
            for node, _ in sorted( G.degree(), key=lambda item: item[1], reverse=True, )
        ]

    if removal_strategy == "highest_betweenness":
        print("Computing betweenness-centrality attack ranking...")
        start = time.perf_counter()
        bc = nx.betweenness_centrality(G)
        print( "Betweenness ranking completed in " f"{time.perf_counter() - start:.2f}s" )

        return [
            node
            for node, _ in sorted( bc.items(), key=lambda item: item[1], reverse=True, )
        ]

    raise ValueError( "Unsupported removal strategy. " "Choose highest_degree or highest_betweenness." )


def get_weak_components(G):
    """Return weak components for directed graphs or components otherwise."""
    if G.number_of_nodes() == 0:
        return []

    if G.is_directed():
        return list(nx.weakly_connected_components(G))

    return list(nx.connected_components(G))


def compute_lwcc(G):
    """
    Return the largest weakly connected component size.

    Returns:
        LWCC node count,
        LWCC directed-edge count.
    """
    components = get_weak_components(G)

    if not components:
        return 0, 0

    largest = max(components, key=len)
    subgraph = G.subgraph(largest)

    return ( subgraph.number_of_nodes(), subgraph.number_of_edges(), )


def compute_cg(G):
    """
    Compute residual connectivity C(G) = sum_i p_i^2.

    Returns:
        C(G),
        number of components,
        component sizes in descending order.
    """
    n_remaining = G.number_of_nodes()

    if n_remaining == 0:
        return 0.0, 0, []

    components = get_weak_components(G)
    component_sizes = sorted( (len(component) for component in components), reverse=True, )

    cg_value = sum( (size / n_remaining) ** 2 for size in component_sizes )

    return ( cg_value, len(component_sizes), component_sizes, )


def append_tsv(path, header, row):
    """Append one tab-separated row, adding the header when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            f.write("\t".join(header) + "\n")

    with path.open("a", encoding="utf-8") as f:
        f.write("\t".join(str(value) for value in row) + "\n")


def write_removed_nodes(path, nodes):
    """Save the exact attacked-node set for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("Node_ID\n")
        for node in nodes:
            f.write(f"{node}\n")


def write_component_sizes(path, component_sizes):
    """Save all residual component sizes used to reconstruct C(G)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("Component_Rank\tNodes\n")
        for rank, size in enumerate(component_sizes, start=1):
            f.write(f"{rank}\t{size}\n")


def evaluate_graph( graph_path: str, removal_strategy: str, metric: str, tx_amt: int, tx_num: int, seed_val: int, result_version: str, max_remove_pct: int, nodes_added: int, ) -> None:
    """Evaluate one evolved long-term graph under one targeted attack."""
    attachment_strategy = extract_attachment_strategy(graph_path)

    if attachment_strategy not in ATTACHMENT_STRATEGIES:
        raise ValueError( "Could not infer one of the ATLAS attachment strategies " f"from filename: {graph_path}" )

    graph_number = extract_graph_number(graph_path)

    need_transactions = metric == "all"
    need_lcc = metric in {"all", "lcc"}
    need_cg = metric in {"all", "cg"}

    graph_folder = ( Path("./results/robustness") / f"{nodes_added}_nodes_{result_version}" / f"{removal_strategy}_removed" / attachment_strategy / f"graph_{graph_number}" )
    graph_folder.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"Graph: {graph_path}")
    print(f"Attachment strategy: {attachment_strategy}")
    print(f"Graph number: {graph_number}")
    print(f"Removal strategy: {removal_strategy}")
    print(f"Metric: {metric}")
    print(f"Seed: {seed_val}")
    print("=" * 72)

    g_full = gio.parse_saved_graph_json( graph_path, exclude_disabled=True, )

    ranked_nodes = rank_nodes( g_full, removal_strategy, seed_val, )

    original_n = g_full.number_of_nodes()

    for pct in range(0, max_remove_pct + 1):
        pct_folder = graph_folder / f"{pct / 100:.2f}"
        pct_folder.mkdir(parents=True, exist_ok=True)

        g_copy = g_full.copy()

        if pct == 0:
            nodes_to_remove = []
            print("\nBaseline: 0% removal")
        else:
            num_remove = max( 1, int(original_n * pct / 100), )
            nodes_to_remove = ranked_nodes[:num_remove]
            g_copy.remove_nodes_from(nodes_to_remove)

            print( f"\n[Seed {seed_val}] Removed top {pct}% " f"({num_remove} nodes) using {removal_strategy}" )

        write_removed_nodes( pct_folder / f"removed_nodes_seed{seed_val}.txt", nodes_to_remove, )

        if need_lcc:
            lwcc_nodes, lwcc_edges = compute_lwcc(g_copy)

            append_tsv( pct_folder / "lcc.txt", ["Seed", "LCC_Nodes", "LCC_Edges"], [seed_val, lwcc_nodes, lwcc_edges], )

            print( f"LCC (computed as LWCC): " f"{lwcc_nodes} nodes, {lwcc_edges} edges" )

        if need_cg:
            cg_value, num_components, component_sizes = compute_cg(g_copy)
            largest_component = component_sizes[0] if component_sizes else 0

            append_tsv(
                pct_folder / "connectivity_cg.txt",
                [
                    "Seed",
                    "Remaining_Nodes",
                    "Num_Components",
                    "Largest_Component_Nodes",
                    "C_G",
                ],
                [
                    seed_val,
                    g_copy.number_of_nodes(),
                    num_components,
                    largest_component,
                    f"{cg_value:.12f}",
                ],
            )

            write_component_sizes( pct_folder / f"component_sizes_seed{seed_val}.txt", component_sizes, )

            print( f"C(G): {cg_value:.12f} | " f"components: {num_components}" )

        if need_transactions:
            results_file = pct_folder / f"seed{seed_val}.txt"
            start_time = time.perf_counter()

            results = sim.simulate_transactions( g_copy, tx_amt=tx_amt, tx_num=tx_num, src=None, file=str(results_file), )

            duration = time.perf_counter() - start_time

            total_attempts = sum( values[0] for values in results.values() )
            total_sent = sum( values[1] for values in results.values() )
            total_fees = sum( values[4] for values in results.values() )

            success_rate = ( total_sent / total_attempts if total_attempts > 0 else 0 )
            avg_fees = ( total_fees / total_sent if total_sent > 0 else 0 )

            print( f"Transaction simulation: {duration:.2f}s | " f"success={success_rate:.6f} | " f"avg_fees={avg_fees:.6f}" )

            append_tsv( pct_folder / "stats.txt", [ "Seed", "Success_Rate", "Average_Fees", ], [ seed_val, f"{success_rate:.6f}", f"{avg_fees:.6f}", ], )


def start_robustness_eval(
    graph_path: str | None = None,
    removal_strategy: str = "highest_degree",
    metric: str = "all",
    tx_amt: int = 100,
    tx_num: int = 1000,
    seed_val: int = 1,
    result_version: str = "1",
    max_remove_pct: int = 5,
    nodes_added: int = 10000,
) -> None:
    """Evaluate the configured evolved graphs, or one graph supplied by --graph-path."""
    valid_metrics = {"all", "lcc", "cg"}

    if removal_strategy not in REMOVAL_STRATEGIES:
        raise ValueError( f"Choose one of: {', '.join(REMOVAL_STRATEGIES)}." )

    if metric not in valid_metrics:
        raise ValueError( f"Unknown metric '{metric}'. " f"Choose one of: {', '.join(sorted(valid_metrics))}." )

    graph_paths = [graph_path] if graph_path else GRAPH_PATHS

    for graph_path in graph_paths:
        if not os.path.exists(graph_path):
            print( f"WARNING: graph not found; skipping: {graph_path}" )
            continue

        seed(seed_val)

        evaluate_graph(
            graph_path=graph_path,
            removal_strategy=removal_strategy,
            metric=metric,
            tx_amt=tx_amt,
            tx_num=tx_num,
            seed_val=seed_val,
            result_version=result_version,
            max_remove_pct=max_remove_pct,
            nodes_added=nodes_added,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser( description=( "Run ATLAS targeted-node-removal robustness evaluation." ) )

    parser.add_argument( "removal_strategy", choices=REMOVAL_STRATEGIES, )

    parser.add_argument( "metric", choices=("all", "lcc", "cg"), help=( "'all' runs transactions, LCC and C(G); " "'lcc' computes only LCC; " "'cg' computes only C(G)." ), )

    parser.add_argument( "--graph-path", default=None, help=( "Optional path to one evolved long-term JSON graph. " "If omitted, the GRAPH_PATHS list in this file is evaluated." ), )

    parser.add_argument( "--nodes-added", type=int, default=10000, help=( "Long-term checkpoint to evaluate. Default: 10000." ), )

    parser.add_argument("--tx-amt", type=int, default=100)
    parser.add_argument("--tx-num", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=10)
    parser.add_argument("--result-version", default="1")
    parser.add_argument("--max-remove-pct", type=int, default=5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for seed_val in range( args.seed_start, args.seed_end + 1, ):
        print( "\nRunning robustness evaluation: " f"strategy={args.removal_strategy}, " f"metric={args.metric}, " f"tx_amt={args.tx_amt}, " f"tx_num={args.tx_num}, " f"seed={seed_val}" )

        start_robustness_eval(
            graph_path=args.graph_path,
            removal_strategy=args.removal_strategy,
            metric=args.metric,
            tx_amt=args.tx_amt,
            tx_num=args.tx_num,
            seed_val=seed_val,
            result_version=args.result_version,
            max_remove_pct=args.max_remove_pct,
            nodes_added=args.nodes_added,
        )
