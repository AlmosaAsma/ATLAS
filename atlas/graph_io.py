"""ATLAS graph input/output and graph-view construction utilities.

This module provides the graph semantics used by ATLAS:

* Original LN snapshots are read as bidirectional channels and represented by
  two directed simulator edges whose initial balances split the advertised
  capacity equally.
* Evolved ATLAS checkpoints store one JSON record per directed simulator edge;
  those records are reloaded exactly as stored, without creating a reverse
  edge or halving capacity.
* Fee graphs are derived for a specified transaction amount.

The public function names are kept compatible with the rest of ATLAS so this
module can replace the previous ``graph_creation.py`` without changing callers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import numpy as np


_DEFAULT_POLICY = {
    "base_fee": 1000.0,
    "fee_rate": 1.0,
    "delay": 144,
    "disabled": False,
}


def _read_payload(path: str | Path) -> dict[str, Any]:
    if path is None:
        raise ValueError("A graph JSON file must be provided.")

    source = Path(path)
    print("opening", str(source), "...")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict) or "nodes" not in payload or "edges" not in payload:
        raise ValueError("Graph JSON must contain top-level 'nodes' and 'edges' fields.")

    return payload


def _policy_attributes(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    """Translate an LN JSON policy into simulator edge attributes."""
    if policy is None:
        return dict(_DEFAULT_POLICY)

    return {
        "base_fee": float(policy.get("fee_base_msat", _DEFAULT_POLICY["base_fee"])),
        "fee_rate": float(policy.get("fee_rate_milli_msat", _DEFAULT_POLICY["fee_rate"])),
        "delay": policy.get("time_lock_delta", _DEFAULT_POLICY["delay"]),
        "disabled": bool(policy.get("disabled", _DEFAULT_POLICY["disabled"])),
    }


def _add_snapshot_direction(
    graph: nx.MultiDiGraph,
    source: int,
    target: int,
    balance: float,
    policy: Mapping[str, Any] | None,
) -> None:
    attrs = _policy_attributes(policy)
    graph.add_edge(source, target, capacity=balance, **attrs)


def parse_multi_di_graph(file=None, exclude_disabled=False):
    """Load an original LN snapshot as the ATLAS directed channel graph.

    Each source JSON edge is interpreted as one bidirectional LN channel. Its
    advertised capacity is initialized as a 50:50 balance split across the two
    directed simulator edges.
    """
    payload = _read_payload(file)
    graph = nx.MultiDiGraph()

    pubkey_to_node: dict[str, int] = {}
    for node_id, node in enumerate(payload["nodes"]):
        pubkey = node["pub_key"]
        graph.add_node(node_id, key=pubkey, alias=node.get("alias", ""))
        pubkey_to_node[pubkey] = node_id

    for channel in payload["edges"]:
        if "node1_pub" not in channel or "node2_pub" not in channel:
            raise ValueError(
                "parse_multi_di_graph() expects original LN snapshot edges with "
                "node1_pub/node2_pub; use parse_saved_graph_json() for evolved ATLAS graphs."
            )

        first_policy = channel.get("node1_policy")
        second_policy = channel.get("node2_policy")
        channel_disabled = bool(first_policy and first_policy.get("disabled")) or bool(
            second_policy and second_policy.get("disabled")
        )
        if exclude_disabled and channel_disabled:
            continue

        u = pubkey_to_node[channel["node1_pub"]]
        v = pubkey_to_node[channel["node2_pub"]]
        half_capacity = int(channel["capacity"]) / 2

        _add_snapshot_direction(graph, u, v, half_capacity, first_policy)
        _add_snapshot_direction(graph, v, u, half_capacity, second_policy)

    return graph


def _serialize_policy(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fee_base_msat": str(attrs["base_fee"]),
        "fee_rate_milli_msat": str(attrs["fee_rate"]),
        "time_lock_delta": attrs["delay"],
        "disabled": attrs["disabled"],
    }


def save_graph_json(graph, filename):
    """Write an evolved ATLAS graph with one record per directed edge."""
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        nodes.append(
            {
                "id": node_id,
                "pub_key": str(attrs.get("key", "")),
                "alias": str(attrs.get("alias", "")),
                "last_update": attrs.get("last_update", 0),
            }
        )

    edges = []
    for source, target, attrs in graph.edges(data=True):
        policy = _serialize_policy(attrs)
        edges.append(
            {
                "node1_id": source,
                "node2_id": target,
                "capacity": int(attrs["capacity"]),
                "node1_policy": policy,
                "node2_policy": dict(policy),
                "channel_id": str(attrs.get("channel_id", "")),
                "chan_point": str(attrs.get("chan_point", "")),
            }
        )

    destination = Path(filename)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump({"nodes": nodes, "edges": edges}, handle, indent=4)


def parse_saved_graph_json(file, exclude_disabled=False):
    """Load an ATLAS evolved/checkpoint graph without altering directed balances."""
    payload = _read_payload(file)
    graph = nx.MultiDiGraph()

    for node in payload["nodes"]:
        node_id = int(node["id"])
        graph.add_node(
            node_id,
            key=node.get("pub_key", ""),
            alias=node.get("alias", ""),
            last_update=node.get("last_update", 0),
        )

    for record in payload["edges"]:
        if "node1_id" not in record or "node2_id" not in record:
            raise ValueError(
                "parse_saved_graph_json() expects evolved ATLAS edges with node1_id/node2_id."
            )

        attrs = _policy_attributes(record.get("node1_policy"))
        if exclude_disabled and attrs["disabled"]:
            continue

        graph.add_edge(
            int(record["node1_id"]),
            int(record["node2_id"]),
            capacity=float(record["capacity"]),
            channel_id=record.get("channel_id", ""),
            chan_point=record.get("chan_point", ""),
            **attrs,
        )

    return graph


def _reverse_balance(graph: nx.MultiDiGraph, source: int, target: int, key: int) -> float:
    """Return the opposite-direction balance for the matching simulator edge key."""
    if graph.has_edge(target, source) and key in graph[target][source]:
        return graph[target][source][key].get("capacity", 0)
    return 0


def _routing_fee_msat(edge: Mapping[str, Any], amount: float) -> float:
    fee_sat = amount * (int(edge["fee_rate"]) / 1_000_000) + (int(edge["base_fee"]) / 1000)
    if fee_sat == 0:
        fee_sat = np.finfo(np.float32).eps
    return fee_sat * 1000


def directed_fee_graph(g, tx_amount, exclude_edges=True, multi_graph=True):
    """Build the directed fee graph evaluated for ``tx_amount`` satoshis.

    For a simple fee graph, only the minimum-fee eligible parallel channel for
    each direction is retained. For a multigraph fee view, every eligible
    parallel channel is retained.
    """
    source = g if g.is_multigraph() else nx.MultiDiGraph(g)
    fee_graph = nx.MultiDiGraph() if multi_graph else nx.DiGraph()
    fee_graph.add_nodes_from(g.nodes(data=True))

    processed_pairs: set[tuple[int, int]] = set()
    for u, v in source.edges():
        if (u, v) in processed_pairs:
            continue
        processed_pairs.add((u, v))

        eligible_fees: list[float] = []
        for key, attrs in source[u][v].items():
            if exclude_edges:
                total_channel_balance = attrs.get("capacity", 0) + _reverse_balance(source, u, v, key)
                if total_channel_balance < tx_amount:
                    continue

            fee = _routing_fee_msat(attrs, tx_amount)
            eligible_fees.append(fee)
            if multi_graph:
                fee_graph.add_edge(u, v, fee=fee)

        if not multi_graph and eligible_fees:
            fee_graph.add_edge(u, v, fee=min(eligible_fees))

    return fee_graph


def multi_to_largest_di_graph(g):
    """Collapse parallel channels by retaining the pair with greatest total balance."""
    print("reducing MultiDiGraph to largest DiGraph...")
    result = nx.DiGraph()
    result.add_nodes_from(g.nodes(data=True))

    visited: set[frozenset[int]] = set()
    for u, v in g.edges():
        pair = frozenset((u, v))
        if pair in visited:
            continue
        visited.add(pair)

        if not g.has_edge(v, u):
            continue

        candidate_keys = set(g[u][v]).intersection(g[v][u])
        if not candidate_keys:
            continue

        best_key = max(
            candidate_keys,
            key=lambda key: g[u][v][key].get("capacity", 0) + g[v][u][key].get("capacity", 0),
        )

        for source_node, target_node in ((u, v), (v, u)):
            attrs = g[source_node][target_node][best_key]
            result.add_edge(
                source_node,
                target_node,
                capacity=attrs["capacity"],
                fee_rate=attrs["fee_rate"],
                base_fee=attrs["base_fee"],
                delay=attrs["delay"],
                disabled=attrs["disabled"],
            )

    return result
