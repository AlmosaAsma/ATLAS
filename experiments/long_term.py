"""
ATLAS long-term evaluation of Lightning Network attachment strategies with network churn.

Example:

    python -m experiments.long_term random 1 0.956 100 1000 10000 1 1

Arguments:
    strategy graph_id churn_rate tx_amt eval_interval nodes_to_add cpus seed
"""

import argparse
import time
from pathlib import Path

import networkx as nx

from atlas import graph_io as gio
from atlas import network_model as nm


ATTACHMENT_STRATEGIES = ( "random", "highest_degree", "bc_approx_10", "k-center_gon_deg", "closeness", )

SNAPSHOTS = {1: Path("./data/snapshots/ln_topology_2022_07_01.json"), 2: Path("./data/snapshots/network_graph_2023_04_22.json"), 3: Path("./data/snapshots/network_graph_2025_01_08.json")}


def start_eval( strategy: str, graph_id: int, churn_rate: float, tx_amt: int = 100, interval: int = 1000, n: int = 10000, cpus: int | None = None, seed_val: int = 1, ) -> None:
    """
    Add ``n`` nodes to one representative LN topology and periodically apply
    churn, payment simulation, and structural evaluation.

    The graph is reduced to its largest strongly connected component before
    growth.
    """
    if strategy not in ATTACHMENT_STRATEGIES:
        raise ValueError( f"Unsupported strategy '{strategy}'. " f"Choose one of: {', '.join(ATTACHMENT_STRATEGIES)}." )

    if graph_id not in SNAPSHOTS:
        raise ValueError(f"graph_id must be one of {sorted(SNAPSHOTS)}.")

    graph_path = SNAPSHOTS[graph_id]

    if not graph_path.exists():
        raise FileNotFoundError( f"Snapshot for graph {graph_id} not found: {graph_path}" )

    graph_folder = ( Path("./results/long_term") / strategy / f"graph_{graph_id}" / f"seed_{seed_val}" )
    graph_folder.mkdir(parents=True, exist_ok=True)

    g = gio.parse_multi_di_graph(str(graph_path), exclude_disabled=True)

    print( f"Graph {graph_id} has {g.number_of_nodes()} nodes, " f"{g.number_of_edges()} directed edges and " f"{g.number_of_edges() / 2} active channels" )

    max_comp = max(nx.strongly_connected_components(g), key=len)
    g = g.subgraph(max_comp).copy()

    print( f"Graph {graph_id}: largest strongly connected component has " f"{g.number_of_nodes()} nodes and {g.number_of_edges()} edges" )

    start = time.perf_counter()

    nm.run_network_model(g, tx_amt, strategy, n, str(graph_folder) + "/", None, churn_rate, i=interval, c=cpus, s=seed_val, graph_id=graph_id)

    print( f"Graph {graph_id}: simulation took " f"{time.perf_counter() - start:.2f} seconds" )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser( description="Run the ATLAS long-term growth experiment with churn." )
    parser.add_argument("strategy", choices=ATTACHMENT_STRATEGIES)
    parser.add_argument( "graph_id", type=int, choices=sorted(SNAPSHOTS), help="Representative topology: 1, 2, or 3.", )
    parser.add_argument( "churn_rate", type=float, help=( "Period-specific churn factor: number of nodes removed per " "joining node, as calibrated from the corresponding data period." ), )
    parser.add_argument( "tx_amt", type=int, nargs="?", default=100, choices=(100, 10000, 1000000), )
    parser.add_argument("eval_interval", type=int, nargs="?", default=1000)
    parser.add_argument("nodes_to_add", type=int, nargs="?", default=10000)
    parser.add_argument( "cpus", type=int, nargs="?", default=None, help="Retained for compatibility with the original simulator.", )
    parser.add_argument("seed", type=int, nargs="?", default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_eval( strategy=args.strategy, graph_id=args.graph_id, churn_rate=args.churn_rate, tx_amt=args.tx_amt, interval=args.eval_interval, n=args.nodes_to_add, cpus=args.cpus, seed_val=args.seed, )
