"""
ATLAS short-term evaluation of Lightning Network attachment strategies.

Run this module from the repository root, for example:

    python -m experiments.short_term random 100 1 1 1 --graphs 1,2,3

Arguments correspond to:
    strategy, transaction amount, first k, CPUs, seed.
"""

import argparse
from pathlib import Path

import networkx as nx

from atlas import graph_io as gio
from atlas import network_model as nm


ATTACHMENT_STRATEGIES = ("random", "highest_degree", "bc_approx_10", "k-center_gon_deg", "closeness")

# Representative LN snapshots used in the ATLAS evaluation.
SNAPSHOTS = {1: Path("./data/snapshots/ln_topology_2022_07_01.json"), 
2: Path("./data/snapshots/network_graph_2023_04_22.json"), 
3: Path("./data/snapshots/network_graph_2025_01_08.json")}


def parse_graph_ids(value: str) -> list[int]:
    """Parse a comma-separated list such as ``1,2,3``."""
    try:
        graph_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError( "Graph IDs must be comma-separated integers, e.g. 1,2,3." ) from exc

    unknown = [graph_id for graph_id in graph_ids if graph_id not in SNAPSHOTS]
    if unknown:
        raise argparse.ArgumentTypeError( f"Unknown graph IDs {unknown}; valid IDs are {sorted(SNAPSHOTS)}." )

    if not graph_ids:
        raise argparse.ArgumentTypeError("At least one graph ID is required.")

    return graph_ids


def process_graph(graph_path: Path, strategy: str, tx_amt: int, k_start: int, cpus: int | None, seed_val: int, graph_id: int) -> None:
    """Run the short-term experiment for one representative LN snapshot."""
    output_root = Path("./results/short_term") / strategy / f"graph_{graph_id}"

    if not graph_path.exists():
        raise FileNotFoundError( f"Snapshot for graph {graph_id} not found: {graph_path}" )

    g = gio.parse_multi_di_graph(str(graph_path), exclude_disabled=True)

    if g.number_of_edges() == 0:
        raise RuntimeError(f"Graph {graph_id} contains no active edges.")

    #u, v, edge_key = next(iter(g.edges(keys=True)))
    #print("Sample edge:")
    #print("base_fee =", g[u][v][edge_key]["base_fee"])
    #print("fee_rate =", g[u][v][edge_key]["fee_rate"])

    print( f"Graph {graph_id} has {g.number_of_nodes()} nodes " f"and {g.number_of_edges()} directed edges" )

    # Preserve the original ATLAS convention: short-term attachment is
    # evaluated on the largest strongly connected component.
    max_comp = max(nx.strongly_connected_components(g), key=len)
    g = g.subgraph(max_comp).copy()

    print( f"Graph {graph_id}: largest strongly connected component has " f"{g.number_of_nodes()} nodes and {g.number_of_edges()} edges" )

    for k in range(k_start, 16):
        result_dir = output_root / f"amt_{tx_amt}" / f"k_{k}"
        result_dir.mkdir(parents=True, exist_ok=True)

        result_file = ( result_dir / f"{strategy}_k{k}_{int(seed_val)}.txt" )

        sim_graph = g.copy()
        nm.run_network_model(sim_graph, tx_amt, strategy, k, None, str(result_file), c=cpus, s=int(seed_val))


def run_short_term( strategy: str = "random", tx_amt: int = 100, k_start: int = 1, cpus: int | None = None, seed_val: int = 1, graph_ids: tuple[int, ...] | list[int] = (1, 2, 3), ) -> None:
    """Run the short-term experiment on the selected representative graphs."""
    if strategy not in ATTACHMENT_STRATEGIES:
        raise ValueError( f"Unsupported strategy '{strategy}'. " f"Choose one of: {', '.join(ATTACHMENT_STRATEGIES)}." )

    if tx_amt not in (100, 10000, 1000000):
        raise ValueError("tx_amt must be one of 100, 10000, or 1000000 sat.")

    if k_start not in range(1, 16):
        raise ValueError("k_start must be between 1 and 15.")

    for graph_id in graph_ids:
        process_graph( graph_path=SNAPSHOTS[graph_id], strategy=strategy, tx_amt=tx_amt, k_start=k_start, cpus=cpus, seed_val=seed_val, graph_id=graph_id, )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser( description="Run the ATLAS short-term attachment-strategy experiment." )
    parser.add_argument("strategy", choices=ATTACHMENT_STRATEGIES)
    parser.add_argument( "tx_amt", type=int, choices=(100, 10000, 1000000), help="Transaction amount in satoshis.", )
    parser.add_argument( "k_start", type=int, choices=range(1, 16), metavar="[1-15]", help="First channel-count k; experiments continue through k=15.", )
    parser.add_argument( "cpus", type=int, nargs="?", default=None, help="Retained for compatibility with the original simulator.", )
    parser.add_argument("seed", type=int, nargs="?", default=1)
    parser.add_argument( "--graphs", type=parse_graph_ids, default=[1, 2, 3], help="Comma-separated representative graph IDs. Default: 1,2,3.", )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_short_term( strategy=args.strategy, tx_amt=args.tx_amt, k_start=args.k_start, cpus=args.cpus, seed_val=args.seed, graph_ids=args.graphs, )
