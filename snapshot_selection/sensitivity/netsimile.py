"""
NetSimile graph-signature and clustering pipeline.

Based on:
M. Berlingerio, D. Koutra, T. Eliassi-Rad, and C. Faloutsos,
"NetSimile: A Scalable Approach to Size-Independent Network Similarity,"
arXiv:1209.2684, 2012.

Paper-faithful components
------------------------
1. Seven local/egonet features per node.
2. Five aggregators per feature in the paper's order:
   median, mean, standard deviation, skewness, kurtosis.
3. A 35-dimensional graph signature.
4. Canberra distance for graph comparison.

Clustering note
---------------
The paper leaves the clustering algorithm open. This implementation uses average-linkage agglomerative clustering over the precomputed Canberra
distance matrix. The number of clusters is selected by the highest silhouette score computed from that same distance matrix. Each representative is a
cluster medoid: the actual snapshot with the smallest total Canberra distance to the other members of its cluster.

Dependencies
------------
pip install numpy scipy networkx scikit-learn tqdm
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kurtosis, skew
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("./data/snapshots")
OUTPUT_DIR = Path("./results/snapshot_selection/sensitivity/netsimile")

SIGNATURES_CSV = OUTPUT_DIR / "netsimile_signatures.csv"
DISTANCES_CSV = OUTPUT_DIR / "netsimile_canberra_distances.csv"
K_RESULTS_CSV = OUTPUT_DIR / "netsimile_k_metrics.csv"
CLUSTER_MEMBERS_CSV = OUTPUT_DIR / "netsimile_cluster_members.csv"
REPRESENTATIVES_CSV = OUTPUT_DIR / "netsimile_representatives.csv"
LOG_FILE = OUTPUT_DIR / "netsimile_pipeline_log.txt"

MIN_K = 2
MAX_K = 10

# The original paper is formulated for unweighted, undirected graphs.
# Lightning Network channel policies and capacities are therefore not used
# when constructing the NetSimile signature.
SUPPORTED_EXTENSIONS = {".json"}


# ---------------------------------------------------------------------------
# Feature and aggregation names
# ---------------------------------------------------------------------------

NODE_FEATURE_NAMES = [
    "degree",
    "clustering_coefficient",
    "average_neighbour_degree",
    "average_neighbour_clustering",
    "egonet_internal_edges",
    "egonet_outgoing_edges",
    "egonet_external_neighbours",
]

# Keep the exact order used in Algorithm 3 of the paper.
AGGREGATOR_NAMES = [
    "median",
    "mean",
    "standard_deviation",
    "skewness",
    "kurtosis",
]

SIGNATURE_FEATURE_NAMES = [
    f"{feature}_{aggregator}"
    for feature in NODE_FEATURE_NAMES
    for aggregator in AGGREGATOR_NAMES
]


@dataclass(frozen=True)
class KResult:
    k: int
    silhouette: float
    total_within_cluster_distance: float


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def reset_log() -> None:
    LOG_FILE.write_text("", encoding="utf-8")


def log(message: str = "") -> None:
    print(message)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


# ---------------------------------------------------------------------------
# Lightning Network JSON loading
# ---------------------------------------------------------------------------


def load_graph_from_json(path: Path) -> nx.Graph:
    """Load one Lightning Network JSON snapshot as a simple undirected graph.

    Expected fields are compatible with LND describegraph-style files:
    - nodes[*].pub_key
    - edges[*].node1_pub
    - edges[*].node2_pub

    Parallel channels collapse into one undirected edge because the original
    NetSimile formulation uses simple unweighted graphs.
    """

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    graph = nx.Graph()

    for node in data.get("nodes", []):
        pub_key = node.get("pub_key")
        if pub_key:
            graph.add_node(str(pub_key))

    for edge in data.get("edges", []):
        node1 = edge.get("node1_pub")
        node2 = edge.get("node2_pub")

        if not node1 or not node2:
            continue

        node1 = str(node1)
        node2 = str(node2)

        # Ignore self-loops. NetSimile's degree, clustering, and egonet
        # definitions assume ordinary graph edges between distinct nodes.
        if node1 != node2:
            graph.add_edge(node1, node2)

    # Remove any self-loop that may have entered through malformed data.
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def load_graphs(directory: Path) -> Tuple[List[nx.Graph], List[str]]:
    """Load all valid JSON snapshots in deterministic filename order."""

    if not directory.exists():
        raise FileNotFoundError(f"Snapshot directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Snapshot path is not a directory: {directory}")

    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not paths:
        raise FileNotFoundError(f"No JSON snapshots found in {directory}")

    graphs: List[nx.Graph] = []
    filenames: List[str] = []

    log("Loading snapshots...")

    for path in tqdm(paths, desc="Loading graphs"):
        try:
            graph = load_graph_from_json(path)
        except ValueError as exc:
            log(f"WARNING: skipping {path.name}: {exc}")
            continue

        if graph.number_of_nodes() < 2:
            log(f"WARNING: skipping {path.name}: fewer than two nodes")
            continue

        graphs.append(graph)
        filenames.append(path.name)

    if len(graphs) < 3:
        raise ValueError("At least three valid graphs are required for clustering.")

    log(f"Loaded {len(graphs)} graphs.")
    return graphs, filenames


# ---------------------------------------------------------------------------
# NetSimile feature extraction: Algorithm 2
# ---------------------------------------------------------------------------


def extract_node_features(graph: nx.Graph) -> np.ndarray:
    """Return the node-by-7 NetSimile feature matrix for one graph.

    The columns correspond to Algorithm 2 in the original paper:

    1. degree of node i;
    2. clustering coefficient of node i;
    3. average degree of i's neighbours;
    4. average clustering coefficient of i's neighbours;
    5. number of edges inside i's egonet;
    6. number of edges leaving i's egonet;
    7. number of distinct external neighbours of i's egonet.

    Here the radius-1 egonet contains the focal node and its immediate neighbours. The egonet is the subgraph induced by that node set.
    """

    if graph.is_directed():
        graph = graph.to_undirected(as_view=False)
    if graph.is_multigraph():
        graph = nx.Graph(graph)

    graph.remove_edges_from(nx.selfloop_edges(graph))

    degrees: Dict[str, int] = dict(graph.degree())
    clustering: Dict[str, float] = nx.clustering(graph)

    rows: List[List[float]] = []

    for node in graph.nodes():
        neighbours = set(graph.neighbors(node))
        degree = float(degrees[node])
        node_clustering = float(clustering[node])

        if neighbours:
            average_neighbour_degree = float(
                np.mean([degrees[neighbour] for neighbour in neighbours])
            )
            average_neighbour_clustering = float(
                np.mean([clustering[neighbour] for neighbour in neighbours])
            )
        else:
            average_neighbour_degree = 0.0
            average_neighbour_clustering = 0.0

        egonet_nodes = neighbours | {node}

        # Count edges whose two endpoints are both in the egonet.
        egonet_internal_edges = 0

        # Count edges with exactly one endpoint in the egonet and collect the
        # distinct nodes reached by those outgoing edges.
        egonet_outgoing_edges = 0
        external_neighbours = set()

        for source in egonet_nodes:
            for target in graph.neighbors(source):
                if target in egonet_nodes:
                    # Each undirected internal edge is visited from both ends.
                    egonet_internal_edges += 1
                else:
                    egonet_outgoing_edges += 1
                    external_neighbours.add(target)

        egonet_internal_edges //= 2

        rows.append(
            [
                degree,
                node_clustering,
                average_neighbour_degree,
                average_neighbour_clustering,
                float(egonet_internal_edges),
                float(egonet_outgoing_edges),
                float(len(external_neighbours)),
            ]
        )

    matrix = np.asarray(rows, dtype=float)

    if matrix.shape != (graph.number_of_nodes(), 7):
        raise RuntimeError(
            f"Unexpected node-feature shape {matrix.shape}; expected "
            f"({graph.number_of_nodes()}, 7)."
        )

    return matrix


# ---------------------------------------------------------------------------
# NetSimile aggregation: Algorithm 3
# ---------------------------------------------------------------------------


def safe_skewness(values: np.ndarray) -> float:
    """Compute skewness and return zero for undefined constant samples."""

    if values.size < 3 or np.allclose(values, values[0]):
        return 0.0

    value = float(skew(values, bias=True, nan_policy="omit"))
    return value if np.isfinite(value) else 0.0


def safe_kurtosis(values: np.ndarray) -> float:
    """Compute excess kurtosis and return zero when it is undefined.

    The paper names kurtosis but does not specify Fisher versus Pearson convention. This implementation uses excess kurtosis (Fisher=True), the
    SciPy default, and records that choice explicitly for reproducibility.
    """

    if values.size < 4 or np.allclose(values, values[0]):
        return 0.0

    value = float(
        kurtosis(values, fisher=True, bias=True, nan_policy="omit")
    )
    return value if np.isfinite(value) else 0.0


def aggregate_feature(values: np.ndarray) -> List[float]:
    """Apply Algorithm 3's five aggregators in the paper's order."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return [0.0] * 5

    return [
        float(np.median(values)),
        float(np.mean(values)),
        float(np.std(values, ddof=0)),
        safe_skewness(values),
        safe_kurtosis(values),
    ]


def compute_netsimile_signature(graph: nx.Graph) -> np.ndarray:
    """Create the 35-dimensional NetSimile signature for one graph."""

    node_features = extract_node_features(graph)
    signature: List[float] = []

    for column_index in range(node_features.shape[1]):
        signature.extend(aggregate_feature(node_features[:, column_index]))

    result = np.asarray(signature, dtype=float)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    if result.shape != (35,):
        raise RuntimeError(
            f"Unexpected signature shape {result.shape}; expected (35,)."
        )

    return result


def embed_all_graphs(graphs: Sequence[nx.Graph]) -> np.ndarray:
    """Compute a 35-dimensional signature for every graph."""

    log("Computing NetSimile signatures...")
    signatures = [
        compute_netsimile_signature(graph)
        for graph in tqdm(graphs, desc="NetSimile signatures")
    ]

    matrix = np.vstack(signatures)
    log(f"Finished computing signatures. Shape: {matrix.shape}")
    return matrix


# ---------------------------------------------------------------------------
# NetSimile comparison: Canberra distance
# ---------------------------------------------------------------------------


def canberra_distance_matrix(signatures: np.ndarray) -> np.ndarray:
    """Return all pairwise Canberra distances between graph signatures.

    SciPy defines terms with a zero denominator as zero, which is the standard convention when both coordinates are zero.
    """

    log("Computing pairwise Canberra-distance matrix...")
    condensed = pdist(signatures, metric="canberra")
    distances = squareform(condensed)
    np.fill_diagonal(distances, 0.0)

    if not np.all(np.isfinite(distances)):
        raise RuntimeError("The Canberra-distance matrix contains NaN or infinity.")

    log("Finished computing Canberra distances.")
    return distances


# ---------------------------------------------------------------------------
# Clustering and representative selection
# ---------------------------------------------------------------------------


def make_agglomerative_model(k: int) -> AgglomerativeClustering:
    """Create a version-compatible average-linkage clustering model."""

    try:
        return AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        # Compatibility with older scikit-learn releases.
        return AgglomerativeClustering(
            n_clusters=k,
            affinity="precomputed",
            linkage="average",
        )


def total_within_cluster_distance(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Sum upper-triangle pairwise distances within all clusters."""

    total = 0.0
    for cluster_id in np.unique(labels):
        indices = np.flatnonzero(labels == cluster_id)
        if indices.size < 2:
            continue
        block = distance_matrix[np.ix_(indices, indices)]
        total += float(np.sum(np.triu(block, k=1)))
    return total


def determine_optimal_k(
    distance_matrix: np.ndarray,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
) -> Tuple[int, List[KResult]]:
    """Choose k using silhouette score on Canberra distances."""

    number_of_graphs = distance_matrix.shape[0]
    maximum_valid_k = min(max_k, number_of_graphs - 1)

    if min_k > maximum_valid_k:
        raise ValueError(
            f"Cannot evaluate k={min_k}..{max_k} with only "
            f"{number_of_graphs} graphs."
        )

    log("Determining optimal number of clusters (k)...")
    log(f"{'k':<5} {'Silhouette':<15} {'Within-cluster distance'}")

    results: List[KResult] = []
    best_k: int | None = None
    best_silhouette = -np.inf

    for k in range(min_k, maximum_valid_k + 1):
        model = make_agglomerative_model(k)
        labels = model.fit_predict(distance_matrix)

        # A valid silhouette requires at least two labels and fewer labels
        # than observations.
        unique_labels = np.unique(labels)
        if unique_labels.size < 2 or unique_labels.size >= number_of_graphs:
            continue

        silhouette = float(
            silhouette_score(
                distance_matrix,
                labels,
                metric="precomputed",
            )
        )
        within_distance = total_within_cluster_distance(distance_matrix, labels)

        result = KResult(
            k=k,
            silhouette=silhouette,
            total_within_cluster_distance=within_distance,
        )
        results.append(result)

        log(f"{k:<5} {silhouette:<15.4f} {within_distance:.6f}")

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k

    if best_k is None:
        raise RuntimeError("No valid clustering solution was found.")

    log(
        f"Selected k = {best_k} based on the highest "
        f"Silhouette Score = {best_silhouette:.4f}"
    )
    return best_k, results


def cluster_graphs(
    distance_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[KResult]]:
    """Cluster signatures and return labels and in-cluster medoids."""

    optimal_k, k_results = determine_optimal_k(distance_matrix)
    log(f"Clustering graphs into {optimal_k} clusters...")

    model = make_agglomerative_model(optimal_k)
    labels = model.fit_predict(distance_matrix)

    representative_indices: List[int] = []

    for cluster_id in sorted(np.unique(labels)):
        member_indices = np.flatnonzero(labels == cluster_id)
        member_distances = distance_matrix[np.ix_(member_indices, member_indices)]

        # A medoid is the observed member whose total distance to all members
        # of the same cluster is minimal.
        distance_sums = np.sum(member_distances, axis=1)
        local_medoid_index = int(np.argmin(distance_sums))
        global_medoid_index = int(member_indices[local_medoid_index])
        representative_indices.append(global_medoid_index)

    return (
        labels.astype(int),
        np.asarray(representative_indices, dtype=int),
        k_results,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def describe_graph(graph: nx.Graph) -> Tuple[int, int, float, float, object]:
    """Return descriptive statistics for reporting only, not clustering."""

    number_of_nodes = graph.number_of_nodes()
    number_of_edges = graph.number_of_edges()
    average_degree = (
        sum(dict(graph.degree()).values()) / number_of_nodes
        if number_of_nodes > 0
        else 0.0
    )
    average_clustering = nx.average_clustering(graph)

    if number_of_nodes == 0:
        diameter: object = "N/A"
    else:
        components = list(nx.connected_components(graph))
        if not components:
            diameter = "N/A"
        else:
            largest_component_nodes = max(components, key=len)
            largest_component = graph.subgraph(largest_component_nodes)
            try:
                diameter = nx.diameter(largest_component)
            except nx.NetworkXError:
                diameter = "N/A"

    return (
        number_of_nodes,
        number_of_edges,
        average_degree,
        average_clustering,
        diameter,
    )


def save_signatures(filenames: Sequence[str], signatures: np.ndarray) -> None:
    with SIGNATURES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", *SIGNATURE_FEATURE_NAMES])
        for filename, signature in zip(filenames, signatures):
            writer.writerow([filename, *signature.tolist()])


def save_distance_matrix(
    filenames: Sequence[str],
    distance_matrix: np.ndarray,
) -> None:
    with DISTANCES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", *filenames])
        for filename, row in zip(filenames, distance_matrix):
            writer.writerow([filename, *row.tolist()])


def save_k_results(results: Sequence[KResult]) -> None:
    with K_RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "silhouette", "total_within_cluster_distance"])
        for result in results:
            writer.writerow(
                [
                    result.k,
                    f"{result.silhouette:.10f}",
                    f"{result.total_within_cluster_distance:.10f}",
                ]
            )


def save_cluster_members(
    filenames: Sequence[str],
    labels: np.ndarray,
) -> Dict[int, List[str]]:
    cluster_members: Dict[int, List[str]] = defaultdict(list)

    for filename, label in zip(filenames, labels):
        cluster_members[int(label)].append(filename)

    with CLUSTER_MEMBERS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster_id", "filename"])
        for cluster_id in sorted(cluster_members):
            for filename in cluster_members[cluster_id]:
                writer.writerow([cluster_id, filename])

    return cluster_members


def save_representatives(
    graphs: Sequence[nx.Graph],
    filenames: Sequence[str],
    labels: np.ndarray,
    representative_indices: np.ndarray,
    distance_matrix: np.ndarray,
) -> None:
    header = [
        "cluster",
        "file",
        "nodes",
        "edges",
        "avg_degree",
        "clustering",
        "diameter",
        "cluster_size",
        "medoid_total_canberra_distance",
    ]

    log("\nRepresentative graphs per cluster:")
    log(
        f"{'Cluster':<8} {'File':<42} {'Nodes':<8} {'Edges':<8} "
        f"{'AvgDeg':<9} {'Clust.':<9} {'Diam.':<7} {'Size':<6} "
        f"{'Medoid distance'}"
    )

    with REPRESENTATIVES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for representative_index in representative_indices:
            cluster_id = int(labels[representative_index])
            member_indices = np.flatnonzero(labels == cluster_id)
            medoid_distance = float(
                np.sum(distance_matrix[representative_index, member_indices])
            )

            graph = graphs[representative_index]
            filename = filenames[representative_index]
            nodes, edges, avg_degree, avg_clustering, diameter = describe_graph(graph)

            log(
                f"{cluster_id:<8} {filename:<42} {nodes:<8} {edges:<8} "
                f"{avg_degree:<9.2f} {avg_clustering:<9.4f} "
                f"{str(diameter):<7} {len(member_indices):<6} "
                f"{medoid_distance:.6f}"
            )

            writer.writerow(
                [
                    cluster_id,
                    filename,
                    nodes,
                    edges,
                    f"{avg_degree:.10f}",
                    f"{avg_clustering:.10f}",
                    diameter,
                    len(member_indices),
                    f"{medoid_distance:.10f}",
                ]
            )


def log_cluster_members(cluster_members: Dict[int, List[str]]) -> None:
    log("\nAll snapshots grouped by cluster:\n")
    for cluster_id in sorted(cluster_members):
        log(f"Cluster {cluster_id} ({len(cluster_members[cluster_id])} snapshots):")
        for filename in cluster_members[cluster_id]:
            log(f"  - {filename}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reset_log()
    log("Starting NetSimile pipeline...\n")

    graphs, filenames = load_graphs(DATA_DIR)
    signatures = embed_all_graphs(graphs)
    distance_matrix = canberra_distance_matrix(signatures)

    labels, representatives, k_results = cluster_graphs(distance_matrix)

    save_signatures(filenames, signatures)
    save_distance_matrix(filenames, distance_matrix)
    save_k_results(k_results)

    cluster_members = save_cluster_members(filenames, labels)
    log_cluster_members(cluster_members)

    save_representatives(
        graphs,
        filenames,
        labels,
        representatives,
        distance_matrix,
    )

    log("\nSaved outputs:")
    log(f"  Signatures:       {SIGNATURES_CSV}")
    log(f"  Distance matrix:  {DISTANCES_CSV}")
    log(f"  k diagnostics:    {K_RESULTS_CSV}")
    log(f"  Cluster members:  {CLUSTER_MEMBERS_CSV}")
    log(f"  Representatives:  {REPRESENTATIVES_CSV}")
    log(f"  Full log:          {LOG_FILE}")


if __name__ == "__main__":
    main()
