"""LN-adapted NetSimile-style snapshot representation used by ATLAS."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, pairwise_distances_argmin_min, silhouette_score
from tqdm import tqdm


DATA_DIR = Path("./data/snapshots")
OUTPUT_DIR = Path("./results/snapshot_selection/netsimile_LN")

SEED = 42
MIN_K = 2
MAX_K = 10

EMBEDDINGS_CSV = OUTPUT_DIR / "netsimile_LN_embeddings.csv"
K_METRICS_CSV = OUTPUT_DIR / "netsimile_LN_k_metrics.csv"
CLUSTER_MEMBERS_CSV = OUTPUT_DIR / "netsimile_LN_cluster_members.csv"
REPRESENTATIVES_CSV = OUTPUT_DIR / "netsimile_LN_representatives.csv"
CONFIG_JSON = OUTPUT_DIR / "netsimile_LN_config.json"
LOG_FILE = OUTPUT_DIR / "netsimile_LN_pipeline_log.txt"


def log(message: str = "") -> None:
    print(message)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(str(message) + "\n")


def load_graphs(directory: Path):
    """Load all JSON snapshots as simple undirected graphs."""
    if not directory.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {directory.resolve()}")

    graphs, filenames = [], []
    log("Loading snapshots...")

    for path in tqdm(sorted(directory.glob("*.json")), desc="Loading snapshots"):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        graph = nx.Graph()

        for node in data.get("nodes", []):
            pub_key = node.get("pub_key")
            if pub_key:
                graph.add_node(str(pub_key))

        for edge in data.get("edges", []):
            node1 = edge.get("node1_pub")
            node2 = edge.get("node2_pub")
            if node1 and node2 and node1 != node2:
                graph.add_edge(str(node1), str(node2))

        graph.remove_edges_from(nx.selfloop_edges(graph))

        if graph.number_of_nodes() >= 2:
            graphs.append(graph)
            filenames.append(path.name)

    if len(graphs) < 3:
        raise RuntimeError("At least three valid snapshots are required.")

    log(f"Loaded {len(graphs)} graphs.")
    return graphs, filenames


def compute_features(graph: nx.Graph) -> np.ndarray:
    """Return the 21-dimensional LN graph representation."""
    degrees = [degree for _, degree in graph.degree()]
    clustering = list(nx.clustering(graph).values())

    if graph.number_of_nodes() > 1:
        betweenness = list(
            nx.betweenness_centrality(
                graph,
                k=min(50, graph.number_of_nodes()),
                seed=SEED,
            ).values()
        )
    else:
        betweenness = [0.0]

    def stats(values):
        values = np.asarray(values, dtype=float)
        return [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
            float(np.median(values)),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 75)),
        ]

    return np.asarray(stats(degrees) + stats(clustering) + stats(betweenness), dtype=float)


def embed_all_graphs(graphs) -> np.ndarray:
    log("Computing LN-adapted NetSimile representations...")
    embeddings = np.vstack([compute_features(graph) for graph in tqdm(graphs, desc="Representations")])
    log(f"Embedding matrix shape: {embeddings.shape}")
    return embeddings


def determine_optimal_k(embeddings: np.ndarray):
    rows = []
    best_k = MIN_K
    best_score = -np.inf

    log("Determining optimal number of clusters (k)...")
    log(f"{'k':<5} {'Silhouette':<15} {'Davies-Bouldin':<20} {'Inertia'}")

    for k in range(MIN_K, min(MAX_K, len(embeddings) - 1) + 1):
        kmeans = KMeans(n_clusters=k, random_state=SEED)
        labels = kmeans.fit_predict(embeddings)

        silhouette = float(silhouette_score(embeddings, labels))
        db_index = float(davies_bouldin_score(embeddings, labels))
        inertia = float(kmeans.inertia_)
        rows.append([k, silhouette, db_index, inertia])

        log(f"{k:<5} {silhouette:<15.4f} {db_index:<20.4f} {inertia:.2f}")

        if silhouette > best_score:
            best_score = silhouette
            best_k = k

    with K_METRICS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "silhouette", "davies_bouldin", "inertia"])
        writer.writerows(rows)

    log(f"Selected k = {best_k} based on the highest Silhouette score = {best_score:.4f}")
    return best_k


def cluster_graphs(embeddings: np.ndarray):
    optimal_k = determine_optimal_k(embeddings)
    kmeans = KMeans(n_clusters=optimal_k, random_state=SEED)
    labels = kmeans.fit_predict(embeddings)
    representatives, distances = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
    return labels, representatives, distances, optimal_k


def describe_graph(graph: nx.Graph):
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    avg_degree = sum(dict(graph.degree()).values()) / nodes if nodes else 0.0
    avg_clustering = nx.average_clustering(graph) if nodes > 1 else 0.0

    try:
        largest = graph.subgraph(max(nx.connected_components(graph), key=len))
        diameter = nx.diameter(largest)
    except (ValueError, nx.NetworkXError):
        diameter = "N/A"

    return nodes, edges, avg_degree, avg_clustering, diameter


def save_outputs(graphs, filenames, embeddings, labels, representatives, distances) -> None:
    with EMBEDDINGS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename"] + [f"embedding_{i}" for i in range(embeddings.shape[1])])
        for filename, vector in zip(filenames, embeddings):
            writer.writerow([filename] + vector.tolist())

    cluster_members = defaultdict(list)

    with CLUSTER_MEMBERS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster_id", "filename"])
        for filename, label in zip(filenames, labels):
            cluster_members[int(label)].append(filename)
            writer.writerow([int(label), filename])

    with REPRESENTATIVES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster", "file", "nodes", "edges", "avg_degree", "clustering", "diameter", "distance_to_centroid"])

        for cluster_id, index in enumerate(representatives):
            nodes, edges, avg_degree, avg_clustering, diameter = describe_graph(graphs[index])
            writer.writerow([
                cluster_id,
                filenames[index],
                nodes,
                edges,
                f"{avg_degree:.6f}",
                f"{avg_clustering:.6f}",
                diameter,
                f"{float(distances[cluster_id]):.10f}",
            ])
            log(f"Cluster {cluster_id}: {filenames[index]}")

    log(f"Saved embeddings to {EMBEDDINGS_CSV}")
    log(f"Saved cluster members to {CLUSTER_MEMBERS_CSV}")
    log(f"Saved representatives to {REPRESENTATIVES_CSV}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    config = {
        "method": "LN-adapted NetSimile-style representation",
        "seed": SEED,
        "features": ["degree", "clustering_coefficient", "approximate_betweenness_centrality"],
        "statistics": ["mean", "standard_deviation", "minimum", "maximum", "median", "25th_percentile", "75th_percentile"],
        "betweenness_sample_size": 50,
        "embedding_dimension": 21,
        "clustering": "k-means",
        "min_k": MIN_K,
        "max_k": MAX_K,
        "selection_metric": "silhouette_score",
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")

    graphs, filenames = load_graphs(DATA_DIR)
    embeddings = embed_all_graphs(graphs)
    labels, representatives, distances, optimal_k = cluster_graphs(embeddings)
    save_outputs(graphs, filenames, embeddings, labels, representatives, distances)
    log(f"Selected k={optimal_k}")


if __name__ == "__main__":
    main()
