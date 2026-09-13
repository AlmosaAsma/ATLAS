import os
from pathlib import Path
import json
import csv
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict

from karateclub import Graph2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ----------------------------
# Configuration
# ----------------------------
DATA_DIR = './data/snapshots'
MIN_K = 2
MAX_K = 10

EMBED_DIM = 128
WL_ITERATIONS = 2
EPOCHS = 100
WORKERS = 1
SEED = 42

OUTPUT_DIR = Path("./results/snapshot_selection/sensitivity/graph2vec")
SUMMARY_CSV = OUTPUT_DIR / "graph2vec_representatives.csv"
CLUSTER_MEMBERS_CSV = OUTPUT_DIR / "graph2vec_cluster_members.csv"
EMBEDDINGS_CSV = OUTPUT_DIR / "graph2vec_embeddings.csv"
K_METRICS_CSV = OUTPUT_DIR / "graph2vec_k_metrics.csv"
LOG_FILE = OUTPUT_DIR / "graph2vec_pipeline_log.txt"

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

# ----------------------------
# Logging helper
# ----------------------------
def log(message):
    print(message)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(str(message) + '\n')

# ----------------------------
# Load Graphs from JSON
# ----------------------------
def load_graphs(directory):
    graphs = []
    filenames = []

    log('Loading snapshots...')

    for filename in tqdm(sorted(os.listdir(directory))):
        if not filename.endswith('.json'):
            continue

        path = os.path.join(directory, filename)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            G = nx.Graph()

            for node in data.get('nodes', []):
                pub_key = node.get('pub_key')
                if pub_key:
                    G.add_node(pub_key)

            for edge in data.get('edges', []):
                node_1 = edge.get('node1_pub')
                node_2 = edge.get('node2_pub')

                if node_1 and node_2 and node_1 != node_2:
                    G.add_edge(node_1, node_2)

            G.remove_edges_from(nx.selfloop_edges(G))

            if G.number_of_nodes() < 2:
                log(f'Skipping {filename}: fewer than two nodes.')
                continue

            # Karate Club expects consecutively integer-labelled graphs.
            G = nx.convert_node_labels_to_integers(
                G,
                first_label=0,
                ordering='default',
            )

            graphs.append(G)
            filenames.append(filename)

        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            log(f'Skipping {filename}: {exc}')

    log(f'Loaded {len(graphs)} graphs.')
    return graphs, filenames

# ----------------------------
# Graph2Vec Embedding
# ----------------------------
def embed_all_graphs(graphs):
    """
    Fit one Graph2Vec model jointly across all snapshots.

    This produces one graph-level vector per snapshot in a shared embedding
    space. Graph2Vec uses Weisfeiler-Lehman rooted-subgraph features and a
    Doc2Vec-style unsupervised objective.
    """
    log('Generating Graph2Vec embeddings...')
    log(
        'Graph2Vec settings: '
        f'dimensions={EMBED_DIM}, '
        f'wl_iterations={WL_ITERATIONS}, '
        f'epochs={EPOCHS}, '
        f'workers={WORKERS}, '
        f'seed={SEED}'
    )

    model = Graph2Vec(
        wl_iterations=WL_ITERATIONS,
        attributed=False,
        dimensions=EMBED_DIM,
        workers=WORKERS,
        epochs=EPOCHS,
        seed=SEED,
        min_count=1,
    )

    model.fit(graphs)
    embeddings = model.get_embedding()

    if embeddings.shape[0] != len(graphs):
        raise RuntimeError(
            'The number of Graph2Vec embeddings does not match '
            'the number of loaded graphs.'
        )

    if not np.isfinite(embeddings).all():
        raise ValueError('Graph2Vec produced NaN or infinite values.')

    log(f'Generated embedding matrix with shape {embeddings.shape}.')
    return embeddings

# ----------------------------
# Save Embeddings
# ----------------------------
def save_embeddings(embeddings, filenames):
    with open(EMBEDDINGS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['filename'] + [
            f'embedding_{i}' for i in range(embeddings.shape[1])
        ]
        writer.writerow(header)

        for filename, vector in zip(filenames, embeddings):
            writer.writerow([filename] + vector.tolist())

    log(f"Saved graph embeddings to '{EMBEDDINGS_CSV}'.")

# ----------------------------
# Determine Optimal K
# ----------------------------
def determine_optimal_k(embeddings, min_k=MIN_K, max_k=MAX_K):
    if len(embeddings) < 3:
        raise ValueError('At least three graphs are required for clustering.')

    max_valid_k = min(max_k, len(embeddings) - 1)

    if min_k > max_valid_k:
        raise ValueError(
            f'Invalid k range: min_k={min_k}, max_valid_k={max_valid_k}.'
        )

    best_k = min_k
    best_score = -np.inf
    rows = []

    log('Determining optimal number of clusters (k)...')
    log(f"{'k':<5} {'Silhouette':<15} {'Davies-Bouldin':<20} {'Inertia'}")

    for k in range(min_k, max_valid_k + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=10,
        )
        labels = kmeans.fit_predict(embeddings)

        if len(np.unique(labels)) < 2:
            log(f'{k:<5} skipped: fewer than two non-empty clusters.')
            continue

        silhouette = silhouette_score(embeddings, labels)
        db_index = davies_bouldin_score(embeddings, labels)
        inertia = kmeans.inertia_

        rows.append([k, silhouette, db_index, inertia])

        log(
            f'{k:<5} '
            f'{silhouette:<15.4f} '
            f'{db_index:<20.4f} '
            f'{inertia:.2f}'
        )

        if silhouette > best_score:
            best_score = silhouette
            best_k = k

    with open(K_METRICS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'silhouette', 'davies_bouldin', 'inertia'])
        writer.writerows(rows)

    log(
        f'Selected k = {best_k} based on best '
        f'Silhouette Score = {best_score:.4f}'
    )
    log(f"Saved k-selection metrics to '{K_METRICS_CSV}'.")
    return best_k

# ----------------------------
# Select One Representative Per Cluster
# ----------------------------
def select_cluster_representatives(embeddings, labels, cluster_centers):
    """
    Select the actual snapshot nearest to each centroid, restricted to
    snapshots assigned to that cluster.

    Restricting the search prevents the same snapshot from representing
    multiple clusters and guarantees that each representative is a member
    of its corresponding cluster.
    """
    representatives = {}

    for cluster_id in sorted(np.unique(labels)):
        member_indices = np.where(labels == cluster_id)[0]

        if len(member_indices) == 0:
            raise RuntimeError(f'Cluster {cluster_id} has no members.')

        member_embeddings = embeddings[member_indices]
        centroid = cluster_centers[cluster_id]
        distances = np.linalg.norm(member_embeddings - centroid, axis=1)

        local_index = int(np.argmin(distances))
        global_index = int(member_indices[local_index])
        representatives[int(cluster_id)] = global_index

    return representatives

# ----------------------------
# Cluster Graphs
# ----------------------------
def cluster_graphs(embeddings):
    # Standardisation is fitted once across all graph embeddings.
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    optimal_k = determine_optimal_k(scaled_embeddings)

    log(f'Clustering graphs into {optimal_k} clusters...')

    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=SEED,
        n_init=10,
    )
    labels = kmeans.fit_predict(scaled_embeddings)

    representatives = select_cluster_representatives(
        scaled_embeddings,
        labels,
        kmeans.cluster_centers_,
    )

    return labels, representatives

# ----------------------------
# Graph Summary Statistics
# ----------------------------
def describe_graph(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    avg_degree = (
        sum(dict(G.degree()).values()) / num_nodes
        if num_nodes > 0 else 0.0
    )

    average_clustering = nx.average_clustering(G) if num_nodes > 1 else 0.0

    if num_nodes == 0:
        diameter = 'N/A'
    else:
        try:
            largest_cc_nodes = max(nx.connected_components(G), key=len)
            largest_cc = G.subgraph(largest_cc_nodes)
            diameter = nx.diameter(largest_cc)
        except (nx.NetworkXError, ValueError):
            diameter = 'N/A'

    return num_nodes, num_edges, avg_degree, average_clustering, diameter

# ----------------------------
# Save Cluster Membership
# ----------------------------
def save_cluster_members(labels, filenames):
    cluster_members = defaultdict(list)

    for index, label in enumerate(labels):
        cluster_members[int(label)].append(filenames[index])

    log('\nAll Snapshots Grouped by Cluster:\n')

    for cluster_id in sorted(cluster_members):
        log(f'Cluster {cluster_id}:')
        for filename in cluster_members[cluster_id]:
            log(f'  - {filename}')

    with open(CLUSTER_MEMBERS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cluster_id', 'filename'])

        for cluster_id in sorted(cluster_members):
            for filename in cluster_members[cluster_id]:
                writer.writerow([cluster_id, filename])

    return cluster_members

# ----------------------------
# Save Representative Summary
# ----------------------------
def save_representatives(representatives, graphs, filenames):
    log('\nRepresentative graphs per cluster:')
    log(
        f"{'Cluster':<8} {'File':<35} {'Nodes':<8} {'Edges':<8} "
        f"{'AvgDeg':<10} {'Clustering':<12} {'Diameter'}"
    )

    with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'cluster', 'file', 'nodes', 'edges',
            'avg_degree', 'clustering', 'diameter'
        ])

        for cluster_id in sorted(representatives):
            graph_index = representatives[cluster_id]
            G = graphs[graph_index]
            filename = filenames[graph_index]

            nodes, edges, avg_degree, clustering, diameter = describe_graph(G)

            log(
                f'{cluster_id:<8} {filename:<35} {nodes:<8} {edges:<8} '
                f'{avg_degree:<10.2f} {clustering:<12.4f} {diameter}'
            )

            writer.writerow([
                cluster_id,
                filename,
                nodes,
                edges,
                f'{avg_degree:.6f}',
                f'{clustering:.6f}',
                diameter,
            ])

# ----------------------------
# Main Pipeline
# ----------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    open(LOG_FILE, 'w', encoding='utf-8').close()
    set_seed()

    log('Starting Graph2Vec clustering pipeline...\n')

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Snapshot directory not found: '{DATA_DIR}'")

    graphs, filenames = load_graphs(DATA_DIR)

    if len(graphs) < 3:
        raise ValueError('At least three valid graph snapshots are required.')

    embeddings = embed_all_graphs(graphs)
    save_embeddings(embeddings, filenames)

    labels, representatives = cluster_graphs(embeddings)

    save_cluster_members(labels, filenames)
    save_representatives(representatives, graphs, filenames)

    log(f"\nSaved representative summaries to '{SUMMARY_CSV}'.")
    log(f"Saved all cluster members to '{CLUSTER_MEMBERS_CSV}'.")
    log(f"Full log written to '{LOG_FILE}'.")


if __name__ == '__main__':
    main()
