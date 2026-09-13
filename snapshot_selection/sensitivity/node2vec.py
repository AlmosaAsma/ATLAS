"""Node2Vec-based sensitivity-analysis pipeline for ATLAS snapshot selection."""

from __future__ import annotations

import csv
import gc
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from node2vec import Node2Vec
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path("./data/snapshots")
OUTPUT_DIR = Path("./results/snapshot_selection/sensitivity/node2vec")

SEED = 42
DIMENSIONS = 64
WALK_LENGTH = 10
NUM_WALKS = 10          # reduced from 50 to control runtime/memory
WINDOW = 5
MIN_COUNT = 1
BATCH_WORDS = 4
WORD2VEC_EPOCHS = 5
WORKERS = 1             # deterministic and safest on macOS

# Explicit Node2Vec walk-bias parameters.
P = 1.0
Q = 1.0

# Alignment requirements.
# Each independently trained Word2Vec space may be arbitrarily rotated/reflected.
# We align every snapshot to one fixed reference using shared node identities.
MIN_ALIGNMENT_NODES = 100
REFERENCE_MODE = "max_overlap"  # choose snapshot with largest total node overlap

MIN_K = 2
MAX_K = 10
KMEANS_N_INIT = 50

LOG_FILE = OUTPUT_DIR / "node2vec_pipeline_log.txt"
EMBEDDINGS_CSV = OUTPUT_DIR / "node2vec_embeddings.csv"
K_METRICS_CSV = OUTPUT_DIR / "node2vec_k_metrics.csv"
SUMMARY_CSV = OUTPUT_DIR / "node2vec_representatives.csv"
CLUSTER_MEMBERS_CSV = OUTPUT_DIR / "node2vec_cluster_members.csv"
MANIFEST_CSV = OUTPUT_DIR / "node2vec_snapshot_manifest.csv"
CONFIG_JSON = OUTPUT_DIR / "node2vec_config.json"


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def log(message: str = "") -> None:
    print(message)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(str(message) + "\n")


# -----------------------------------------------------------------------------
# Snapshot discovery / lightweight node scan
# -----------------------------------------------------------------------------
def discover_snapshot_paths() -> list[Path]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {DATA_DIR.resolve()}")
    paths = sorted(DATA_DIR.glob("*.json"))
    if len(paths) < 3:
        raise RuntimeError("At least three JSON snapshots are required.")
    return paths


def read_node_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    result = set()
    for node in data.get("nodes", []):
        node_id = node.get("pub_key") or node.get("id") or node.get("node_id")
        if node_id is not None:
            result.add(str(node_id))
    return result


def choose_reference(paths: list[Path]) -> tuple[int, list[set[str]]]:
    """Choose the snapshot with greatest aggregate node overlap."""
    log("Scanning node identities to choose a stable alignment reference...")
    node_sets: list[set[str]] = []
    valid_paths: list[Path] = []

    for path in tqdm(paths, desc="Scanning nodes"):
        try:
            nodes = read_node_ids(path)
        except Exception as exc:
            log(f"WARNING: could not scan {path.name}: {exc}")
            continue
        if len(nodes) >= 2:
            valid_paths.append(path)
            node_sets.append(nodes)

    # Replace caller list contents so every later stage uses exactly this set.
    paths[:] = valid_paths
    if len(paths) < 3:
        raise RuntimeError("Fewer than three usable snapshots after node scan.")

    if REFERENCE_MODE == "middle":
        ref_idx = len(paths) // 2
    else:
        # For each candidate, sum its intersections with every snapshot.
        # 142^2 set intersections is manageable and avoids graph construction.
        scores = []
        for i, s in enumerate(node_sets):
            score = sum(len(s & other) for other in node_sets)
            scores.append(score)
        ref_idx = int(np.argmax(scores))

    log(f"Alignment reference: {paths[ref_idx].name} ({len(node_sets[ref_idx])} nodes)")
    return ref_idx, node_sets


# -----------------------------------------------------------------------------
# Graph loader
# -----------------------------------------------------------------------------
def load_graph(path: Path) -> nx.Graph:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    graph = nx.Graph()
    for node in data.get("nodes", []):
        node_id = node.get("pub_key") or node.get("id") or node.get("node_id")
        if node_id is not None:
            graph.add_node(str(node_id))

    for edge in data.get("edges", []):
        n1 = edge.get("node1_pub") or edge.get("source") or edge.get("src")
        n2 = edge.get("node2_pub") or edge.get("target") or edge.get("dst")
        if n1 is not None and n2 is not None and str(n1) != str(n2):
            graph.add_edge(str(n1), str(n2))

    graph.remove_edges_from(nx.selfloop_edges(graph))
    if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
        raise ValueError("graph is too small or edgeless")
    return graph


# -----------------------------------------------------------------------------
# Node2Vec + alignment
# -----------------------------------------------------------------------------
def train_node2vec(graph: nx.Graph, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    n2v = Node2Vec(
        graph,
        dimensions=DIMENSIONS,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        p=P,
        q=Q,
        workers=WORKERS,
        quiet=True,
        seed=seed,
    )
    model = n2v.fit(
        window=WINDOW,
        min_count=MIN_COUNT,
        batch_words=BATCH_WORDS,
        sg=1,
        epochs=WORD2VEC_EPOCHS,
        seed=seed,
    )
    return model


def model_vectors(model, nodes: list[str]) -> np.ndarray:
    return np.asarray([model.wv[str(node)] for node in nodes], dtype=np.float64)


def normalise_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def fit_alignment(current_model, reference_model, common_nodes: list[str]) -> np.ndarray:
    """Return orthogonal matrix R such that current @ R ~= reference."""
    x = model_vectors(current_model, common_nodes)
    y = model_vectors(reference_model, common_nodes)

    # Unit-normalisation prevents arbitrary embedding-scale differences from
    # dominating the Procrustes fit while preserving angular geometry.
    x = normalise_rows(x)
    y = normalise_rows(y)
    rotation, _ = orthogonal_procrustes(x, y)
    return rotation


def graph_embedding_from_model(model, graph: nx.Graph, rotation: np.ndarray | None) -> np.ndarray:
    nodes = [str(n) for n in graph.nodes() if str(n) in model.wv]
    if not nodes:
        raise RuntimeError("No graph nodes found in Node2Vec vocabulary.")
    vectors = model_vectors(model, nodes)
    if rotation is not None:
        vectors = vectors @ rotation
    return vectors.mean(axis=0)


def embed_all_snapshots(paths: list[Path], ref_idx: int, node_sets: list[set[str]]):
    """
    Train one snapshot at a time. Only the fixed reference model remains in RAM.
    """
    log("\nTraining fixed reference Node2Vec model...")
    reference_graph = load_graph(paths[ref_idx])
    reference_model = train_node2vec(reference_graph, SEED + ref_idx)
    reference_nodes = set(str(k) for k in reference_model.wv.key_to_index.keys())
    reference_embedding = graph_embedding_from_model(reference_model, reference_graph, None)

    embeddings: list[np.ndarray | None] = [None] * len(paths)
    embeddings[ref_idx] = reference_embedding
    alignment_rows = []

    log(
        f"Node2Vec settings: dimensions={DIMENSIONS}, walk_length={WALK_LENGTH}, "
        f"num_walks={NUM_WALKS}, p={P}, q={Q}, workers={WORKERS}, seed={SEED}"
    )
    log("Training snapshots one at a time and aligning to the fixed reference...")

    for i, path in enumerate(tqdm(paths, desc="Node2Vec snapshots")):
        if i == ref_idx:
            alignment_rows.append([path.name, len(reference_nodes), "reference"])
            continue

        graph = None
        model = None
        try:
            graph = load_graph(path)
            model = train_node2vec(graph, SEED + i)
            current_nodes = set(str(k) for k in model.wv.key_to_index.keys())
            common = sorted(reference_nodes & current_nodes)

            if len(common) < MIN_ALIGNMENT_NODES:
                raise RuntimeError(
                    f"only {len(common)} nodes overlap with reference; "
                    f"minimum is {MIN_ALIGNMENT_NODES}"
                )

            rotation = fit_alignment(model, reference_model, common)
            embeddings[i] = graph_embedding_from_model(model, graph, rotation)
            alignment_rows.append([path.name, len(common), "aligned"])

        except Exception as exc:
            log(f"ERROR: {path.name}: {type(exc).__name__}: {exc}")
            alignment_rows.append([path.name, 0, f"failed: {type(exc).__name__}"])
            raise
        finally:
            # Critical for large LN snapshots: release per-snapshot Node2Vec
            # transition tables, walks, graph, and Word2Vec model immediately.
            del model
            del graph
            gc.collect()

    matrix = np.vstack([x for x in embeddings if x is not None])
    if matrix.shape[0] != len(paths):
        raise RuntimeError("Embedding count does not match snapshot count.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain NaN/inf values.")

    # release reference structures before clustering
    del reference_model
    del reference_graph
    gc.collect()

    return matrix, alignment_rows


# -----------------------------------------------------------------------------
# K-means / representative selection
# -----------------------------------------------------------------------------
def determine_optimal_k(x: np.ndarray) -> int:
    rows = []
    best = None
    max_valid_k = min(MAX_K, len(x) - 1)

    log("\nDetermining optimal number of clusters (k)...")
    log(f"{'k':<5} {'Silhouette':<15} {'Davies-Bouldin':<20} {'Inertia'}")

    for k in range(MIN_K, max_valid_k + 1):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=KMEANS_N_INIT)
        labels = km.fit_predict(x)
        sil = float(silhouette_score(x, labels))
        db = float(davies_bouldin_score(x, labels))
        inertia = float(km.inertia_)
        rows.append([k, sil, db, inertia])
        log(f"{k:<5} {sil:<15.4f} {db:<20.4f} {inertia:.4f}")
        candidate = (sil, -db)
        if best is None or candidate > best[0]:
            best = (candidate, k)

    with K_METRICS_CSV.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["k", "silhouette", "davies_bouldin", "inertia"])
        w.writerows(rows)

    log(f"Selected k = {best[1]} based primarily on Silhouette Score.")
    return int(best[1])


def cluster_embeddings(embeddings: np.ndarray):
    x = StandardScaler().fit_transform(embeddings)
    k = determine_optimal_k(x)
    km = KMeans(n_clusters=k, random_state=SEED, n_init=KMEANS_N_INIT)
    labels = km.fit_predict(x)

    reps = {}
    for cid in sorted(np.unique(labels)):
        members = np.flatnonzero(labels == cid)
        distances = np.linalg.norm(x[members] - km.cluster_centers_[cid], axis=1)
        reps[int(cid)] = int(members[np.argmin(distances)])
    return labels, reps, k


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def describe_graph(path: Path):
    graph = load_graph(path)
    n = graph.number_of_nodes()
    e = graph.number_of_edges()
    avg_degree = sum(dict(graph.degree()).values()) / n if n else 0.0
    clustering = nx.average_clustering(graph) if n > 1 else 0.0
    try:
        lcc_nodes = max(nx.connected_components(graph), key=len)
        diameter = nx.diameter(graph.subgraph(lcc_nodes))
    except Exception:
        diameter = "N/A"
    del graph
    gc.collect()
    return n, e, avg_degree, clustering, diameter


def save_outputs(paths, embeddings, labels, reps, alignment_rows):
    with EMBEDDINGS_CSV.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["filename"] + [f"embedding_{i}" for i in range(embeddings.shape[1])])
        for path, vector in zip(paths, embeddings):
            w.writerow([path.name] + vector.tolist())

    with CLUSTER_MEMBERS_CSV.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["cluster_id", "filename"])
        for path, label in zip(paths, labels):
            w.writerow([int(label), path.name])

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["filename", "alignment_shared_nodes", "status"])
        w.writerows(alignment_rows)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["cluster", "file", "nodes", "edges", "avg_degree", "clustering", "diameter"])
        log("\nRepresentative graphs per cluster:")
        for cid in sorted(reps):
            idx = reps[cid]
            n, e, avg_deg, clust, diam = describe_graph(paths[idx])
            log(f"Cluster {cid}: {paths[idx].name} | nodes={n} edges={e} avgdeg={avg_deg:.2f}")
            w.writerow([cid, paths[idx].name, n, e, f"{avg_deg:.6f}", f"{clust:.6f}", diam])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")
    set_seed()

    config = {
        "seed": SEED,
        "dimensions": DIMENSIONS,
        "walk_length": WALK_LENGTH,
        "num_walks": NUM_WALKS,
        "window": WINDOW,
        "word2vec_epochs": WORD2VEC_EPOCHS,
        "workers": WORKERS,
        "p": P,
        "q": Q,
        "alignment": "orthogonal_procrustes_to_fixed_reference",
        "reference_mode": REFERENCE_MODE,
        "min_alignment_nodes": MIN_ALIGNMENT_NODES,
        "graph_pooling": "mean_after_alignment",
        "min_k": MIN_K,
        "max_k": MAX_K,
        "kmeans_n_init": KMEANS_N_INIT,
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")

    log("Starting memory-safe aligned Node2Vec snapshot pipeline.\n")
    paths = discover_snapshot_paths()
    log(f"Discovered {len(paths)} JSON files.")
    ref_idx, node_sets = choose_reference(paths)
    log(f"Using {len(paths)} snapshots after validation scan.")

    embeddings, alignment_rows = embed_all_snapshots(paths, ref_idx, node_sets)
    labels, reps, k = cluster_embeddings(embeddings)
    save_outputs(paths, embeddings, labels, reps, alignment_rows)

    log(f"\nSelected k={k}")
    log(f"Saved embeddings to {EMBEDDINGS_CSV}")
    log(f"Saved k metrics to {K_METRICS_CSV}")
    log(f"Saved cluster members to {CLUSTER_MEMBERS_CSV}")
    log(f"Saved representatives to {SUMMARY_CSV}")
    log(f"Saved alignment manifest to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
