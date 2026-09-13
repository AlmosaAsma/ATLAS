"""Compare ATLAS snapshot-selection sensitivity-analysis outputs."""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


RESULT_ROOT = Path("./results/snapshot_selection")

METHODS = {
    "netsimile_LN": {
        "dir": RESULT_ROOT / "netsimile_LN",
        "k_metrics": "netsimile_LN_k_metrics.csv",
        "cluster_members": "netsimile_LN_cluster_members.csv",
        "representatives": "netsimile_LN_representatives.csv",
    },
    "netsimile": {
        "dir": RESULT_ROOT / "sensitivity" / "netsimile",
        "k_metrics": "netsimile_k_metrics.csv",
        "cluster_members": "netsimile_cluster_members.csv",
        "representatives": "netsimile_representatives.csv",
    },
    "graphsage": {
        "dir": RESULT_ROOT / "sensitivity" / "graphsage",
        "k_metrics": "graphsage_k_metrics.csv",
        "cluster_members": "graphsage_cluster_members.csv",
        "representatives": "graphsage_representatives.csv",
    },
    "graph2vec": {
        "dir": RESULT_ROOT / "sensitivity" / "graph2vec",
        "k_metrics": "graph2vec_k_metrics.csv",
        "cluster_members": "graph2vec_cluster_members.csv",
        "representatives": "graph2vec_representatives.csv",
    },
    "node2vec": {
        "dir": RESULT_ROOT / "sensitivity" / "node2vec",
        "k_metrics": "node2vec_k_metrics.csv",
        "cluster_members": "node2vec_cluster_members.csv",
        "representatives": "node2vec_representatives.csv",
    },
}

OUTPUT_DIR = RESULT_ROOT / "sensitivity_comparison"
SUMMARY_CSV = OUTPUT_DIR / "method_summary.csv"
AGREEMENT_CSV = OUTPUT_DIR / "partition_agreement.csv"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_best_k(method_cfg: dict):
    path = method_cfg["dir"] / method_cfg["k_metrics"]
    rows = read_csv(path)

    if not rows:
        return None, None, None

    best = max(rows, key=lambda row: float(row["silhouette"]))
    db = best.get("davies_bouldin")
    return int(best["k"]), float(best["silhouette"]), float(db) if db not in (None, "") else None


def read_representatives(method_cfg: dict) -> list[str]:
    rows = read_csv(method_cfg["dir"] / method_cfg["representatives"])
    reps = []

    for row in rows:
        for key in ("file", "representative_file"):
            if row.get(key):
                reps.append(row[key])
                break

    return reps


def read_partition(method_cfg: dict) -> dict[str, int]:
    rows = read_csv(method_cfg["dir"] / method_cfg["cluster_members"])
    result = {}

    for row in rows:
        filename = row.get("filename")
        cluster = row.get("cluster_id")
        if filename is not None and cluster is not None:
            result[filename] = int(cluster)

    return result


def write_method_summary() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "best_k", "silhouette", "davies_bouldin", "representatives"])

        for method, method_cfg in METHODS.items():
            best_k, silhouette, db = find_best_k(method_cfg)
            representatives = read_representatives(method_cfg)
            writer.writerow([
                method,
                "" if best_k is None else best_k,
                "" if silhouette is None else f"{silhouette:.10f}",
                "" if db is None else f"{db:.10f}",
                ";".join(representatives),
            ])


def write_partition_agreement() -> None:
    partitions = {method: read_partition(cfg) for method, cfg in METHODS.items()}

    with AGREEMENT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method_a", "method_b", "common_snapshots", "adjusted_rand_index", "normalized_mutual_information"])

        for method_a, method_b in combinations(METHODS, 2):
            part_a = partitions[method_a]
            part_b = partitions[method_b]
            common = sorted(set(part_a) & set(part_b))

            if len(common) < 2:
                continue

            labels_a = [part_a[name] for name in common]
            labels_b = [part_b[name] for name in common]

            writer.writerow([
                method_a,
                method_b,
                len(common),
                f"{adjusted_rand_score(labels_a, labels_b):.10f}",
                f"{normalized_mutual_info_score(labels_a, labels_b):.10f}",
            ])


def main() -> None:
    write_method_summary()
    write_partition_agreement()
    print(f"Saved method summary to {SUMMARY_CSV}")
    print(f"Saved partition agreement to {AGREEMENT_CSV}")


if __name__ == "__main__":
    main()
