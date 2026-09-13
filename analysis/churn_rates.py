"""Calculate Lightning Network node-arrival and node-departure rates from snapshot sequences."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path


DATE_PATTERN = re.compile(r"(\d{4})[_-](\d{2})[_-](\d{2})")


def extract_date_from_filename(path: Path) -> str:
    """Extract a YYYY-MM-DD date from a snapshot filename."""
    match = DATE_PATTERN.search(path.name)

    if not match:
        raise ValueError(f"Could not extract a date from filename: {path.name}")

    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def load_snapshot(path: Path) -> dict:
    """Load one Lightning Network JSON snapshot."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_node_ids(snapshot: dict) -> set[str]:
    """Return unique node public keys from a snapshot."""
    return {str(node["pub_key"]) for node in snapshot.get("nodes", []) if node.get("pub_key") is not None}


def extract_channels(snapshot: dict) -> set[tuple[str, str]]:
    """Return unique undirected channel endpoint pairs."""
    channels = set()

    for channel in snapshot.get("edges", []):
        node1 = channel.get("node1_pub")
        node2 = channel.get("node2_pub")

        if node1 is None or node2 is None:
            continue

        channels.add(tuple(sorted((str(node1), str(node2)))))

    return channels


def discover_snapshots(snapshot_dir: Path) -> list[Path]:
    """Find dated JSON snapshots and return them in chronological order."""
    paths = []

    for path in snapshot_dir.glob("*.json"):
        try:
            date = extract_date_from_filename(path)
        except ValueError:
            continue
        paths.append((datetime.strptime(date, "%Y-%m-%d"), path))

    return [path for _, path in sorted(paths)]


def analyse_snapshots(snapshot_paths: list[Path]) -> list[dict]:
    """
    Compare consecutive snapshots.

    Arrivals are V_i minus V_(i-1), and departures are V_(i-1) minus V_i.
    Both rates use the number of nodes in the current snapshot as denominator,
    matching the ATLAS churn calculation.
    """
    if len(snapshot_paths) < 2:
        raise ValueError("At least two dated snapshots are required.")

    loaded = [(path, load_snapshot(path)) for path in snapshot_paths]
    rows = []

    for index in range(1, len(loaded)):
        previous_path, previous_snapshot = loaded[index - 1]
        current_path, current_snapshot = loaded[index]

        previous_nodes = extract_node_ids(previous_snapshot)
        current_nodes = extract_node_ids(current_snapshot)
        previous_channels = extract_channels(previous_snapshot)
        current_channels = extract_channels(current_snapshot)

        arrivals = current_nodes - previous_nodes
        departures = previous_nodes - current_nodes
        opened_channels = current_channels - previous_channels
        closed_channels = previous_channels - current_channels

        previous_total_nodes = len(previous_nodes)
        current_total_nodes = len(current_nodes)
        denominator = current_total_nodes

        arrival_rate = (len(arrivals) / denominator * 100.0) if denominator else 0.0
        departure_rate = (len(departures) / denominator * 100.0) if denominator else 0.0

        rows.append({
            "previous_date": extract_date_from_filename(previous_path),
            "current_date": extract_date_from_filename(current_path),
            "previous_total_nodes": previous_total_nodes,
            "current_total_nodes": current_total_nodes,
            "arrivals": len(arrivals),
            "departures": len(departures),
            "arrival_rate_percent": arrival_rate,
            "departure_rate_percent": departure_rate,
            "opened_channels": len(opened_channels),
            "closed_channels": len(closed_channels),
            "current_total_channels": len(current_channels),
        })

    return rows


def calculate_summary(rows: list[dict]) -> dict:
    """Calculate mean arrival/departure rates and the departure-to-arrival factor."""
    if not rows:
        return {
            "intervals": 0,
            "average_arrival_rate_percent": 0.0,
            "average_departure_rate_percent": 0.0,
            "departure_to_arrival_factor": 0.0,
        }

    average_arrival = sum(row["arrival_rate_percent"] for row in rows) / len(rows)
    average_departure = sum(row["departure_rate_percent"] for row in rows) / len(rows)
    factor = average_departure / average_arrival if average_arrival else 0.0

    return {
        "intervals": len(rows),
        "average_arrival_rate_percent": average_arrival,
        "average_departure_rate_percent": average_departure,
        "departure_to_arrival_factor": factor,
    }


def write_interval_csv(rows: list[dict], output_path: Path) -> None:
    """Write interval dates, counts, denominators, and rates for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "previous_date",
        "current_date",
        "previous_total_nodes",
        "current_total_nodes",
        "arrivals",
        "departures",
        "arrival_rate_percent",
        "departure_rate_percent",
        "opened_channels",
        "closed_channels",
        "current_total_channels",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            output = dict(row)
            output["arrival_rate_percent"] = f'{row["arrival_rate_percent"]:.6f}'
            output["departure_rate_percent"] = f'{row["departure_rate_percent"]:.6f}'
            writer.writerow(output)


def write_summary(summary: dict, output_path: Path) -> None:
    """Write the average churn statistics used by the long-term experiment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f'Number of snapshot intervals: {summary["intervals"]}\n')
        handle.write(f'Average node arrival rate: {summary["average_arrival_rate_percent"]:.6f}%\n')
        handle.write(f'Average node departure rate: {summary["average_departure_rate_percent"]:.6f}%\n')
        handle.write(f'Departure-to-arrival factor: {summary["departure_to_arrival_factor"]:.6f}\n')


def run(snapshot_dir: Path, output_dir: Path, label: str) -> dict:
    """Calculate churn for one snapshot sequence and write reproducible outputs."""
    snapshot_paths = discover_snapshots(snapshot_dir)

    if len(snapshot_paths) < 2:
        raise RuntimeError(f"At least two dated JSON snapshots are required in {snapshot_dir}")

    rows = analyse_snapshots(snapshot_paths)
    summary = calculate_summary(rows)

    write_interval_csv(rows, output_dir / f"{label}_churn_intervals.csv")
    write_summary(summary, output_dir / f"{label}_churn_summary.txt")

    print(f"Snapshots: {len(snapshot_paths)}")
    print(f"Intervals: {summary['intervals']}")
    print(f"Average arrival rate: {summary['average_arrival_rate_percent']:.6f}%")
    print(f"Average departure rate: {summary['average_departure_rate_percent']:.6f}%")
    print(f"Departure-to-arrival factor: {summary['departure_to_arrival_factor']:.6f}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate ATLAS churn rates from a sequence of LN snapshots.")
    parser.add_argument("snapshot_dir", type=Path, help="Directory containing the dated JSON snapshots for one period.")
    parser.add_argument("--output-dir", type=Path, default=Path("./results/churn"), help="Directory for churn interval and summary files.")
    parser.add_argument("--label", default="churn", help="Prefix identifying the dataset/period, e.g. 2023 or 2025.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.snapshot_dir, args.output_dir, args.label)
