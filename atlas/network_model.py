"""ATLAS  network growth and experiment orchestration.

This module contains  the network-model functionality required by the
ATLAS short-term and long-term experiments.  Payment routing/transaction
primitives remain external dependencies supplied by ``simulator.py`` and
``utils.py`` from the original simulator repository.

Expected package layout after users obtain the upstream dependencies::

    atlas/
        attachment_strategies.py
        graph_io.py
        network_model.py
        simulator.py
        utils.py
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path

import networkx as nx
import simpy

from . import attachment_strategies
from . import graph_io as gio
from . import simulator as sim
from . import utils


SHORT_TERM_TX_COUNT = 2000
LONG_TERM_TX_COUNT = 1000
NEW_NODE_CHANNELS = 10
NEW_CHANNEL_BASE_FEE = 1000
NEW_CHANNEL_FEE_RATE = 1
NEW_CHANNEL_DELAY = 144


def _write(path: str | Path, data: str, mode: str = "w") -> None:
    """Write text to ``path``, creating its parent directory when needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open(mode, encoding="utf-8") as handle:
        handle.write(data)


def _largest_strong_component(graph: nx.DiGraph) -> nx.DiGraph:
    """Return a copy of the largest strongly connected component."""
    component = max(nx.strongly_connected_components(graph), key=len)
    return graph.subgraph(component).copy()


def _fee_view(graph, tx_amt: int):
    """Build the ATLAS fee graph and keep its largest strongly connected component."""
    fee_graph = gio.directed_fee_graph(graph, tx_amt, exclude_edges=True, multi_graph=False)
    return _largest_strong_component(fee_graph)


def _candidate_cache_path(result_file: str | Path) -> Path:
    """Return the shared short-term candidate-cache path for one graph/amount."""
    return Path(result_file).parent.parent / "candidates.txt"


def _metric_output_path(result_file: str | Path, suffix: str) -> Path:
    """Return the seed-independent short-term statistics/timing file path."""
    path = Path(result_file)
    stem = re.sub(r"_\d+$", "_", path.stem)
    return path.with_name(f"{stem}{suffix}.txt")


def _load_cached_candidates(path: Path) -> list[int]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return []
    text = path.read_text(encoding="utf-8").split()
    return [int(value) for value in text]


def _save_cached_candidates(path: Path, candidates: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(str(node) for node in candidates) + (" " if candidates else ""), encoding="utf-8")


def _random_existing_capacity(graph, tx_amt: int):
    """Sample an existing directed balance large enough for a new channel."""
    eligible = [
        capacity
        for _, _, capacity in graph.edges(data="capacity")
        if capacity is not None and capacity >= tx_amt / 2
    ]
    if not eligible:
        raise RuntimeError("No existing channel capacity is large enough for the requested transaction amount.")
    return random.choice(eligible)


def _rebalance_channels(graph) -> None:
    """Reset every bidirectional channel to an equal balance split."""
    if not graph.is_multigraph():
        edges = list(graph.edges())
        for u, v in edges:
            if graph.has_edge(v, u):
                total = graph[u][v]["capacity"] + graph[v][u]["capacity"]
                graph[u][v]["capacity"] = total / 2
                graph[v][u]["capacity"] = total / 2
        return

    for u, v, key in list(graph.edges(keys=True)):
        if graph.has_edge(v, u, key):
            total = graph[u][v][key]["capacity"] + graph[v][u][key]["capacity"]
            graph[u][v][key]["capacity"] = total / 2
            graph[v][u][key]["capacity"] = total / 2


def _append_joining_node_channels(path: str | Path, node: int, graph) -> None:
    lines = [f"-----Channels of Node {node}-----\n"]
    for neighbour in nx.neighbors(graph, node):
        lines.append(f"{node} --> {neighbour}: {graph[node][neighbour]}\n")
        if graph.has_edge(neighbour, node):
            lines.append(f"{neighbour} --> {node}: {graph[neighbour][node]}\n")
    lines.append("--------------------------------\n\n")
    _write(path, "".join(lines), mode="a")


def evaluate_graph_centralization(fee_graph, output_folder: str | Path) -> None:
    """Append the long-term structural metrics reported by ATLAS."""
    graph = _largest_strong_component(fee_graph)
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)

    print(
        f"Fee graph (largest strongly connected component) has "
        f"{graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )

    start = time.time()
    degrees = dict(nx.degree(graph))
    _write(folder / "degrees.txt", str(degrees), mode="a")
    print(f"degrees took {time.time() - start:.2f} seconds")

    start = time.time()
    betweenness = nx.betweenness_centrality(graph, weight="fee")
    _write(folder / "betweenness_centrality.txt", str(betweenness), mode="a")
    print(f"betweenness centrality took {time.time() - start:.2f} seconds")

    start = time.time()
    clustering = nx.clustering(graph)
    _write(folder / "clustering.txt", str(clustering), mode="a")
    print(f"clustering took {time.time() - start:.2f} seconds")

    start = time.time()
    transitivity = nx.transitivity(graph)
    _write(folder / "transitivity.txt", f"{transitivity}\n", mode="a")
    print(f"transitivity took {time.time() - start:.2f} seconds")

    start = time.time()
    diameter = nx.diameter(graph)
    _write(folder / "diameter.txt", f"{diameter}\n", mode="a")
    print(f"diameter took {time.time() - start:.2f} seconds")


class AtlasNetworkModel:
    """State and event logic required by the ATLAS experiments."""

    def __init__(
        self,
        env: simpy.Environment,
        graph,
        strategy: str,
        k: int,
        tx_amt: int,
        output: str | Path,
        ticks: int,
        mode: str,
        interval: int | None,
        churn_rate: float | None,
        seed_val: int,
        graph_id: int = 1,
    ):
        self.env = env
        self.graph = graph.copy()
        self.strategy = strategy
        self.k = k
        self.tx_amt = tx_amt
        self.output = Path(output)
        self.ticks = ticks
        self.mode = mode
        self.interval = interval
        self.churn_rate = churn_rate
        self.seed_val = seed_val
        self.graph_id = graph_id

        self.nodes_added_since_churn = 0
        self.total_nodes_added = 0
        self.next_node = max(self.graph.nodes()) + 1
        self.stats = {node: [0, 0, 0, 0, 0] for node in self.graph.nodes()}
        self.fee_graph = _fee_view(self.graph, self.tx_amt)

        print(
            f"Init ATLAS Network Model "
            f"(fee graph: {self.fee_graph.number_of_nodes()} nodes, "
            f"{self.fee_graph.number_of_edges()} edges)"
        )

        if mode == "short":
            self.process = env.process(self._run_short_term())
        elif mode == "long":
            evaluate_graph_centralization(self.fee_graph, self.output)
            self.process = env.process(self._run_long_term())
        else:
            raise ValueError("mode must be 'short' or 'long'")

    def _select_candidates(self, node: int, k: int) -> list[int]:
        return attachment_strategies.select_attachment_nodes(
            self.fee_graph,
            node,
            k,
            strategy=self.strategy,
            weight="fee",
        )

    def _add_short_term_node(self) -> str:
        node = self.next_node
        self.graph.add_node(node, key="joining_node", alias="new_node")
        self.fee_graph.add_node(node, key="joining_node", alias="new_node")

        reusable = {"bc_approx_10", "highest_degree", "closeness"}
        if self.strategy in reusable:
            cache = _candidate_cache_path(self.output)
            candidates = _load_cached_candidates(cache)
            if not candidates:
                start = time.time()
                candidates = self._select_candidates(node, 15)
                elapsed = time.time() - start
                _write(_metric_output_path(self.output, "times"), f"{elapsed}\n", mode="a")
                _save_cached_candidates(cache, candidates)
            connections = candidates[: self.k]
        else:
            start = time.time()
            connections = self._select_candidates(node, self.k)
            elapsed = time.time() - start
            _write(_metric_output_path(self.output, "times"), f"{elapsed}\n", mode="a")

        utils.connect(self.graph, node, connections, cap=self.tx_amt * 2000)
        self.fee_graph = _fee_view(self.graph, self.tx_amt)
        self.stats[node] = [0, 0, 0, 0, 0]
        self.next_node += 1

        print(f"Added node {node} and connected it to {connections}")
        return f"J {node:4} (opened channels to {connections})\n"

    def _add_long_term_node(self) -> str:
        node = self.next_node
        self.graph.add_node(node, key="joining_node", alias="new_node")
        self.fee_graph.add_node(node, key="joining_node", alias="new_node")

        connections = self._select_candidates(node, NEW_NODE_CHANNELS)
        for neighbour in connections:
            capacity = _random_existing_capacity(self.graph, self.tx_amt)
            attributes = {
                "capacity": capacity,
                "fee_rate": NEW_CHANNEL_FEE_RATE,
                "base_fee": NEW_CHANNEL_BASE_FEE,
                "delay": NEW_CHANNEL_DELAY,
                "disabled": False,
            }
            self.graph.add_edge(node, neighbour, **attributes)
            self.graph.add_edge(neighbour, node, **attributes)

        self.fee_graph = _fee_view(self.graph, self.tx_amt)
        self.stats[node] = [0, 0, 0, 0, 0]
        self.next_node += 1
        self.nodes_added_since_churn += 1
        self.total_nodes_added += 1

        print(f"Added node {node} and connected it to {connections}")
        return f"J {node:4} (opened channels to {connections})\n"

    def _send_short_term_transaction(self, source: int | None = None) -> str:
        source = random.choice(list(self.graph.nodes())) if source is None else source
        target = random.choice(list(set(self.graph.nodes()) - {source}))
        result = f"T {source:4} --{self.tx_amt}--> {target:4} : "
        self.stats[source][0] += 1

        if self.graph.has_edge(source, target):
            for key in self.graph[source][target]:
                edge = self.graph[source][target][key]
                if not edge["disabled"] and edge["capacity"] >= self.tx_amt:
                    failed = sim.send_a_to_b(self.graph, source, target, self.tx_amt)
                    if not failed:
                        self.stats[source][1] += 1
                        return result + "S (direct channel)\n"
                    break

        path = sim.routing(self.graph, source, target, self.tx_amt)
        if not path:
            return result + "F (no sufficiently funded route exists)\n"

        failed, fees = sim.route_tx(self.graph, path, self.tx_amt)
        if failed:
            return result + "F (temporary channel failure)\n"

        self.stats[source][1] += 1
        self.stats[source][2] += abs(fees[source])
        for index in range(1, len(fees)):
            forwarding_node = path[index][0]
            self.stats[forwarding_node][3] += 1
            self.stats[forwarding_node][4] += fees[forwarding_node]
        return result + f"S (Route: {path})\n"

    def _flush_short_term_statistics(self) -> None:
        joining_node = self.next_node - 1
        total_attempts = 0
        total_successes = 0
        lines = [
            "\n---------------STATISTICS "
            "[tx_attempts, tx_sent, fees_paid, tx_routed, fees_earned]---------------\n",
            "(statistics of nodes are not listed if they equal [0,0,0,0,0])\n",
        ]
        joining_stats = str(self.stats[joining_node]) + " "

        for node in self.graph.nodes():
            node_stats = self.stats[node]
            if node_stats != [0, 0, 0, 0, 0]:
                lines.append(f"{str(node):5}{node_stats}\n")
                total_attempts += node_stats[0]
                total_successes += node_stats[1]
                self.stats[node] = [0, 0, 0, 0, 0]

        _rebalance_channels(self.graph)
        lines.append("\n")
        _write(self.output, "".join(lines), mode="a")
        _append_joining_node_channels(self.output, joining_node, self.graph)

        joining_stats += f"{total_successes}/{total_attempts}\n"
        _write(_metric_output_path(self.output, "stats"), joining_stats, mode="a")

    def _run_short_term(self):
        while True:
            if self.env.now == 1:
                _write(self.output, self._add_short_term_node(), mode="a")

            source = None if self.env.now <= self.ticks / 2 else self.next_node - 1
            _write(self.output, self._send_short_term_transaction(source), mode="a")

            if self.env.now in (self.ticks / 2, self.ticks):
                self._flush_short_term_statistics()

            yield self.env.timeout(1)

    def _evaluate_payments(self, label: int) -> None:
        simulation_graph = self.graph.copy()
        stats = sim.simulate_transactions(
            simulation_graph,
            self.tx_amt,
            LONG_TERM_TX_COUNT,
            file=str(self.output / f"tx_{label}.txt"),
        )
        successes = sum(node_stats[1] for node_stats in stats.values())
        success_rate = successes / LONG_TERM_TX_COUNT
        average_fees = (
            sum(node_stats[2] for node_stats in stats.values()) / successes
            if successes
            else 0
        )
        _write(
            self.output / "stats.txt",
            f"Now:{label}\tSuccess Rate:{success_rate}\tAverage Fees:{average_fees}\n",
            mode="a",
        )

    def _remove_random_nodes(self, count: int) -> None:
        if count <= 0:
            return
        if self.graph.number_of_nodes() < count:
            raise RuntimeError(
                f"Not enough nodes to remove: requested {count}, "
                f"available {self.graph.number_of_nodes()}."
            )

        removed = random.sample(list(self.graph.nodes()), count)
        log_path = (
        	self.output / f"removed_nodes_graph{self.graph_id}_{self.strategy}_{self.seed_val}.txt"
        )
        lines = [f"NOW: {self.env.now}\n", "Nodes removed: "]

        for node in removed:
            if self.graph.has_node(node):
                self.graph.remove_node(node)
                lines.append(f"{node} ")

        lines.append("\n")
        _write(log_path, "".join(lines), mode="a")
        

    def _long_term_checkpoint(self) -> None:
        if self.churn_rate is not None:
            removals = int(self.nodes_added_since_churn * self.churn_rate)
            print(f"{removals} nodes will be removed")
            self.nodes_added_since_churn = 0
            self._remove_random_nodes(removals)

        self.fee_graph = _fee_view(self.graph, self.tx_amt)

        graph_file = (
            self.output
            / f"updated_graph{self.graph_id}_{self.strategy}_"
              f"{self.seed_val}_{self.total_nodes_added}.json"
        )
        gio.save_graph_json(self.graph, str(graph_file))
        evaluate_graph_centralization(self.fee_graph, self.output)
        self._evaluate_payments(int(self.env.now))

    def _run_long_term(self):
        self._evaluate_payments(0)

        while True:
            _write(self.output / "results.txt", self._add_long_term_node(), mode="a")
            if self.interval and self.total_nodes_added % self.interval == 0:
                self._long_term_checkpoint()
            yield self.env.timeout(1)


def infer_graph_id(folder: str | Path | None) -> int:
    """Infer ``graph_N`` from a result path; default to graph 1."""
    if folder is None:
        return 1
    match = re.search(r"graph_(\d+)", str(folder))
    return int(match.group(1)) if match else 1


def run_network_model(
    graph,
    tx_amt: int,
    strategy: str,
    k: int,
    folder: str | None,
    file: str | None,
    churn_rate: float | None = 0.956,
    i: int | None = None,
    c: int | None = None,
    s: int = 23,
    graph_id: int | None = None,
) -> None:
    """Run the ATLAS short-term or long-term network model.

    The signature intentionally remains compatible with the existing ATLAS
    experiment drivers. ``c`` is retained only for CLI/API compatibility.
    """
    del c

    seed_val = int(s)
    random.seed(seed_val)
    resolved_graph_id = infer_graph_id(folder) if graph_id is None else int(graph_id)

    env = simpy.Environment(initial_time=1)

    if folder is None:
        if file is None:
            raise ValueError("Short-term mode requires an output file.")
        until = SHORT_TERM_TX_COUNT
        _write(file, f"seed value: {seed_val}\n\n")
        AtlasNetworkModel(
            env=env,
            graph=graph,
            strategy=strategy,
            k=k,
            tx_amt=tx_amt,
            output=file,
            ticks=until,
            mode="short",
            interval=None,
            churn_rate=churn_rate,
            seed_val=seed_val,
            graph_id=resolved_graph_id,
        )
    else:
        if i is None or i <= 0:
            raise ValueError("Long-term mode requires a positive evaluation interval.")
        until = k
        output_folder = Path(folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        _write(output_folder / "results.txt", f"seed value: {seed_val}\n\n")
        _write(
            output_folder / "results.txt",
            f"strategy={strategy}\ntx_amt={tx_amt}\nn={k}\ninterval={i}\n\n",
            mode="a",
        )
        AtlasNetworkModel(
            env=env,
            graph=graph,
            strategy=strategy,
            k=k,
            tx_amt=tx_amt,
            output=output_folder,
            ticks=until,
            mode="long",
            interval=i,
            churn_rate=churn_rate,
            seed_val=seed_val,
            graph_id=resolved_graph_id,
        )

    while env.peek() <= until:
        env.step()
