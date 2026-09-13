"""Plotting utilities for ATLAS short-term and long-term experiment results."""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import argparse


ATTACHMENT_STRATEGIES = ( "highest_degree", "random", "bc_approx_10", "k-center_gon_deg", "closeness")

DISPLAY_NAMES = {
    "highest_degree": "Highest Degree",
    "random": "Random",
    "bc_approx_10": "Betweenness",
    "k-center_gon_deg": "K-Center",
    "closeness": "Closeness",
}

MARKERS = {
    "highest_degree": "d",
    "random": "*",
    "bc_approx_10": "s",
    "k-center_gon_deg": "v",
    "closeness": ".",
}

color = {
    'random': '#228833',
    'highest_degree': '#DDAA33',
    'bc_approx_10': '#EE6677',
    'k-center_gon_deg': '#1965B0',
    'closeness': '#FFDD44',
}

LONG_TERM_ROOT = Path("./results/long_term")
SHORT_TERM_ROOT = Path("./results/short_term")
FIGURE_ROOT = Path("./results/figures")

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


def ensure_figure_root() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


def available_seed_ids(strategy: str, graph_id: int) -> list[int]:
    """Return all available long-term seed IDs for one strategy and graph."""
    graph_folder = LONG_TERM_ROOT / strategy / f"graph_{graph_id}"
    seed_ids = []

    if not graph_folder.exists():
        return seed_ids

    for folder in graph_folder.glob("seed_*"):
        if not folder.is_dir():
            continue
        try:
            seed_ids.append(int(folder.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue

    return sorted(seed_ids)


def long_term_file(strategy: str, graph_id: int, seed_id: int, filename: str) -> Path:
    """Return a path to one seeded long-term result file."""
    return LONG_TERM_ROOT / strategy / f"graph_{graph_id}" / f"seed_{seed_id}" / filename


def short_term_folder(strategy: str, graph_id: int, amount: int, k: int) -> Path:
    """Return a path to one short-term result folder."""
    return SHORT_TERM_ROOT / strategy / f"graph_{graph_id}" / f"amt_{amount}" / f"k_{k}"


def mean_ci_by_checkpoint(series_by_seed: list[list[float]], confidence: float = 0.95) -> tuple[list[float], list[float], list[float]]:
    """Return checkpoint means and lower/upper confidence bounds."""
    if not series_by_seed:
        return [], [], []

    max_len = max(len(series) for series in series_by_seed)
    means, lower, upper = [], [], []

    for index in range(max_len):
        values = np.asarray([series[index] for series in series_by_seed if index < len(series)], dtype=float)
        mean = float(np.mean(values))

        if len(values) < 2:
            half_width = 0.0
        else:
            standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            critical_value = float(t.ppf(0.5 + confidence / 2, df=len(values) - 1))
            half_width = critical_value * standard_error

        means.append(mean)
        lower.append(mean - half_width)
        upper.append(mean + half_width)

    return means, lower, upper


def gini_coefficient(values) -> float:
    """Return the Gini coefficient of a non-negative sequence."""
    array = np.asarray(list(values), dtype=float)

    if array.size == 0 or np.sum(array) == 0:
        return 0.0

    if np.any(array < 0):
        array = array - np.min(array)

    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def read_dict_series(path: Path) -> list[dict]:
    """Read concatenated Python dictionaries written by the long-term simulator."""
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    dictionaries = []
    start = 0
    depth = 0

    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:index + 1]
                try:
                    value = ast.literal_eval(chunk)
                except (SyntaxError, ValueError):
                    continue
                if isinstance(value, dict):
                    dictionaries.append(value)

    return dictionaries


def read_scalar_series(path: Path) -> list[float]:
    """Read one numeric value per line."""
    if not path.exists():
        return []

    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue

    return values


def read_long_term_stats(path: Path, feature: str, tx_amount: int) -> list[float]:
    """Read success-rate or average-fee values from a long-term stats file."""
    if feature not in {"success_rate", "fees"}:
        raise ValueError("feature must be 'success_rate' or 'fees'")

    if not path.exists():
        return []

    values = []

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue

        try:
            success_rate = float(parts[1].split(":", 1)[1])
            average_fees = float(parts[2].split(":", 1)[1])
        except (IndexError, ValueError):
            continue

        if feature == "success_rate":
            values.append(success_rate * 100.0)
        else:
            values.append(max(0.0, average_fees / tx_amount * 100.0))

    return values


def plot_with_ci(ax, x, means, lower, upper, strategy, bounded=None) -> None:
    """Plot a mean series with 95% confidence intervals."""
    if not means:
        return

    x_values = x[:len(means)]
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)

    if bounded is not None:
        lower_values = np.clip(lower_values, bounded[0], bounded[1])
        upper_values = np.clip(upper_values, bounded[0], bounded[1])

    means_array = np.asarray(means, dtype=float)
    yerr = np.vstack([means_array - lower_values, upper_values - means_array])

    ax.errorbar(x_values, means_array, yerr=yerr, fmt="-", color=color[strategy], marker=MARKERS[strategy], label=DISPLAY_NAMES[strategy], capsize=3, elinewidth=1)


def plot_long_term_gini(metric: str, strategies, graph_id: int, x, ax=None):
    """Plot degree, betweenness, or clustering Gini across long-term seeds."""
    file_map = {
        "degree": "degrees.txt",
        "betweenness": "betweenness_centrality.txt",
        "clustering": "clustering.txt",
    }

    if metric not in file_map:
        raise ValueError("metric must be degree, betweenness, or clustering")

    if ax is None:
        _, ax = plt.subplots()

    ylabel = {
        "degree": "Gini Coefficient (Degree)",
        "betweenness": "Gini Coefficient (Betweenness)",
        "clustering": "Gini Coefficient (Clustering)",
    }[metric]

    ax.set(xlabel="Nodes Added", ylabel=ylabel)

    for strategy in strategies:
        seed_series = []

        for seed_id in available_seed_ids(strategy, graph_id):
            dictionaries = read_dict_series(long_term_file(strategy, graph_id, seed_id, file_map[metric]))
            values = [gini_coefficient(data.values()) for data in dictionaries if data]
            if values:
                seed_series.append(values)

        means, lower, upper = mean_ci_by_checkpoint(seed_series)
        plot_with_ci(ax, x, means, lower, upper, strategy, bounded=(0.0, 1.0))

    ax.legend(loc="best")
    ax.grid(axis="y")
    return ax


def plot_long_term_scalar(metric: str, strategies, graph_id: int, x, ax=None):
    """Plot diameter or transitivity across long-term seeds."""
    if metric not in {"diameter", "transitivity"}:
        raise ValueError("metric must be diameter or transitivity")

    if ax is None:
        _, ax = plt.subplots()

    ylabel = "Diameter" if metric == "diameter" else "Transitivity"
    ax.set(xlabel="Nodes Added", ylabel=ylabel)

    for strategy in strategies:
        seed_series = []

        for seed_id in available_seed_ids(strategy, graph_id):
            values = read_scalar_series(long_term_file(strategy, graph_id, seed_id, f"{metric}.txt"))
            if values:
                seed_series.append(values)

        means, lower, upper = mean_ci_by_checkpoint(seed_series)
        bounds = (0.0, 1.0) if metric == "transitivity" else (0.0, float("inf"))
        plot_with_ci(ax, x, means, lower, upper, strategy, bounded=bounds)

    ax.legend(loc="best")
    ax.grid(axis="y")
    return ax


def plot_long_term_performance(feature: str, strategies, graph_id: int, tx_amount: int, x, ax=None):
    """Plot payment success rate or average fees across long-term seeds."""
    if feature not in {"success_rate", "fees"}:
        raise ValueError("feature must be success_rate or fees")

    if ax is None:
        _, ax = plt.subplots()

    if feature == "success_rate":
        ax.set(xlabel="Nodes Added", ylabel="Success Rate (%)")
        bounds = (0.0, 100.0)
    else:
        ax.set(xlabel="Nodes Added", ylabel="Average Fees (% of payment amount)")
        bounds = (0.0, float("inf"))

    for strategy in strategies:
        seed_series = []

        for seed_id in available_seed_ids(strategy, graph_id):
            values = read_long_term_stats(long_term_file(strategy, graph_id, seed_id, "stats.txt"), feature, tx_amount)
            if values:
                seed_series.append(values)

        means, lower, upper = mean_ci_by_checkpoint(seed_series)
        plot_with_ci(ax, x, means, lower, upper, strategy, bounded=bounds)

    if feature == "success_rate":
        ax.set_ylim(0, 100)

    ax.legend(loc="best")
    ax.grid(axis="y")
    return ax


def read_run_configuration(graph_id: int, strategies=ATTACHMENT_STRATEGIES) -> tuple[int, int, int]:
    """Read transaction amount, nodes added, and checkpoint interval from results.txt."""
    for strategy in strategies:
        for seed_id in available_seed_ids(strategy, graph_id):
            path = long_term_file(strategy, graph_id, seed_id, "results.txt")
            if not path.exists():
                continue

            tx_amount = nodes_to_add = interval = None

            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("tx_amt="):
                    tx_amount = int(line.split("=", 1)[1])
                elif line.startswith("n="):
                    nodes_to_add = int(line.split("=", 1)[1])
                elif line.startswith("interval="):
                    interval = int(line.split("=", 1)[1])

            if None not in (tx_amount, nodes_to_add, interval):
                return tx_amount, nodes_to_add, interval

    raise FileNotFoundError(f"No complete long-term results.txt found for graph {graph_id}")


def save_long_term_figures(graph_id: int, strategies=ATTACHMENT_STRATEGIES) -> None:
    """Generate the main long-term ATLAS figures for one representative graph."""
    ensure_figure_root()
    tx_amount, nodes_to_add, interval = read_run_configuration(graph_id, strategies)
    x = [step * interval for step in range(nodes_to_add // interval + 1)]

    figures = [
        ("degree_gini", lambda ax: plot_long_term_gini("degree", strategies, graph_id, x, ax)),
        ("betweenness_gini", lambda ax: plot_long_term_gini("betweenness", strategies, graph_id, x, ax)),
        ("clustering_gini", lambda ax: plot_long_term_gini("clustering", strategies, graph_id, x, ax)),
        ("diameter", lambda ax: plot_long_term_scalar("diameter", strategies, graph_id, x, ax)),
        ("transitivity", lambda ax: plot_long_term_scalar("transitivity", strategies, graph_id, x, ax)),
        ("success_rate", lambda ax: plot_long_term_performance("success_rate", strategies, graph_id, tx_amount, x, ax)),
        ("fees", lambda ax: plot_long_term_performance("fees", strategies, graph_id, tx_amount, x, ax)),
    ]

    for figure_name, plotter in figures:
        fig, ax = plt.subplots()
        plotter(ax)
        fig.tight_layout()
        fig.savefig(FIGURE_ROOT / f"graph_{graph_id}_{figure_name}.pdf", bbox_inches="tight")
        plt.close(fig)

def retrieve_stats(strategies, graph_id, amt=100, routing=False):
    stats = {}
    for strategy in strategies:
        stats_k = {}
        for k in range(1, 16):
            stats_k[k] = []
            file = f'./results/short_term/{strategy}/graph_{graph_id}/amt_{amt}/k_{k}/{strategy}_k{k}_stats.txt'
            successes, test = 0, 0
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    i = 0
                    seed_count = 0 #####
                    for line in f:
                        line = line.strip().split('] ')
                        if not routing and line[0][:5] == '[1000':
                            stats_k[k].append((int(line[0].split(',')[1]), float(line[0].split(',')[2])))
                            seed_count += 1 #####
                        elif routing and line[0][:5] != '[1000':
                            stats_k[k].append((int(line[0].split(',')[3]), float(line[0].split(',')[4])))
                            successes += int(line[1].split('/')[0])
                            test += int(line[1].split('/')[1])
                            seed_count += 1 #####
                        i += 1
                        if seed_count >= 10: ##### seed value (only take 10 seeds)
                            break
                    if routing:
                        stats_k[k].append(successes)
            except FileNotFoundError:
                print(f"File not found: {file}")
        stats[strategy] = stats_k
    return stats

def fill_y(data, strategy, tx_amt, fees, routing, times):
    """
        helper function for 'plot()' filling the 'y' vector
        now also returns standard deviation per node
    """
    y = []
    y_std = []  # new list for standard deviation

    if times:
        y = data[strategy]
        y_std = [[0, 0]] * len(y)  # optional, no std for execution times
    else:
        for j in data[strategy].values():
            if not j:
                if routing:
                    y.append(0)
                    y_std.append([0,0])
                else:
                    y.append(np.NaN)
                    y_std.append([0,0])
            else:
                res_list = []
                if routing:  # -> j[len(j)-1] stores only one value (number of successful tx)
                    if fees:
                        res_list = [i[1] for i in j[:-1]]
                    else:
                        res_list = [(i[0]/j[-1])*100 for i in j[:-1]]
                         #print("{} routed {} ({}) of {} successful transaction".format(strategy, sum([i[0] for i in j[:-1]]), res, j[len(j) - 1]))
                else:
                    if fees:
                        res_list = [(i[1]/i[0])/tx_amt*100 for i in j if i[0] != 0]
                    else:
                        res_list = [i[0]/1000 for i in j]
                #y.append(np.mean(res_list))
                #y_std.append(np.std(res_list))  # standard deviation added
                center = np.mean(res_list)
                lower = np.percentile(res_list, 0)  # Set to 0 and 100 to include all values but can be changed to 25 and 75 
                upper = np.percentile(res_list, 100)
                
                
                err_low = center - lower
                err_high = upper - center
                # fix negatives safely
                if err_low < 0:
                	err_low = 0
                if err_high < 0:
                	err_high = 0
                
                y.append(center)
                y_std.append([err_low, err_high])
            

    #return y, y_std
    return y, np.array(y_std).T

def plot(data, tx_amt, fees=False, routing=False, times=False, baseline=None, ax=None, show_error=True):  # NEW
    """
        plots the data as a line graph
    """
    filename = ""
    if ax is None:
        fig, ax = plt.subplots()
        # plt.xlabel('Nodes')
    if times:
        ax.set(xlabel='Nodes', ylabel='Seconds')
        plt.title('Execution time (in Seconds)')
        filename += "exe_time"
    elif fees:
        if routing:
            ax.set(xlabel='Nodes', ylabel='Fees (in Satoshis)')
            filename += "routing_fees"
        else:
            ax.set(xlabel='Nodes', ylabel='Fees (in %)')
            filename += "fees"
    else:
        if routing:
            ax.set(xlabel='Nodes', ylabel='Routed Transactions (in %)')
            filename += "routing_success"
        else:
            ax.set(xlabel='Nodes', ylabel='Success Rate')
            filename += "success_rate"

    x = [str(i) for i in range(1, 16)]
    idx = 0
    legend = []
    for strategy in data:
        y, y_std = fill_y(data, strategy, tx_amt, fees, routing, times)  # UPDATED

        # NEW: plot with error bars if show_error
        if show_error:
            leg = ax.errorbar(x, y, yerr=y_std, fmt='-', color=color[strategy], marker=MARKERS[strategy], label=DISPLAY_NAMES[strategy], capsize=3, elinewidth=1)

        else:
            leg = ax.plot(x, y, color[strategy], marker=MARKERS[strategy], label=DISPLAY_NAMES[strategy])

        legend.append(leg[0])
        idx += 1

    if baseline:
        ax.axhline(baseline, color='gray', linestyle='--', label='Network Average')

    plt.grid(axis='y')
    ax.legend(loc='best', ncol=1)

    return legend

def plot_short_term(strategies, tx_amt, baseline, plots, graph_id):
    if plots == 'fees_routing':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        d1 = retrieve_stats(strategies, graph_id, amt=tx_amt, routing=False)
        a = plot(d1, tx_amt, fees=True, baseline=baseline, ax=ax1)
        ax1.yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
        d2 = retrieve_stats(strategies, graph_id, amt=tx_amt, routing=True)
        b = plot(d2, tx_amt, fees=False, routing=True, ax=ax2)
        plt.subplots_adjust(wspace=0.9)
        ax1.grid(axis='y')
        ax2.grid(axis='y')
    elif plots == 'success_rate_bars':
        fig, ax1 = plt.subplots(figsize=(5, 4))
        d1 = retrieve_stats(strategies, graph_id, amt=tx_amt, routing=False)
        plot(d1, tx_amt, fees=False, baseline=baseline, ax=ax1)
        ax1.grid(axis='y')
    else:
        print("ERROR! Unknown 'plots' name")
        return
    plt.tight_layout()
    ensure_figure_root()
    plt.savefig(FIGURE_ROOT / f'graph_{graph_id}_{plots}.pdf', bbox_inches='tight')
    plt.show()
    print("test")

def plot_success_rate_node_removal(attack_type, strategies, graph_nodes, graph_id, result_version="7", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.set(xlabel='Ratio of Nodes Removed', ylabel='Success Rate (in %)')

    legend = []

    for strategy in strategies:
        graph_folder = f'./results/robustness/{graph_nodes}_nodes_{result_version}/{attack_type}_removed/{strategy}/graph_{graph_id}'
        pct_folders = sorted([f for f in os.listdir(graph_folder) if os.path.isdir(os.path.join(graph_folder, f))])

        x_percent = []
        avg_success = []
        std_success = []  # store standard deviation

        for pct in pct_folders:
            stats_file = os.path.join(graph_folder, pct, 'stats.txt')
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    next(f)
                    success_values = []
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) < 3:
                            continue
                        _, sr, _ = parts
                        success_values.append(float(sr) * 100)

                    if success_values:
                        avg_success.append(np.mean(success_values))
                        std_success.append(np.std(success_values))  #  std dev
                        x_percent.append(float(pct))
            except FileNotFoundError:
                print(f"File not found: {stats_file}")
                continue

        #  Error bars instead of plain line
        leg = ax.errorbar(x_percent, avg_success, yerr=std_success, color=color[strategy], marker=MARKERS[strategy], label=DISPLAY_NAMES[strategy], capsize=3, elinewidth=1)
        legend.append(leg)

    ax.legend(loc='best')
    plt.grid(axis='y')
    plt.savefig(FIGURE_ROOT / f'graph_{graph_id}_success_rate_node_removal.pdf', bbox_inches='tight')
    ##tikz_save(f'./figures/graph_{graph_id}_success_rate_node_removal.tex')
    plt.show()
    return legend

def plot_fees_node_removal(attack_type, strategies, graph_nodes, graph_id, result_version="7", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.set(xlabel='Ratio of Nodes Removed', ylabel='Average Transaction Fees')

    legend = []

    for strategy in strategies:
        graph_folder = f'./results/robustness/{graph_nodes}_nodes_{result_version}/{attack_type}_removed/{strategy}/graph_{graph_id}'
        pct_folders = sorted([f for f in os.listdir(graph_folder) if os.path.isdir(os.path.join(graph_folder, f))])

        #x_percent, avg_fees, std_fees = [], [], []  #  added std list
        x_percent, avg_fees, lower_errors, upper_errors = [], [], [], []

        for pct in pct_folders:
            stats_file = os.path.join(graph_folder, pct, 'stats.txt')
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    next(f)
                    fees_values = []
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) < 3:
                            continue
                        _, _, fee = parts
                        fees_values.append(float(fee))

                    if fees_values:
                        #avg_fees.append(np.mean(fees_values))
                        #std_fees.append(np.std(fees_values))  # std dev
                        mean_fee = np.mean(fees_values)
                        n = len(fees_values)
                        sample_std = np.std(fees_values, ddof=1)
                        standard_error = sample_std / np.sqrt(n)
                        margin = t.ppf(0.975, df=n - 1) * standard_error
                        ci_lower = max(0, mean_fee - margin)
                        ci_upper = mean_fee + margin
                        avg_fees.append(mean_fee)
                        lower_errors.append(mean_fee - ci_lower)
                        upper_errors.append(ci_upper - mean_fee)
                        x_percent.append(float(pct))
            except FileNotFoundError:
                print(f"File not found: {stats_file}")
                continue

        #  Error bars
        #leg = ax.errorbar(x_percent, avg_fees, yerr=std_fees, color=color[strategy], marker=MARKERS[strategy], label=DISPLAY_NAMES[strategy], capsize=3, elinewidth=1)
        yerr = np.array([lower_errors, upper_errors])
        leg = ax.errorbar(x_percent, avg_fees, yerr=yerr, color=color[strategy], marker=MARKERS[strategy], label=DISPLAY_NAMES[strategy], capsize=3, elinewidth=1)
        legend.append(leg)

    ymax = ax.get_ylim()[1]
    ax.set_ylim(bottom=-0.02 * ymax)
    ax.legend(loc='best')
    plt.grid(axis='y')

    ensure_figure_root()
    plt.savefig(FIGURE_ROOT / f'graph_{graph_id}_fees_node_removal.pdf', bbox_inches='tight')
    ##tikz_save(f'./figures/graph_{graph_id}_fees_node_removal.tex')
    plt.show()
    return legend

def plot_lcc_nodes_channels(feature, attack_type, strategies, graph_nodes, graph_id, result_version="7", ax=None):
    if feature not in ['nodes', 'channels']:
        print("Invalid feature! Choose 'nodes' or 'channels'.")
        return

    if ax is None:
        fig, ax = plt.subplots()

    ylabel = 'Number of Nodes in LCC' if feature == 'nodes' else 'Number of Channels in LCC'
    ax.set(xlabel='Ratio of Nodes Removed', ylabel=ylabel)
    legend = []

    col_index = 1 if feature == 'nodes' else 2

    for strategy in strategies:
        graph_folder = f'./results/robustness/{graph_nodes}_nodes_{result_version}/{attack_type}_removed/{strategy}/graph_{graph_id}'
        pct_folders = sorted([f for f in os.listdir(graph_folder) if os.path.isdir(os.path.join(graph_folder, f))])

        x_percent, avg_values, std_values = [], [], []  #  added std list

        for pct in pct_folders:
            lcc_file = os.path.join(graph_folder, pct, 'lcc.txt')
            try:
                with open(lcc_file, 'r', encoding='utf-8') as f:
                    next(f)
                    values = []
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) <= col_index:
                            continue
                        values.append(float(parts[col_index]))

                    if values:
                        avg_values.append(np.mean(values))
                        std_values.append(np.std(values))  #  std dev
                        x_percent.append(float(pct))
            except FileNotFoundError:
                print(f"File not found: {lcc_file}")
                continue

        #  Error bars
        leg = ax.errorbar(x_percent, avg_values, yerr=std_values, color=color[strategy], marker=MARKERS[strategy], label=DISPLAY_NAMES[strategy], capsize=3, elinewidth=1)
        legend.append(leg)

    ax.legend(loc='best')
    plt.grid(axis='y')

    ensure_figure_root()
    plt.savefig(FIGURE_ROOT / f'graph_{graph_id}_lcc_{feature}.pdf', bbox_inches='tight')
    ##tikz_save(f'./figures/graph_{graph_id}_lcc_{feature}.tex')
    plt.show()
    return legend

def plot_cg_node_removal( attack_type, strategies, graph_nodes, graph_id, result_version="7", ax=None):
    """
    Plot residual connectivity C(G) against the ratio of nodes removed.

    Reads:
      ./results/robustness/{graph_nodes}_nodes_{result_version}/
        {attack_type}_removed/{strategy}/graph_{graph_id}/{pct}/connectivity_cg.txt

    connectivity_cg.txt columns:
      Seed  Remaining_Nodes  Num_Components  Largest_Component_Nodes  C_G

    C(G) = sum_i p_i^2, where p_i is the proportion of remaining nodes
    in weakly connected component i.

    NOTE:
    These are C(G-S) values for the selected HD/HB attack sets.
    They are not the globally optimized R_k(G).
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.set( xlabel='Ratio of Nodes Removed', ylabel=r'Residual Connectivity $C(G)$' )

    legend = []

    for strategy in strategies:
        graph_folder = ( f'./results/robustness/' f'{graph_nodes}_nodes_{result_version}/' f'{attack_type}_removed/' f'{strategy}/graph_{graph_id}' )

        if not os.path.isdir(graph_folder):
            print(f"[WARN] Missing folder: {graph_folder}")
            continue

        pct_folders = sorted( [ f for f in os.listdir(graph_folder) if os.path.isdir(os.path.join(graph_folder, f)) ], key=lambda s: float(s) )

        x_percent = []
        avg_cg = []
        std_cg = []

        for pct in pct_folders:
            cg_file = os.path.join( graph_folder, pct, 'connectivity_cg.txt' )

            if not os.path.isfile(cg_file):
                print(f"File not found: {cg_file}")
                continue

            cg_values = []

            with open(cg_file, 'r', encoding='utf-8') as f:
                next(f, None)  # header

                for line in f:
                    parts = line.strip().split('\t')

                    if len(parts) < 5:
                        continue

                    try:
                        cg_values.append(float(parts[4]))
                    except ValueError:
                        continue

            if cg_values:
                x_percent.append(float(pct))
                avg_cg.append(float(np.mean(cg_values)))
                std_cg.append(float(np.std(cg_values)))

        if not avg_cg:
            continue

        # C(G) is deterministic for HD/HB in the current robustness code.
        # If the file contains only one value, the error bar is zero.
        leg = ax.errorbar( x_percent, avg_cg, yerr=std_cg, color=color[strategy], marker=MARKERS[strategy], label=DISPLAY_NAMES[strategy], capsize=3, elinewidth=1 )
        legend.append(leg)

    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='best')
    ax.grid(axis='y')

    ensure_figure_root()

    plt.savefig(FIGURE_ROOT / f'graph_{graph_id}_cg_node_{attack_type}_removal.pdf', bbox_inches='tight')
    plt.show()

    return legend

def plot_success_rate_vs_cg( attack_type, strategies, graph_nodes, graph_id, result_version="7", ax=None):
    """
    Plot unconditional transaction success rate against residual C(G).

    This directly supports the interpretation discussed for Reviewer 3:
    the x-axis represents the connectivity actually remaining after the
    attack instead of merely the fraction of nodes removed.

    Reads stats.txt and connectivity_cg.txt from the same attack-step folder.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.set( xlabel=r'Residual Connectivity $C(G)$', ylabel='Transaction Success Rate (%)' )

    legend = []

    for strategy in strategies:
        graph_folder = ( f'./results/robustness/' f'{graph_nodes}_nodes_{result_version}/' f'{attack_type}_removed/' f'{strategy}/graph_{graph_id}' )

        if not os.path.isdir(graph_folder):
            print(f"[WARN] Missing folder: {graph_folder}")
            continue

        pct_folders = sorted( [ f for f in os.listdir(graph_folder) if os.path.isdir(os.path.join(graph_folder, f)) ], key=lambda s: float(s) )

        cg_means = []
        success_means = []
        success_std = []

        for pct in pct_folders:
            pct_path = os.path.join(graph_folder, pct)
            cg_file = os.path.join(pct_path, 'connectivity_cg.txt')
            stats_file = os.path.join(pct_path, 'stats.txt')

            if not ( os.path.isfile(cg_file) and os.path.isfile(stats_file) ):
                continue

            cg_values = []
            with open(cg_file, 'r', encoding='utf-8') as f:
                next(f, None)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 5:
                        continue
                    try:
                        cg_values.append(float(parts[4]))
                    except ValueError:
                        continue

            success_values = []
            with open(stats_file, 'r', encoding='utf-8') as f:
                next(f, None)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3:
                        continue
                    try:
                        success_values.append(float(parts[1]) * 100.0)
                    except ValueError:
                        continue

            if cg_values and success_values:
                cg_means.append(float(np.mean(cg_values)))
                success_means.append(float(np.mean(success_values)))
                success_std.append(float(np.std(success_values)))

        if not cg_means:
            continue

        # Sort from more fragmented to more connected.
        order = np.argsort(cg_means)
        x = np.array(cg_means)[order]
        y = np.array(success_means)[order]
        yerr = np.array(success_std)[order]

        leg = ax.errorbar( x, y, yerr=yerr, color=color[strategy], marker=MARKERS[strategy], label=DISPLAY_NAMES[strategy], capsize=3, elinewidth=1 )
        legend.append(leg)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(-2, 102)
    ax.legend(loc='best')
    ax.grid(axis='y')

    ensure_figure_root()

    plt.savefig(FIGURE_ROOT / f'graph_{graph_id}_success_rate_vs_cg_{attack_type}.pdf', bbox_inches='tight')
    plt.show()

    return legend

def run_short_term(strategies, amount, graph_ids):
    """Plot the short-term results for the selected representative graphs."""
    for graph_id in graph_ids:
        plot_short_term(strategies, amount, None, "fees_routing", graph_id)
        plot_short_term(strategies, amount, None, "success_rate_bars", graph_id)


def run_long_term(strategies, graph_ids):
    """Plot the long-term results for the selected representative graphs."""
    for graph_id in graph_ids:
        save_long_term_figures(graph_id, strategies)


def run_robustness(strategies, attack_type, graph_nodes, graph_ids, robustness_plot="cg", result_version="1"):
    """Plot the selected robustness metric for the selected representative graphs."""
    for graph_id in graph_ids:
        if robustness_plot == "all":
            plot_success_rate_node_removal(attack_type, strategies, graph_nodes, graph_id, result_version)
            plot_fees_node_removal(attack_type, strategies, graph_nodes, graph_id, result_version)
            plot_lcc_nodes_channels("nodes", attack_type, strategies, graph_nodes, graph_id, result_version)
            plot_lcc_nodes_channels("channels", attack_type, strategies, graph_nodes, graph_id, result_version)
            plot_cg_node_removal(attack_type, strategies, graph_nodes, graph_id, result_version)
            plot_success_rate_vs_cg(attack_type, strategies, graph_nodes, graph_id, result_version)
        elif robustness_plot == "success":
            plot_success_rate_node_removal(attack_type, strategies, graph_nodes, graph_id, result_version)
        elif robustness_plot == "fees":
            plot_fees_node_removal(attack_type, strategies, graph_nodes, graph_id, result_version)
        elif robustness_plot == "lcc_nodes":
            plot_lcc_nodes_channels("nodes", attack_type, strategies, graph_nodes, graph_id, result_version)
        elif robustness_plot == "lcc_channels":
            plot_lcc_nodes_channels("channels", attack_type, strategies, graph_nodes, graph_id, result_version)
        elif robustness_plot == "cg":
            plot_cg_node_removal(attack_type, strategies, graph_nodes, graph_id, result_version)
        elif robustness_plot == "success_vs_cg":
            plot_success_rate_vs_cg(attack_type, strategies, graph_nodes, graph_id, result_version)
        else:
            raise ValueError(f"Unknown robustness plot: {robustness_plot}")


def parse_graph_ids(value):
    """Convert a comma-separated graph list such as 1,2,3 into integer graph IDs."""
    try:
        graph_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Graph IDs must be comma-separated integers, for example 1,2,3.") from exc
    if not graph_ids:
        raise argparse.ArgumentTypeError("At least one graph ID is required.")
    return graph_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ATLAS short-term, long-term, or robustness experiment results.")
    parser.add_argument("mode", choices=["short_term", "long_term", "robustness"], help="Type of results to plot.")
    parser.add_argument("--graphs", type=parse_graph_ids, default=[1, 2, 3], help="Comma-separated graph IDs. Default: 1,2,3")
    parser.add_argument("--amount", type=int, default=100, help="Transaction amount for short-term plots. Default: 100")
    parser.add_argument("--attack-type", choices=["highest_degree", "highest_betweenness"], default="highest_degree", help="Node-removal attack used for robustness plots.")
    parser.add_argument("--graph-nodes", default="10000", help="Robustness result graph-size folder prefix. Default: 10000")
    parser.add_argument("--robustness-plot", choices=["all", "success", "fees", "lcc_nodes", "lcc_channels", "cg", "success_vs_cg"], default="cg", help="Robustness figure to produce. Default: cg")
    parser.add_argument("--result-version", default="1", help="Robustness results version suffix. Default: 1")
    args = parser.parse_args()

    if args.mode == "short_term":
        run_short_term(ATTACHMENT_STRATEGIES, args.amount, args.graphs)
    elif args.mode == "long_term":
        run_long_term(ATTACHMENT_STRATEGIES, args.graphs)
    else:
        run_robustness(ATTACHMENT_STRATEGIES, args.attack_type, args.graph_nodes, args.graphs, args.robustness_plot, args.result_version)
