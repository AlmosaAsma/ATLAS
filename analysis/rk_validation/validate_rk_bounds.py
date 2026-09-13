OUTPUT_DIR = Path("./rk_manuscript_bound_validation")#!/usr/bin/env python3
"""
validate_rk_manuscript_bounds.py

Small-graph validation of the SAME R_k bounds proposed for the ATLAS manuscript:

    C(G) = sum_i p_i^2
    R_k  = min_{S subseteq V, |S| <= k} C(G-S)

Manuscript bounds:

    1/n <= R_k <= min{ C(G-S_HD), C(G-S_HB) }

where:
- n is the number of nodes in the original graph,
- S_HD is the fixed highest-degree removal prefix,
- S_HB is the fixed highest-betweenness removal prefix,
- for each attack we evaluate all prefix sizes s = 0,...,k and use the
  minimum C(G-S), consistent with |S| <= k.

For each small graph, the script also computes the TRUE R_k by exhaustive
enumeration of every removal set with |S| <= k. This lets us verify directly:

    1/n <= exact R_k <= min(HD upper bound, HB upper bound)

No additional lower-bound method is used.

Outputs:
    ./rk_bound_validation/
        rk_bounds_summary.csv
        rk_bounds_all_sets.csv
        rk_bounds_report.txt

Requires:
    networkx
"""

import csv
import itertools
from pathlib import Path

import networkx as nx


OUTPUT_DIR = Path("analysis/rk_validation/results")
SUMMARY_CSV = OUTPUT_DIR / "rk_bounds_summary.csv"
ALL_SETS_CSV = OUTPUT_DIR / "rk_bounds_all_sets.csv"
REPORT_TXT = OUTPUT_DIR / "rk_bounds_report.txt"

SEED = 42
TOL = 1e-12


def connectivity_cg(G):
    """C(G) = sum_i p_i^2 over connected components."""
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    sizes = [len(c) for c in nx.connected_components(G)]
    return sum((size / n) ** 2 for size in sizes)


def exact_rk_exhaustive(G, k):
    """Enumerate all S with |S| <= k and return the exact R_k."""
    nodes = list(G.nodes())
    best = float("inf")
    best_sets = []
    rows = []

    for s in range(min(k, len(nodes)) + 1):
        for S in itertools.combinations(nodes, s):
            H = G.copy()
            H.remove_nodes_from(S)
            c = connectivity_cg(H)

            comps = (
                sorted((len(comp) for comp in nx.connected_components(H)), reverse=True)
                if H.number_of_nodes() else []
            )

            rows.append({
                "removed_count": s,
                "removed_nodes": " ".join(map(str, S)),
                "remaining_nodes": H.number_of_nodes(),
                "num_components": len(comps),
                "largest_component": comps[0] if comps else 0,
                "C_G": c,
            })

            if c < best - TOL:
                best = c
                best_sets = [tuple(S)]
            elif abs(c - best) <= TOL:
                best_sets.append(tuple(S))

    return best, best_sets, rows


def lower_bound_1_over_n(G):
    """
    Universal lower bound used in the manuscript for |S| <= k, assuming k < n:
        R_k >= 1/n
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return 1.0 / n


def fixed_degree_ranking(G):
    return [
        node for node, _ in sorted(
            G.degree(),
            key=lambda item: (-item[1], str(item[0]))
        )
    ]


def fixed_betweenness_ranking(G):
    bc = nx.betweenness_centrality(G)
    return [
        node for node, _ in sorted(
            bc.items(),
            key=lambda item: (-item[1], str(item[0]))
        )
    ]


def attack_upper_bound(G, ranking, k):
    """
    Evaluate every prefix size 0..k of a fixed ranking and return the minimum C.
    This is a feasible upper bound on R_k.
    """
    best = float("inf")
    best_s = None
    best_set = None

    for s in range(min(k, G.number_of_nodes()) + 1):
        S = tuple(ranking[:s])
        H = G.copy()
        H.remove_nodes_from(S)
        c = connectivity_cg(H)

        if c < best - TOL:
            best = c
            best_s = s
            best_set = S

    return best, best_s, best_set


def build_graphs():
    graphs = {
        "path_10": nx.path_graph(10),
        "cycle_10": nx.cycle_graph(10),
        "star_10": nx.star_graph(9),
        "complete_10": nx.complete_graph(10),
        "er_10_p035": nx.erdos_renyi_graph(10, 0.35, seed=SEED),
        "ba_10_m2": nx.barabasi_albert_graph(10, 2, seed=SEED),
        "ws_10_k4_p02": nx.watts_strogatz_graph(10, 4, 0.2, seed=SEED),
    }

    # Two dense clusters joined by a single bridge.
    G = nx.Graph()
    G.add_edges_from(nx.complete_graph(range(0, 5)).edges())
    G.add_edges_from(nx.complete_graph(range(5, 10)).edges())
    G.add_edge(4, 5)
    graphs["two_cliques_bridge_10"] = G

    return graphs


def fmt_sets(sets_, limit=4):
    out = []
    for S in sets_[:limit]:
        out.append("{" + ",".join(map(str, S)) + "}")
    if len(sets_) > limit:
        out.append(f"... +{len(sets_) - limit} more")
    return "; ".join(out)


def validate_graph(name, G, k):
    exact, optimal_sets, all_rows = exact_rk_exhaustive(G, k)

    lb = lower_bound_1_over_n(G)

    hd_rank = fixed_degree_ranking(G)
    hb_rank = fixed_betweenness_ranking(G)

    hd_ub, hd_s, hd_set = attack_upper_bound(G, hd_rank, k)
    hb_ub, hb_s, hb_set = attack_upper_bound(G, hb_rank, k)

    ub = min(hd_ub, hb_ub)
    ub_source = "HD" if hd_ub <= hb_ub else "HB"

    lb_valid = lb <= exact + TOL
    ub_valid = exact <= ub + TOL
    interval_valid = lb_valid and ub_valid

    absolute_interval_width = ub - lb
    relative_to_exact = absolute_interval_width / exact if exact > 0 else 0.0
    ub_optimality_gap = ub - exact
    ub_hits_exact = abs(ub - exact) <= TOL

    summary = {
        "graph": name,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "k": k,
        "enumerated_sets": len(all_rows),
        "exact_Rk": exact,
        "example_optimal_sets": fmt_sets(optimal_sets),
        "LB_1_over_n": lb,
        "HD_UB": hd_ub,
        "HD_best_s": hd_s,
        "HD_set": " ".join(map(str, hd_set or ())),
        "HB_UB": hb_ub,
        "HB_best_s": hb_s,
        "HB_set": " ".join(map(str, hb_set or ())),
        "best_UB": ub,
        "UB_source": ub_source,
        "certified_interval_width": absolute_interval_width,
        "interval_width_over_exact": relative_to_exact,
        "UB_minus_exact": ub_optimality_gap,
        "UB_equals_exact_Rk": ub_hits_exact,
        "LB_valid": lb_valid,
        "UB_valid": ub_valid,
        "BOUND_CHECK_PASS": interval_valid,
    }

    for row in all_rows:
        row["graph"] = name
        row["k_budget"] = k
        row["is_exact_optimum"] = abs(row["C_G"] - exact) <= TOL

    return summary, all_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    graphs = build_graphs()
    # k=2: for n=10, exhaustive validation checks 1 + 10 + 45 = 56 sets/graph.
    k = 2

    summaries = []
    all_rows = []

    for name, G in graphs.items():
        summary, rows = validate_graph(name, G, k)
        summaries.append(summary)
        all_rows.extend(rows)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    all_fields = [
        "graph", "k_budget", "removed_count", "removed_nodes",
        "remaining_nodes", "num_components", "largest_component",
        "C_G", "is_exact_optimum"
    ]
    with open(ALL_SETS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("SMALL-GRAPH VALIDATION OF THE ATLAS MANUSCRIPT R_k BOUNDS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Definition:\n")
        f.write("  R_k = min_{S subseteq V, |S| <= k} C(G-S)\n")
        f.write("  C(G) = sum_i p_i^2\n\n")
        f.write("Bounds being validated (exactly as proposed for the manuscript):\n")
        f.write("  1/n <= R_k <= min{C(G-S_HD), C(G-S_HB)}\n\n")
        f.write("The true R_k is computed by exhaustive enumeration of every S with |S| <= k.\n")
        f.write("No other lower-bound method is used.\n\n")

        for r in summaries:
            f.write("-" * 80 + "\n")
            f.write(f"Graph: {r['graph']}\n")
            f.write(f"n={r['n']}, m={r['m']}, k={r['k']}, enumerated={r['enumerated_sets']}\n")
            f.write(f"Exact R_k:                   {r['exact_Rk']:.12f}\n")
            f.write(f"1/n certified lower bound:  {r['LB_1_over_n']:.12f}\n")
            f.write(f"HD feasible upper bound:    {r['HD_UB']:.12f}\n")
            f.write(f"HB feasible upper bound:    {r['HB_UB']:.12f}\n")
            f.write(f"Best feasible upper bound:  {r['best_UB']:.12f} ({r['UB_source']})\n")
            f.write(
                f"Certified interval:          "
                f"[{r['LB_1_over_n']:.12f}, {r['best_UB']:.12f}]\n"
            )
            f.write(f"UB equals exact R_k:         {r['UB_equals_exact_Rk']}\n")
            f.write(f"LB <= exact R_k:             {r['LB_valid']}\n")
            f.write(f"exact R_k <= UB:             {r['UB_valid']}\n")
            f.write(f"BOUND CHECK:                 {'PASS' if r['BOUND_CHECK_PASS'] else 'FAIL'}\n\n")

        overall = all(r["BOUND_CHECK_PASS"] for r in summaries)
        f.write("=" * 80 + "\n")
        f.write(f"OVERALL RESULT: {'PASS' if overall else 'FAIL'}\n")

    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {ALL_SETS_CSV}")
    print(f"Wrote {REPORT_TXT}")
    print()
    print("Graph                     LB=1/n     exact R_k    best HD/HB UB   pass")
    print("-" * 78)
    for r in summaries:
        print(
            f"{r['graph']:<25} "
            f"{r['LB_1_over_n']:<10.6f} "
            f"{r['exact_Rk']:<12.6f} "
            f"{r['best_UB']:<15.6f} "
            f"{'PASS' if r['BOUND_CHECK_PASS'] else 'FAIL'}"
        )

    if not all(r["BOUND_CHECK_PASS"] for r in summaries):
        raise SystemExit("At least one bound validation failed.")


if __name__ == "__main__":
    main()
