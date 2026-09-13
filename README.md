# ATLAS

Reproducibility repository for **ATLAS: A Longitudinal Evaluation Framework for Lightning Network Attachment Strategies**.

ATLAS evaluates Lightning Network (LN) attachment strategies across multiple empirical topology snapshots. The repository contains the simulator, snapshot-selection pipelines, short-term and long-term experiments, churn analysis, targeted-node-removal robustness evaluation, and plot_figures/analysis code used for the study.


## Repository structure

```text
ATLAS/
├── atlas/
│   ├── __init__.py
│   ├── attachment_strategies.py
│   ├── graph_io.py
│   ├── network_model.py
│   ├── simulator.py  				 # obtained from upstream simulator
│   └── utils.py      				 # obtained from upstream simulator
├── experiments/
│   ├── short_term.py
│   ├── long_term.py
│   └── robustness.py
├── snapshot_selection/
│   ├── netsimile_LN.py
│   └── sensitivity/
│       ├── netsimile.py
│       ├── graphsage.py
│       ├── graph2vec.py
│       └── node2vec.py
├── analysis/
│   ├── plot_figures.py
│   ├── churn_rates.py
│   ├── compare_sensitivity.py
│   └── rk_validation/
│       ├── validate_rk_bounds.py
│       └── results/
│           ├── rk_bounds_summary.csv
│           ├── rk_bounds_all_sets.csv
│           └── rk_bounds_report.txt
├── data/
│   ├── snapshots/
│   └── churn/
├── results/
├── configs/
├── tests/
├── requirements.txt
└── README.md
```

Run all commands below **from the repository root**.

## Installation

### 1. Clone the repository

```bash
git clone https://git.soton.ac.uk/asmaalmosa/ATLAS
cd ATLAS
```

### 2. Obtain the upstream simulator files

ATLAS builds on the Lightning Network simulator developed by Lange et al.
Two files from the upstream implementation are required:

- `simulator.py`
- `utils.py`

Obtain these two files from the upstream repository:

[pcn-attachment-data](https://git.tu-berlin.de/rohrer/pcn-attachment-data)


Copy the files into:

```text
ATLAS/atlas/simulator.py
ATLAS/atlas/utils.py
```
### 3. Create a Python environment

A virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

The repository uses packages including NetworkX, NumPy, SciPy, scikit-learn, tqdm, Matplotlib, PyTorch, PyTorch Geometric, Karate Club, and Node2Vec. Exact versions should be taken from `requirements.txt`.

## Data preparation

Place LN topology snapshots in:

```text
data/snapshots/
```

The three representative snapshots used by the ATLAS experiment runners are expected under these filenames:

| Graph ID | Snapshot |
|---|---|
| 1 | `ln_topology_2022_07_01.json` |
| 2 | `network_graph_2023_04_22.json` |
| 3 | `network_graph_2025_01_08.json` |

The snapshot-selection pipelines operate on the JSON snapshots stored in `data/snapshots/`.

We also provide `data/snapshot_manifest.csv` documenting the snapshots used for snapshot selection, including their names and dates.


## Attachment strategies

The final ATLAS evaluation uses five attachment strategies:

| Command-line identifier | Strategy |
|---|---|
| `random` | Random |
| `highest_degree` | Highest Degree |
| `bc_approx_10` | Betweenness |
| `k-center_gon_deg` | K-center |
| `closeness` | Closeness |

The identifiers above should be used when invoking the experiment scripts.

## 1. Snapshot selection

### Primary ATLAS representation: LN adapted NetSimile 

The primary snapshot-selection pipeline is:

```bash
python -m snapshot_selection.netsimile_LN
```

It reads:

```text
data/snapshots/*.json
```

and writes its outputs to:

```text
results/snapshot_selection/netsimile_LN/
```

The implementation constructs a 21-dimensional graph-level representation from summary statistics of degree, clustering coefficient, and approximate betweenness centrality. Candidate cluster counts are evaluated over `k = 2,...,10`, and representative snapshots are selected from the resulting clusters.

Expected output files include:

```text
netsimile_LN_embeddings.csv
netsimile_LN_k_metrics.csv
netsimile_LN_cluster_members.csv
netsimile_LN_representatives.csv
netsimile_LN_config.json
netsimile_LN_pipeline_log.txt
```

### Snapshot-selection sensitivity analysis

Alternative graph representations are provided under `snapshot_selection/sensitivity/`.

#### NetSimile

```bash
python -m snapshot_selection.sensitivity.netsimile
```

Outputs:

```text
results/snapshot_selection/sensitivity/netsimile/
```

This implementation uses the seven local/egonet features and five aggregators of NetSimile to construct a 35-dimensional signature and uses Canberra distance for graph comparison.

#### GraphSAGE

```bash
python -m snapshot_selection.sensitivity.graphsage
```

Outputs:

```text
results/snapshot_selection/sensitivity/graphsage/
```

The script records its training configuration and snapshot manifest as part of its reproducibility outputs.

> GraphSAGE may require a substantially different installation depending on CPU/CUDA and PyTorch/PyTorch-Geometric versions.

#### Graph2Vec

```bash
python -m snapshot_selection.sensitivity.graph2vec
```

Outputs:

```text
results/snapshot_selection/sensitivity/graph2vec/
```

#### Node2Vec

```bash
python -m snapshot_selection.sensitivity.node2vec
```

Outputs:

```text
results/snapshot_selection/sensitivity/node2vec/
```

### Compare sensitivity-analysis results

After running the primary and sensitivity pipelines:

```bash
python -m analysis.compare_sensitivity
```

This reads the snapshot-selection outputs and writes the cross-method comparison to:

```text
results/snapshot_selection/sensitivity_comparison/
```

including method summaries and clustering-partition agreement.

## 2. Short-term attachment experiment

The short-term experiment evaluates a new node establishing `k` channels according to a selected attachment strategy.

General syntax:

```bash
python -m experiments.short_term STRATEGY TX_AMOUNT K_START [CPUS] [SEED] --graphs GRAPH_IDS
```

Arguments:

- `STRATEGY`: one of the five strategy identifiers listed above.
- `TX_AMOUNT`: transaction amount in satoshis; supported values are `100`, `10000`, and `1000000`.
- `K_START`: first value of `k`, from 1 to 15. The script continues through `k=15`.
- `CPUS`: optional compatibility argument retained from the original simulator.
- `SEED`: random seed; default is 1.
- `--graphs`: comma-separated representative graph IDs; default is `1,2,3`.

Example:

```bash
python -m experiments.short_term random 100 1 1 1 --graphs 1,2,3
```

To run only Graph 1:

```bash
python -m experiments.short_term highest_degree 100 1 1 1 --graphs 1
```

Results are written under:

```text
results/short_term/<strategy>/graph_<id>/amt_<amount>/k_<k>/
```

The short-term experiment loads active LN channels and evaluates attachment on the largest strongly connected component of the selected starting snapshot.

## 3. Long-term growth experiment

The long-term experiment adds nodes to a representative LN topology, periodically evaluates the network, and applies the supplied period-specific churn factor.

General syntax:

```bash
python -m experiments.long_term STRATEGY GRAPH_ID CHURN_RATE [TX_AMOUNT] [EVAL_INTERVAL] [NODES_TO_ADD] [CPUS] [SEED]
```

Arguments:

- `STRATEGY`: attachment strategy.
- `GRAPH_ID`: representative topology (`1`, `2`, or `3`).
- `CHURN_RATE`: departure-to-arrival factor calibrated for the corresponding period.
- `TX_AMOUNT`: transaction amount in satoshis; default `100`.
- `EVAL_INTERVAL`: number of joining nodes between evaluation/churn checkpoints; default `1000`.
- `NODES_TO_ADD`: total joining nodes; default `10000`.
- `CPUS`: optional compatibility argument.
- `SEED`: random seed; default `1`.

Example:

```bash
python -m experiments.long_term random 1 0.956 100 1000 10000 1 1
```

This example runs Random on representative Graph 1 with a churn factor of `0.956`, adds 10,000 nodes, and evaluates the network every 1,000 additions.

Results are stored by strategy, representative graph, and seed:

```text
results/long_term/<strategy>/graph_<id>/seed_<seed>/
```

The long-term analysis records structural and payment-related metrics at evaluation checkpoints.  

> **Important:** Use the period-specific churn factors reported for the final manuscript experiments rather than treating the example value above as a universal setting.

## 4. Churn-rate calculation

Churn can be recalculated from dated LN snapshot sequences with:

```bash
python -m analysis.churn_rates SNAPSHOT_DIRECTORY --output-dir results/churn --label PERIOD
```

Example:

```bash
python -m analysis.churn_rates data/churn/2025 --output-dir results/churn --label 2025
```

For each period, the script writes:

```text
<period>_churn_intervals.csv
<period>_churn_summary.txt
```

The interval CSV records the quantities needed to inspect the churn calculation, while the summary reports the average arrival rate, average departure rate, and departure-to-arrival factor used by the long-term experiment.

## 5. Targeted-node-removal robustness experiment

The robustness experiment evaluates evolved long-term graphs under two targeted node-removal attacks:

- `highest_degree`
- `highest_betweenness` 


Available metric modes are:

| Mode | Output |
|---|---|
| `all` | success rate, fees, LCC, and \(C(G)\) |
| `lcc` | largest weakly connected component |
| `cg` | residual connectivity \(C(G)\) |

Run:

```bash
python -m experiments.robustness REMOVAL_STRATEGY METRIC [options]
```

Example:

```bash
python -m experiments.robustness highest_degree all --seed-start 1 --seed-end 10 --tx-amt 100 --tx-num 1000 --max-remove-pct 5
```

Useful options include:

```text
--graph-path        optional path to a single evolved long-term graph
--nodes-added       evolved-network checkpoint to evaluate (default: 10000)
--tx-amt            transaction amount (default: 100 sat)
--tx-num            number of transactions per attack step (default: 1000)
--seed-start        first seed
--seed-end          last seed
--result-version    output-version identifier
--max-remove-pct    maximum percentage of nodes removed (default: 5)
```

By default, the script evaluates the evolved long-term graphs explicitly listed in GRAPH_PATHS. These paths are grouped by representative graph and specify the exact evolved graphs used in the robustness experiments. Each listed evolved graph is kept fixed while the robustness evaluation is repeated for the seeds specified by --seed-start and --seed-end. To evaluate only a single evolved graph instead, provide its path using `--graph-path`.

Example:

```bash
python -m experiments.robustness highest_degree all --graph-path ./results/long_term/random/graph_3/seed_1/updated_graph3_random_1_10000.json --seed-start 1 --seed-end 10
```

The robustness implementation also stores the node-removal sets used for the targeted attacks.

## 6. plot_figures results

The plot_figures code reads the experiment outputs and generates manuscript-oriented figures under:

```text
results/figures/
```

The plot_figures entry point supports short-term, long-term, and robustness result modes. Run:

```bash
python -m analysis.plot_figures --help
```

to view the exact options available in the current release.

Typical usage is:

```bash
python -m analysis.plot_figures long_term --graphs 1,2,3
```

or for robustness figures:

```bash
python -m analysis.plot_figures robustness --graphs 1,2,3
```

### R_k bound validation

The `analysis/rk_validation/` directory contains the small-graph validation used to verify the bounds reported for the adversarial connectivity measure R_k. The script exhaustively enumerates all node removal sets with |S| <= k to obtain the exact R_k and verifies

1/n <= R_k <= min{C(G-S_HD), C(G-S_HB)}.

Run from the repository root with:

python analysis/rk_validation/validate_rk_bounds.py

## Reproducibility notes

### Random seeds

The final ATLAS experiments use fixed sets of independent random seeds to support reproducibility:

- **Short-term experiments:** seeds `1–10` (10 runs per representative snapshot and attachment strategy).
- **Long-term experiments:** seeds `1–5` (5 runs per representative snapshot and attachment strategy).
- **Robustness experiments:** seeds `1–10` (10 runs per configuration).

The experiment interfaces expose the seed value so that each individual run can be reproduced. Snapshot-selection methods define their reproducibility seeds directly in their scripts and configuration outputs.


### Graph representation

LN snapshots are parsed as directed multigraphs with two directed edges representing each bidirectional payment channel. Channel balances are initialized by dividing the advertised channel capacity equally between the two directions.

Different experimental stages may intentionally operate on different graph views. For example, the short-term and long-term attachment experiments begin from the largest strongly connected component, whereas the targeted-removal robustness experiment samples payment endpoints from the entire remaining attacked graph so that fragmentation affects unconditional payment success.

### Runtime

Some attachment strategies are computationally expensive on full LN topologies. Full long-term reproduction may therefore require substantial runtime. Reviewers or users wishing to validate the installation should first run a single representative graph, strategy, and seed before launching the complete experiment matrix.

## Expected workflow

A complete reproduction follows this order:

1. Install the required Python environment.
2. Place the LN snapshot dataset in `data/snapshots/`.
3. Run the NetSimile_LN snapshot-selection pipeline.
4. Optionally run GraphSAGE, Graph2Vec, Node2Vec, and original NetSimile for snapshot-selection sensitivity analysis.
5. Run `analysis.compare_sensitivity`.
6. Run the short-term experiments for the required strategies, snapshots, transaction amounts, and seeds.
7. Calculate/verify the period-specific churn factors.
8. Run the long-term experiments for each strategy, representative snapshot, and seed.
9. Run the targeted-node-removal robustness experiments on the evolved graphs.
10. Run the analysis/plot_figures code to aggregate repeated runs and generate figures.

## Citation

If you use ATLAS, please cite:

```bibtex
@article{atlas,
  title   = {ATLAS: A Longitudinal Evaluation Framework for Lightning Network Attachment Strategies},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

The citation entry will be updated after publication.

## License

This project is licensed under the Apache License, Version 2.0. See the LICENSE file for details.


## Contact

For questions about ATLAS or reproducing the experiments, please contact: **Asma Almosa**  at  `a.almosa@soton.ac.uk`
