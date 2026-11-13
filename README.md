# AML Fraud Detection

Current work is focused on a verified, three-stage AML transaction-graph
preprocessing pipeline. Model training, backend, and frontend components are
not included yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pipeline uses the Kaggle IBM AML dataset and downloads
`HI-Small_Trans.csv` through KaggleHub when it is not already in `data/`.

## Run the pipeline

```bash
python -m ml.pipeline.01_ingest_clean
python -m ml.pipeline.02_build_graph
python -m ml.pipeline.03_to_pyg
```

Outputs are written to `data/parquet/`:

- `clean_tx.parquet` — cleaned transactions and engineered time features
- `nodes.parquet` — bank/account nodes with train-only behavioral features
- `edges.parquet` — chronological directed edges, labels, and split assignments
- `split_stats.json` — class-balance and split metrics
- `graph_data.pt` — PyTorch Geometric graph data

## Verify

```bash
python -m ml.verify_pipeline
```

The verifier checks cleaning rules, duplicate preservation, node and edge
integrity, leakage prevention, class imbalance, split statistics, and the PyG
tensor contract.
