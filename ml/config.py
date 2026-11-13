"""Shared configuration for every stage of the AML graph pipeline."""

from pathlib import Path

# Kaggle source and local artifact paths.
DATASET_ID = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
RAW_FILENAME = "HI-Small_Trans.csv"
RAW_CSV_PATH = Path("data") / RAW_FILENAME
PARQUET_DIR = Path("data/parquet")
CLEAN_TX_PATH = PARQUET_DIR / "clean_tx.parquet"
NODES_PATH = PARQUET_DIR / "nodes.parquet"
EDGES_PATH = PARQUET_DIR / "edges.parquet"
SPLIT_STATS_PATH = PARQUET_DIR / "split_stats.json"
PYG_DATA_PATH = PARQUET_DIR / "graph_data.pt"
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

NODE_FEATURE_COLS = [
    "out_degree", "in_degree", "unique_counterparties", "sum_log_sent",
    "sum_log_recv", "mean_log_sent", "mean_log_recv", "currency_diversity",
]
# Raw Timestamp is deliberately excluded: with a temporal split, test-period
# values fall outside the training range. It is still used for ordering/splits.
EDGE_CONTINUOUS_COLS = [
    "log_amount", "is_cross_currency", "is_cross_bank",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
EDGE_CATEGORICAL_COLS = ["payment_format_id", "currency_id"]
