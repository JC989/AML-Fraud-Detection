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
    # Amount spread and peaks.
    "max_log_sent", "max_log_recv", "std_log_sent", "std_log_recv",
    # Behaviour mix: share of outgoing edges that cross banks/currencies or are
    # self-transfers, and share forwarded within 24h of money arriving.
    "cross_bank_out_ratio", "cross_currency_out_ratio", "self_loop_ratio",
    "pass_through_ratio",
    # Bursts: busiest single day by sends, distinct recipients, distinct senders.
    "active_days_out", "max_daily_out", "max_daily_fanout", "max_daily_fanin",
]
# Heavy-tailed count/sum node features get log1p before standard scaling, so a
# few hub accounts do not compress every other account into a sliver of z-space.
# Ratios and already-log-scale amount stats are left as they are.
NODE_LOG1P_COLS = [
    "out_degree", "in_degree", "unique_counterparties", "sum_log_sent",
    "sum_log_recv", "currency_diversity", "active_days_out", "max_daily_out",
    "max_daily_fanout", "max_daily_fanin",
]
# Raw Timestamp is deliberately excluded: with a temporal split, test-period
# values fall outside the training range. It is still used for ordering/splits.
EDGE_CONTINUOUS_COLS = [
    "log_amount", "log_amount_received", "is_cross_currency", "is_cross_bank",
    "is_self_loop", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Causal relative timing (past/same-minute transactions only, label-free).
    "log_gap_src_prev_out", "log_gap_src_prev_in", "log_gap_dst_prev_in",
    "log_src_out_24h", "log_dst_in_24h",
]
EDGE_CATEGORICAL_COLS = ["payment_format_id", "currency_id", "receiving_currency_id"]
