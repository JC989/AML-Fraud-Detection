"""Tests for node features built by ml/pipeline/02_build_graph.py.

The fixture is a ten-transaction graph with distinct timestamps, so the
chronological split is fixed: rows 0-5 train, 6-7 val, 8-9 test. Expected
values are worked out by hand from the training rows only.
"""
from __future__ import annotations

import importlib
import math
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest

from ml.config import NODE_FEATURE_COLS

build_graph = importlib.import_module("ml.pipeline.02_build_graph")

# (from_bank, from_account, to_bank, to_account, timestamp, amount_paid, amount_received, payment_currency, receiving_currency)
TRANSACTIONS = [
    (1, "A", 1, "B", 0, 100.0, 100.0, "USD", "USD"),
    (1, "A", 1, "B", 3600, 200.0, 200.0, "USD", "USD"),
    (1, "A", 2, "C", 7200, 50.0, 45.0, "USD", "EUR"),      # cross-bank, cross-currency
    (2, "C", 1, "A", 10800, 300.0, 300.0, "USD", "USD"),
    (1, "B", 1, "B", 14400, 10.0, 10.0, "USD", "USD"),     # self-loop
    (1, "A", 1, "B", 18000, 400.0, 400.0, "USD", "USD"),
    (1, "A", 3, "D", 200000, 500.0, 500.0, "USD", "USD"),  # validation period
    (1, "B", 1, "A", 300000, 60.0, 60.0, "USD", "USD"),    # validation period
    (1, "A", 4, "E", 400000, 70.0, 70.0, "USD", "USD"),    # test period
    (2, "C", 4, "E", 500000, 80.0, 80.0, "USD", "USD"),    # test period
]


def _clean_frame() -> pd.DataFrame:
    """Build the columns that 01_ingest_clean.py writes and 02 reads."""
    rows = []
    for fb, fa, tb, ta, ts, paid, received, pay_cur, recv_cur in TRANSACTIONS:
        rows.append({
            "from_bank": fb, "from_account": fa, "to_bank": tb, "to_account": ta,
            "Timestamp": ts, "amount_paid": paid, "amount_received": received,
            "log_amount": math.log1p(paid), "log_amount_received": math.log1p(received),
            "hour_sin": 0.0, "hour_cos": 1.0, "dow_sin": 0.0, "dow_cos": 1.0,
            "is_cross_currency": int(pay_cur != recv_cur),
            "is_cross_bank": int(fb != tb),
            "is_self_loop": int(fb == tb and fa == ta),
            "payment_format": "ACH",
            "payment_currency": pay_cur, "receiving_currency": recv_cur,
            "is_laundering": 0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def graph_outputs(tmp_path, monkeypatch):
    clean_path = tmp_path / "clean_tx.parquet"
    con = duckdb.connect()
    con.register("clean_df", _clean_frame())
    con.execute(f"COPY clean_df TO '{clean_path}' (FORMAT PARQUET)")

    nodes_path = tmp_path / "nodes.parquet"
    edges_path = tmp_path / "edges.parquet"
    monkeypatch.setattr(build_graph, "PARQUET_DIR", tmp_path)
    monkeypatch.setattr(build_graph, "NODES_PATH", nodes_path)
    monkeypatch.setattr(build_graph, "EDGES_PATH", edges_path)
    monkeypatch.setattr(build_graph, "SPLIT_STATS_PATH", tmp_path / "split_stats.json")
    monkeypatch.setattr(sys, "argv", ["02_build_graph", "--clean", str(clean_path)])
    build_graph.main()

    nodes = con.execute("SELECT * FROM read_parquet(?)", [str(nodes_path)]).df()
    edges = con.execute("SELECT * FROM read_parquet(?)", [str(edges_path)]).df()
    return nodes.set_index(["Bank_ID", "Account_String"]), edges


def test_split_matches_fixture_assumption(graph_outputs):
    _, edges = graph_outputs
    counts = edges["split"].value_counts().to_dict()
    assert counts == {"train": 6, "val": 2, "test": 2}


def test_one_node_per_bank_account_pair(graph_outputs):
    nodes, _ = graph_outputs
    assert set(nodes.index) == {(1, "A"), (1, "B"), (2, "C"), (3, "D"), (4, "E")}


def test_all_configured_node_features_present_and_finite(graph_outputs):
    nodes, _ = graph_outputs
    missing = [col for col in NODE_FEATURE_COLS if col not in nodes.columns]
    assert missing == []
    assert np.isfinite(nodes[NODE_FEATURE_COLS].to_numpy(dtype=float)).all()


def test_degree_counts_use_training_edges_only(graph_outputs):
    nodes, _ = graph_outputs
    # Node A sends in rows 0, 1, 2, 5, 6, 8. Only 0, 1, 2, 5 are training rows.
    assert nodes.loc[(1, "A"), "out_degree"] == 4
    # Node A receives in rows 3 and 7. Only row 3 is a training row.
    assert nodes.loc[(1, "A"), "in_degree"] == 1


def test_unique_counterparties_use_training_edges_only(graph_outputs):
    nodes, _ = graph_outputs
    # Training destinations of A are B (rows 0, 1, 5) and C (row 2).
    # D (row 6) is validation and must not be counted.
    assert nodes.loc[(1, "A"), "unique_counterparties"] == 2


def test_amount_statistics_on_training_sends(graph_outputs):
    nodes, _ = graph_outputs
    train_sends = [100.0, 200.0, 50.0, 400.0]
    assert nodes.loc[(1, "A"), "max_log_sent"] == pytest.approx(math.log1p(400.0))
    assert nodes.loc[(1, "A"), "sum_log_sent"] == pytest.approx(sum(math.log1p(v) for v in train_sends))


def test_cross_bank_and_cross_currency_ratios(graph_outputs):
    nodes, _ = graph_outputs
    # One of A's four training sends (row 2) crosses banks and currencies.
    assert nodes.loc[(1, "A"), "cross_bank_out_ratio"] == pytest.approx(0.25)
    assert nodes.loc[(1, "A"), "cross_currency_out_ratio"] == pytest.approx(0.25)


def test_self_loop_ratio(graph_outputs):
    nodes, _ = graph_outputs
    # B's only training send is a self-loop (row 4).
    assert nodes.loc[(1, "B"), "self_loop_ratio"] == pytest.approx(1.0)


def test_daily_burst_features(graph_outputs):
    nodes, _ = graph_outputs
    # All of A's training sends fall on day 0.
    assert nodes.loc[(1, "A"), "active_days_out"] == 1
    assert nodes.loc[(1, "A"), "max_daily_out"] == 4
    assert nodes.loc[(1, "A"), "max_daily_fanout"] == 2  # B and C
    # On day 0, B receives from A (rows 0, 1, 5) and from itself (row 4).
    assert nodes.loc[(1, "B"), "max_daily_fanin"] == 2


def test_cold_start_node_has_zero_train_features(graph_outputs):
    nodes, _ = graph_outputs
    # E appears only in test rows (8 and 9), so no training behaviour exists.
    row = nodes.loc[(4, "E")]
    assert (row[NODE_FEATURE_COLS].to_numpy(dtype=float) == 0).all()
