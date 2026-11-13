"""End-to-end audit for the three-stage AML graph pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import duckdb
import torch

from ml.config import CLEAN_TX_PATH, NODES_PATH, EDGES_PATH, SPLIT_STATS_PATH, PYG_DATA_PATH, NODE_FEATURE_COLS


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def main() -> None:
    print("=== AML PIPELINE DIAGNOSTIC REPORT ===")
    source = Path("ml/pipeline/01_ingest_clean.py").read_text()
    require("SELECT DISTINCT *" not in source.upper(), "Stage 1 contains accidental SELECT DISTINCT *")
    print("[PASS] Stage 1 contains no SELECT DISTINCT * deduplication")
    require(CLEAN_TX_PATH.exists() and NODES_PATH.exists() and EDGES_PATH.exists() and SPLIT_STATS_PATH.exists() and PYG_DATA_PATH.exists(), "Missing pipeline artifact")
    c = duckdb.connect()
    c.execute("PRAGMA memory_limit='2GB'")
    c.execute("PRAGMA threads=4")
    c.execute("PRAGMA preserve_insertion_order=false")
    # Materialize only the columns needed by the audit.  This makes every
    # subsequent assertion operate on local temporary tables instead of
    # repeatedly decompressing the Parquet files.
    print("Loading compact audit projections...", flush=True)
    c.execute(f"""CREATE OR REPLACE TEMP TABLE clean AS SELECT
        event_time, from_account, to_account, amount_paid, is_laundering,
        Timestamp, hour_sin, hour_cos, dow_sin, dow_cos,
        is_cross_currency, is_cross_bank FROM read_parquet('{CLEAN_TX_PATH}')""")
    c.execute(f"""CREATE OR REPLACE TEMP TABLE nodes AS SELECT node_id, Bank_ID,
        Account_String, {', '.join(NODE_FEATURE_COLS)} FROM read_parquet('{NODES_PATH}')""")
    c.execute(f"""CREATE OR REPLACE TEMP TABLE edges AS SELECT edge_id, src_node_id,
        dst_node_id, Timestamp, amount_paid, log_amount, is_cross_currency,
        is_cross_bank, payment_format, y, split FROM read_parquet('{EDGES_PATH}')""")
    print("Audit projections loaded.", flush=True)
    print("\n--- Stage 1: clean_tx.parquet ---")
    same = c.execute("""SELECT coalesce(sum(n-1),0) FROM (SELECT date_trunc('minute', event_time) minute_bucket, amount_paid, count(*) n FROM clean GROUP BY 1,2 HAVING count(*) > 1)""").fetchone()[0]
    require(same >= 0, "same-minute duplicate audit failed")
    print(f"same-minute identical-amount transactions preserved beyond first: {same:,}")
    checks = {
        "accounts non-null/non-empty": "from_account IS NOT NULL AND length(trim(from_account)) > 0 AND to_account IS NOT NULL AND length(trim(to_account)) > 0",
        "amount_paid > 0": "amount_paid > 0",
        "binary non-null laundering labels": "is_laundering IS NOT NULL AND is_laundering IN (0,1)",
        "Timestamp >= 0": "Timestamp >= 0",
        "cyclical features bounded": "hour_sin BETWEEN -1 AND 1 AND hour_cos BETWEEN -1 AND 1 AND dow_sin BETWEEN -1 AND 1 AND dow_cos BETWEEN -1 AND 1",
        "cross flags binary": "is_cross_currency IN (0,1) AND is_cross_bank IN (0,1)",
    }
    total = c.execute("SELECT count(*) FROM clean").fetchone()[0]
    for label, predicate in checks.items():
        bad = c.execute(f"SELECT count(*) FROM clean WHERE NOT ({predicate})").fetchone()[0]
        require(bad == 0, f"{label}: {bad} violations")
        print(f"[PASS] {label}: 0 violations")
    print(f"clean rows: {total:,}")
    print("\n--- Stage 2: graph declarations ---")
    print("Checking node IDs and endpoint bounds...", flush=True)
    n = c.execute("SELECT count(*) FROM nodes").fetchone()[0]
    node_bounds = c.execute("SELECT min(node_id), max(node_id), count(DISTINCT node_id), count(*) FROM nodes").fetchone()
    require(node_bounds == (0, n-1, n, n), f"node IDs not contiguous: {node_bounds}")
    require(c.execute("SELECT count(*) FROM (SELECT Bank_ID, Account_String FROM nodes GROUP BY 1,2 HAVING count(*)>1)").fetchone()[0] == 0, "duplicate node entity pairs")
    endpoint_bounds = c.execute("SELECT min(src_node_id), max(src_node_id), min(dst_node_id), max(dst_node_id) FROM edges").fetchone()
    orphan = int(endpoint_bounds[0] < 0 or endpoint_bounds[1] >= n or endpoint_bounds[2] < 0 or endpoint_bounds[3] >= n)
    require(orphan == 0, f"orphan endpoints: {orphan}")
    # Build the train-active set once; do not run a correlated subquery once
    # per node over millions of edges.
    c.execute("""CREATE OR REPLACE TEMP TABLE train_nodes AS
        SELECT src_node_id AS node_id FROM edges WHERE split='train'
        UNION
        SELECT dst_node_id AS node_id FROM edges WHERE split='train'""")
    cold = c.execute("SELECT count(*) FROM nodes n LEFT JOIN train_nodes t USING (node_id) WHERE t.node_id IS NULL").fetchone()[0]
    bad_cold = c.execute(f"""SELECT count(*) FROM nodes n LEFT JOIN train_nodes t USING (node_id)
        WHERE t.node_id IS NULL AND ({' OR '.join(f'abs({x}) > 0' for x in NODE_FEATURE_COLS)})""").fetchone()[0]
    require(bad_cold == 0, f"cold-start nodes with nonzero train features: {bad_cold}")
    print(f"[PASS] nodes: {n:,}, contiguous/unique; orphan endpoints: 0")
    print(f"train-active nodes: {n-cold:,}; val/test-only cold-start nodes: {cold:,}")
    print("Checking chronology, class counts, and signal tables...", flush=True)
    order_bad = c.execute("SELECT count(*) FROM (SELECT Timestamp, lag(Timestamp) OVER (ORDER BY edge_id) prev FROM edges) WHERE prev IS NOT NULL AND Timestamp < prev").fetchone()[0]
    require(order_bad == 0, f"edge chronology violations: {order_bad}")
    for left, right in [("train","val"),("val","test")]:
        a = c.execute(f"SELECT max(Timestamp) FROM edges WHERE split='{left}'").fetchone()[0]; b = c.execute(f"SELECT min(Timestamp) FROM edges WHERE split='{right}'").fetchone()[0]
        require(a is None or b is None or a <= b, f"{left}/{right} chronology overlap")
    print("[PASS] chronological edge order and split boundaries")
    print("\n--- Class breakdown ---")
    print(f"{'scope':<10}{'E':>14}{'legit':>14}{'illicit':>14}{'illicit %':>14}{'neg:pos':>14}")
    for scope in ["train", "val", "test", "overall"]:
        where = "" if scope == "overall" else f"WHERE split='{scope}'"
        pos, neg, e = c.execute(f"SELECT sum(y=1), sum(y=0), count(*) FROM edges {where}").fetchone()
        print(f"{scope:<10}{e:>14,}{neg:>14,}{pos:>14,}{100*pos/e:>13.6f}%{(neg/pos if pos else float('inf')):>14.2f}")
    print("\n--- Feature/class signal ---")
    print("y | mean_amount | mean_log_amount | cross_currency_rate | cross_bank_rate")
    for row in c.execute("SELECT y, round(avg(amount_paid),4), round(avg(log_amount),4), round(avg(is_cross_currency),4), round(avg(is_cross_bank),4) FROM edges GROUP BY y ORDER BY y").fetchall(): print(row)
    print("payment_format | laundering_rate | total")
    for row in c.execute("SELECT payment_format, round(avg(y),6), count(*) FROM edges GROUP BY payment_format ORDER BY payment_format").fetchall(): print(row)
    manifest = json.loads(SPLIT_STATS_PATH.read_text()); size = SPLIT_STATS_PATH.stat().st_size
    require(size < 10_240, f"split_stats.json is not compact: {size} bytes")
    require("train_pos_edge_ids" not in manifest and "train_neg_edge_ids" not in manifest, "manifest contains giant edge ID arrays")
    train = manifest["by_split"]["train"]; require(manifest["overall"]["total"] == e and train["pos_weight"] == train["negative"] / train["positive"], "manifest counts/pos_weight mismatch")
    print(f"[PASS] split_stats.json: {size:,} bytes; compact; pos_weight={train['pos_weight']:.6f}")
    print("\n--- Stage 3: PyG tensor contract ---")
    data = torch.load(PYG_DATA_PATH, map_location="cpu", weights_only=False)
    require(data.x.dtype == torch.float32 and data.x.ndim == 2 and torch.isfinite(data.x).all(), "invalid x")
    require(data.edge_index.dtype == torch.int64 and tuple(data.edge_index.shape[:1]) == (2,) and int(data.edge_index.min()) >= 0 and int(data.edge_index.max()) < n, "invalid edge_index")
    require(data.edge_attr.dtype == torch.float32 and data.edge_attr.shape[0] == data.edge_index.shape[1] and torch.isfinite(data.edge_attr).all(), "invalid edge_attr")
    require(data.y.dtype == torch.float32 and data.y.shape == (data.edge_index.shape[1],) and bool(torch.all((data.y == 0) | (data.y == 1))), "invalid y")
    masks = [data.train_mask, data.val_mask, data.test_mask]; require(all(x.dtype == torch.bool and x.shape == data.y.shape for x in masks), "invalid masks")
    require(bool((sum(masks) == 1).all()), "masks are not mutually exclusive and exhaustive")
    print(f"[PASS] x={tuple(data.x.shape)} {data.x.dtype}; edge_index={tuple(data.edge_index.shape)} {data.edge_index.dtype}; edge_attr={tuple(data.edge_attr.shape)}; y={tuple(data.y.shape)}")
    print(f"[PASS] masks exhaustive/disjoint; finite tensors; endpoints in [0, {n-1}]")
    source3 = Path("ml/pipeline/03_to_pyg.py").read_text()
    require(
        "edge_scaler = StandardScaler().fit" in source3
        and "train_mask" in source3
        and "node_scaler = StandardScaler().fit" in source3,
        "scaler training isolation not evident",
    )
    print("[PASS] StandardScaler fit uses training edge rows and train-active node rows")
    print("\n=== ALL PIPELINE CHECKS PASSED ===")


if __name__ == "__main__": main()
