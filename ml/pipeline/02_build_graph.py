from __future__ import annotations
import argparse, json
from pathlib import Path
import duckdb
from ml.config import CLEAN_TX_PATH, PARQUET_DIR, NODES_PATH, EDGES_PATH, SPLIT_STATS_PATH, TRAIN_RATIO, VAL_RATIO


def main() -> None:
    """Build deterministic node/edge declarations from clean transactions."""
    p = argparse.ArgumentParser(); p.add_argument("--clean", type=Path, default=CLEAN_TX_PATH); args = p.parse_args()
    PARQUET_DIR.mkdir(parents=True, exist_ok=True); c = duckdb.connect()
    c.execute("CREATE OR REPLACE TEMP TABLE tx AS SELECT * FROM read_parquet(?)", [str(args.clean)])
    # Stable tie-breakers make IDs reproducible when many events share a minute.
    c.execute(f"""CREATE OR REPLACE TEMP TABLE ordered AS SELECT *, row_number() OVER (ORDER BY Timestamp, from_bank, from_account, to_bank, to_account)-1 edge_id,
      CASE WHEN percent_rank() OVER (ORDER BY Timestamp, from_bank, from_account, to_bank, to_account) < {TRAIN_RATIO} THEN 'train'
           WHEN percent_rank() OVER (ORDER BY Timestamp, from_bank, from_account, to_bank, to_account) < {TRAIN_RATIO+VAL_RATIO} THEN 'val' ELSE 'test' END split FROM tx""")
    # Entity identity is the pair (bank, account), not the account string alone.
    c.execute("""CREATE OR REPLACE TEMP TABLE nodes AS SELECT row_number() OVER (ORDER BY Bank_ID, Account_String)-1 node_id, Bank_ID, Account_String FROM
      (SELECT DISTINCT from_bank Bank_ID, from_account Account_String FROM tx UNION SELECT DISTINCT to_bank, to_account FROM tx)""")
    c.execute("CREATE OR REPLACE TEMP TABLE pf AS SELECT payment_format, row_number() OVER (ORDER BY payment_format)-1 payment_format_id FROM (SELECT DISTINCT payment_format FROM tx)")
    c.execute("CREATE OR REPLACE TEMP TABLE cu AS SELECT currency, row_number() OVER (ORDER BY currency)-1 currency_id FROM (SELECT payment_currency currency FROM tx UNION SELECT receiving_currency FROM tx)")
    c.execute("""CREATE OR REPLACE TEMP TABLE edges AS SELECT o.edge_id, s.node_id src_node_id, d.node_id dst_node_id, o.Timestamp, o.amount_paid, o.log_amount,
      o.hour_sin, o.hour_cos, o.dow_sin, o.dow_cos, o.is_cross_currency, o.is_cross_bank, pf.payment_format_id,
      cu.currency_id, o.payment_format, o.is_laundering y, o.split FROM ordered o JOIN nodes s ON s.Bank_ID=o.from_bank AND s.Account_String=o.from_account
      JOIN nodes d ON d.Bank_ID=o.to_bank AND d.Account_String=o.to_account JOIN pf USING(payment_format) JOIN cu ON cu.currency=o.payment_currency ORDER BY o.edge_id""")
    # Aggregate node behavior from training edges only to prevent temporal leakage.
    c.execute("""CREATE OR REPLACE TEMP TABLE nf AS WITH s AS (SELECT src_node_id id, count(*) out_degree, count(DISTINCT dst_node_id) unique_counterparties, sum(log_amount) sum_log_sent, avg(log_amount) mean_log_sent, count(DISTINCT currency_id) sc FROM edges WHERE split='train' GROUP BY 1),
      r AS (SELECT dst_node_id id, count(*) in_degree, sum(log_amount) sum_log_recv, avg(log_amount) mean_log_recv, count(DISTINCT currency_id) rc FROM edges WHERE split='train' GROUP BY 1)
      SELECT n.*, coalesce(s.out_degree,0)::BIGINT out_degree, coalesce(r.in_degree,0)::BIGINT in_degree, coalesce(s.unique_counterparties,0)::BIGINT unique_counterparties,
      coalesce(s.sum_log_sent,0) sum_log_sent, coalesce(r.sum_log_recv,0) sum_log_recv, coalesce(s.mean_log_sent,0) mean_log_sent, coalesce(r.mean_log_recv,0) mean_log_recv,
      (coalesce(s.sc,0)+coalesce(r.rc,0))::BIGINT currency_diversity FROM nodes n LEFT JOIN s ON s.id=n.node_id LEFT JOIN r ON r.id=n.node_id""")
    c.execute(f"COPY nf TO '{NODES_PATH.resolve()}' (FORMAT PARQUET, COMPRESSION ZSTD)"); c.execute(f"COPY edges TO '{EDGES_PATH.resolve()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    rows = c.execute("SELECT split, sum(y=1)::BIGINT positive, sum(y=0)::BIGINT negative, count(*) total FROM edges GROUP BY split ORDER BY split").fetchall()
    overall = c.execute("SELECT sum(y=1)::BIGINT positive, sum(y=0)::BIGINT negative, count(*) total FROM edges").fetchone()
    def metrics(pos, neg, total): return {"positive": pos, "negative": neg, "total": total, "positive_rate": pos/total if total else 0, "imbalance_ratio_neg_to_pos": neg/pos if pos else None, "pos_weight": neg/pos if pos else None}
    manifest = {"num_nodes": c.execute("SELECT count(*) FROM nodes").fetchone()[0], "num_edges": overall[2], "overall": metrics(*overall), "by_split": {s: metrics(pos, neg, total) for s,pos,neg,total in rows}}
    SPLIT_STATS_PATH.write_text(json.dumps(manifest, indent=2)); print(json.dumps(manifest, indent=2)); print(f"wrote {NODES_PATH}, {EDGES_PATH}, {SPLIT_STATS_PATH}")


if __name__ == "__main__": main()
