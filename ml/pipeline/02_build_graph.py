from __future__ import annotations
import argparse, json
from pathlib import Path
import duckdb
from ml.config import CLEAN_TX_PATH, PARQUET_DIR, NODES_PATH, EDGES_PATH, SPLIT_STATS_PATH, TRAIN_RATIO, VAL_RATIO

DAY = 86400


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
    c.execute("""CREATE OR REPLACE TEMP TABLE base AS SELECT o.edge_id, s.node_id src_node_id, d.node_id dst_node_id, o.Timestamp, o.amount_paid, o.amount_received,
      o.log_amount, o.log_amount_received, o.hour_sin, o.hour_cos, o.dow_sin, o.dow_cos, o.is_cross_currency, o.is_cross_bank, o.is_self_loop,
      pf.payment_format_id, cu.currency_id, rcu.currency_id receiving_currency_id, o.payment_format, o.is_laundering y, o.split
      FROM ordered o JOIN nodes s ON s.Bank_ID=o.from_bank AND s.Account_String=o.from_account
      JOIN nodes d ON d.Bank_ID=o.to_bank AND d.Account_String=o.to_account JOIN pf USING(payment_format)
      JOIN cu ON cu.currency=o.payment_currency JOIN cu rcu ON rcu.currency=o.receiving_currency""")

    # Relative-timing edge features. Each looks only at the same or earlier
    # transactions and never at labels, so they are valid in every split and
    # stay current for val/test (unlike the train-only node aggregates below).
    # A missing "previous" event means "longer ago than the dataset spans".
    never = c.execute("SELECT ln(1.0 + max(Timestamp) + 1) FROM base").fetchone()[0]
    # Incoming money excludes self-loops so an edge never matches itself.
    c.execute("CREATE OR REPLACE TEMP TABLE incoming AS SELECT dst_node_id node_id, Timestamp in_ts FROM base WHERE src_node_id <> dst_node_id")
    c.execute(f"""CREATE OR REPLACE TEMP TABLE edges AS SELECT b.*,
      coalesce(ln(1.0 + b.Timestamp - lag(b.Timestamp) OVER (PARTITION BY b.src_node_id ORDER BY b.Timestamp, b.edge_id)), {never}) log_gap_src_prev_out,
      coalesce(ln(1.0 + b.Timestamp - i.in_ts), {never}) log_gap_src_prev_in,
      coalesce(ln(1.0 + b.Timestamp - lag(b.Timestamp) OVER (PARTITION BY b.dst_node_id ORDER BY b.Timestamp, b.edge_id)), {never}) log_gap_dst_prev_in,
      ln(count(*) OVER (PARTITION BY b.src_node_id ORDER BY b.Timestamp RANGE BETWEEN {DAY} PRECEDING AND CURRENT ROW)) log_src_out_24h,
      ln(count(*) OVER (PARTITION BY b.dst_node_id ORDER BY b.Timestamp RANGE BETWEEN {DAY} PRECEDING AND CURRENT ROW)) log_dst_in_24h
      FROM base b ASOF LEFT JOIN incoming i ON i.node_id = b.src_node_id AND b.Timestamp >= i.in_ts
      ORDER BY b.edge_id""")

    # Aggregate node behavior from training edges only to prevent temporal leakage.
    c.execute(f"""CREATE OR REPLACE TEMP TABLE nf AS WITH t AS (SELECT * FROM edges WHERE split='train'),
      s AS (SELECT src_node_id id, count(*) out_degree, count(DISTINCT dst_node_id) unique_counterparties, sum(log_amount) sum_log_sent, avg(log_amount) mean_log_sent,
              max(log_amount) max_log_sent, coalesce(stddev_samp(log_amount), 0) std_log_sent, count(DISTINCT currency_id) sc,
              avg(is_cross_bank) cross_bank_out_ratio, avg(is_cross_currency) cross_currency_out_ratio, avg(is_self_loop) self_loop_ratio,
              avg((log_gap_src_prev_in < ln(1.0 + {DAY}))::INTEGER) FILTER (WHERE is_self_loop = 0) pass_through_ratio,
              count(DISTINCT Timestamp // {DAY}) active_days_out FROM t GROUP BY 1),
      r AS (SELECT dst_node_id id, count(*) in_degree, sum(log_amount) sum_log_recv, avg(log_amount) mean_log_recv,
              max(log_amount) max_log_recv, coalesce(stddev_samp(log_amount), 0) std_log_recv, count(DISTINCT currency_id) rc FROM t GROUP BY 1),
      ds AS (SELECT id, max(n) max_daily_out, max(k) max_daily_fanout FROM
              (SELECT src_node_id id, count(*) n, count(DISTINCT dst_node_id) k FROM t GROUP BY src_node_id, Timestamp // {DAY}) GROUP BY 1),
      dr AS (SELECT id, max(k) max_daily_fanin FROM
              (SELECT dst_node_id id, count(DISTINCT src_node_id) k FROM t GROUP BY dst_node_id, Timestamp // {DAY}) GROUP BY 1)
      SELECT n.*, coalesce(s.out_degree,0)::BIGINT out_degree, coalesce(r.in_degree,0)::BIGINT in_degree, coalesce(s.unique_counterparties,0)::BIGINT unique_counterparties,
      coalesce(s.sum_log_sent,0) sum_log_sent, coalesce(r.sum_log_recv,0) sum_log_recv, coalesce(s.mean_log_sent,0) mean_log_sent, coalesce(r.mean_log_recv,0) mean_log_recv,
      (coalesce(s.sc,0)+coalesce(r.rc,0))::BIGINT currency_diversity,
      coalesce(s.max_log_sent,0) max_log_sent, coalesce(r.max_log_recv,0) max_log_recv, coalesce(s.std_log_sent,0) std_log_sent, coalesce(r.std_log_recv,0) std_log_recv,
      coalesce(s.cross_bank_out_ratio,0) cross_bank_out_ratio, coalesce(s.cross_currency_out_ratio,0) cross_currency_out_ratio,
      coalesce(s.self_loop_ratio,0) self_loop_ratio, coalesce(s.pass_through_ratio,0) pass_through_ratio,
      coalesce(s.active_days_out,0)::BIGINT active_days_out, coalesce(ds.max_daily_out,0)::BIGINT max_daily_out,
      coalesce(ds.max_daily_fanout,0)::BIGINT max_daily_fanout, coalesce(dr.max_daily_fanin,0)::BIGINT max_daily_fanin
      FROM nodes n LEFT JOIN s ON s.id=n.node_id LEFT JOIN r ON r.id=n.node_id LEFT JOIN ds ON ds.id=n.node_id LEFT JOIN dr ON dr.id=n.node_id
      ORDER BY n.node_id""")
    c.execute(f"COPY nf TO '{NODES_PATH.resolve()}' (FORMAT PARQUET, COMPRESSION ZSTD)"); c.execute(f"COPY edges TO '{EDGES_PATH.resolve()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    rows = c.execute("SELECT split, sum(y=1)::BIGINT positive, sum(y=0)::BIGINT negative, count(*) total FROM edges GROUP BY split ORDER BY split").fetchall()
    overall = c.execute("SELECT sum(y=1)::BIGINT positive, sum(y=0)::BIGINT negative, count(*) total FROM edges").fetchone()
    def metrics(pos, neg, total): return {"positive": pos, "negative": neg, "total": total, "positive_rate": pos/total if total else 0, "imbalance_ratio_neg_to_pos": neg/pos if pos else None, "pos_weight": neg/pos if pos else None}
    manifest = {"num_nodes": c.execute("SELECT count(*) FROM nodes").fetchone()[0], "num_edges": overall[2], "overall": metrics(*overall), "by_split": {s: metrics(pos, neg, total) for s,pos,neg,total in rows}}
    SPLIT_STATS_PATH.write_text(json.dumps(manifest, indent=2)); print(json.dumps(manifest, indent=2)); print(f"wrote {NODES_PATH}, {EDGES_PATH}, {SPLIT_STATS_PATH}")


if __name__ == "__main__": main()
