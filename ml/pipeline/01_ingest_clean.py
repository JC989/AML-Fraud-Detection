from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import duckdb

from ml.config import DATASET_ID, RAW_CSV_PATH, RAW_FILENAME, CLEAN_TX_PATH, PARQUET_DIR


def get_raw(path: Path | None) -> Path:
    """Return a caller-provided CSV, or download the configured Kaggle file."""
    if path:
        return path
    if RAW_CSV_PATH.exists():
        return RAW_CSV_PATH
    import kagglehub
    downloaded = Path(kagglehub.dataset_download(DATASET_ID, path=RAW_FILENAME))
    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, RAW_CSV_PATH)
    return RAW_CSV_PATH


def main() -> None:
    """Clean the raw transaction CSV and write the canonical clean Parquet."""
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path)
    args = p.parse_args()
    raw = get_raw(args.raw)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    # Read every source field as text first; explicit casts below make parsing
    # failures visible and keep the cleaning rules deterministic.
    con.execute(
        "CREATE OR REPLACE TEMP TABLE raw AS "
        "SELECT * FROM read_csv_auto(?, header=true, all_varchar=true)",
        [str(raw)],
    )
    def count(label: str, relation: str = "raw") -> None:
        print(f"{label}: {con.execute(f'SELECT count(*) FROM {relation}').fetchone()[0]:,}")
    count("raw rows")
    # Normalize the duplicate Account headers into unambiguous endpoint names.
    con.execute("""CREATE OR REPLACE TEMP TABLE normalized AS SELECT
        trim("From Bank") from_bank, trim("To Bank") to_bank, trim("Account") from_account,
        trim("Account_1") to_account, try_strptime(trim("Timestamp"), '%Y/%m/%d %H:%M') event_time,
        try_cast("Amount Paid" AS DOUBLE) amount_paid, try_cast("Amount Received" AS DOUBLE) amount_received,
        trim("Payment Currency") payment_currency,
        trim("Receiving Currency") receiving_currency, trim("Payment Format") payment_format,
        try_cast("Is Laundering" AS INTEGER) is_laundering FROM raw""")
    count("after header normalization", "normalized")
    # Preserve row multiplicity: there is intentionally no DISTINCT here.
    con.execute("""CREATE OR REPLACE TEMP TABLE valid AS
        SELECT * FROM normalized
        WHERE from_account IS NOT NULL AND to_account IS NOT NULL
          AND event_time IS NOT NULL""")
    print(f"null accounts/timestamps removed: {con.execute('SELECT count(*) FROM normalized') .fetchone()[0] - con.execute('SELECT count(*) FROM valid').fetchone()[0]:,}")
    count("after null filtering", "valid")
    con.execute("""CREATE OR REPLACE TEMP TABLE amounts AS
        SELECT * FROM valid WHERE amount_paid > 0 AND amount_received > 0""")
    print(f"non-positive amounts removed: {con.execute('SELECT count(*) FROM valid').fetchone()[0] - con.execute('SELECT count(*) FROM amounts').fetchone()[0]:,}")
    count("after amount filtering", "amounts")
    con.execute("""CREATE OR REPLACE TEMP TABLE clean AS SELECT *,
      CAST(epoch(event_time) - min(epoch(event_time)) OVER () AS BIGINT) Timestamp,
      ln(1.0 + amount_paid) log_amount, ln(1.0 + amount_received) log_amount_received, sin(2*pi()*hour(event_time)/24.0) hour_sin,
      cos(2*pi()*hour(event_time)/24.0) hour_cos, sin(2*pi()*((dayofweek(event_time)+6)%7)/7.0) dow_sin,
      cos(2*pi()*((dayofweek(event_time)+6)%7)/7.0) dow_cos,
      (payment_currency <> receiving_currency)::INTEGER is_cross_currency,
      (from_bank <> to_bank)::INTEGER is_cross_bank,
      (from_bank = to_bank AND from_account = to_account)::INTEGER is_self_loop
      FROM amounts WHERE payment_currency IS NOT NULL AND receiving_currency IS NOT NULL AND payment_format IS NOT NULL""")
    print(f"null categorical strings removed: {con.execute('SELECT count(*) FROM amounts').fetchone()[0] - con.execute('SELECT count(*) FROM clean').fetchone()[0]:,}")
    count("final cleaned rows", "clean")
    # Zstandard keeps the large intermediate compact while remaining portable.
    con.execute(
        f"COPY clean TO '{CLEAN_TX_PATH.resolve()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    print(f"wrote {CLEAN_TX_PATH}")


if __name__ == "__main__":
    main()
