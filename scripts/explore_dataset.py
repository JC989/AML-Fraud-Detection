"""Run a memory-conscious first-pass EDA on the IBM AML dataset.

The transaction file is streamed because the small/large variants contain
millions of rows. The patterns file is intentionally treated as structured
text; it is not assumed to be a rectangular CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


EXPECTED_TRANSACTION_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing AML data files")
    parser.add_argument(
        "--transactions",
        default="LI-Small_Trans.csv",
        help="Transaction CSV filename (default: LI-Small_Trans.csv)",
    )
    parser.add_argument(
        "--patterns",
        default="LI-Small_Patterns.txt",
        help="Patterns text filename (default: LI-Small_Patterns.txt)",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    values = sorted(values)
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {"min": values[0], "median": median, "max": values[-1]}


def analyze_transactions(path: Path, sample_size: int = 100_000) -> dict:
    result = {
        "path": str(path),
        "rows": 0,
        "columns": [],
        "duplicate_headers": {},
        "missing_values": Counter(),
        "target_counts": Counter(),
        "payment_formats": Counter(),
        "payment_currencies": Counter(),
        "receiving_currencies": Counter(),
        "unique_source_accounts": set(),
        "unique_destination_accounts": set(),
        "unique_source_banks": set(),
        "unique_destination_banks": set(),
        "amount_paid_sample": [],
        "amount_received_sample": [],
        "timestamps": [],
    }

    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        raw_header = next(reader)
        result["columns"] = raw_header
        positions: dict[str, list[int]] = {}
        for index, column in enumerate(raw_header):
            positions.setdefault(column, []).append(index)
        result["duplicate_headers"] = {
            name: indexes for name, indexes in positions.items() if len(indexes) > 1
        }
        if raw_header != EXPECTED_TRANSACTION_COLUMNS:
            result["schema_warning"] = "Header differs from the expected IBM AML transaction schema"

        for row in reader:
            result["rows"] += 1
            if len(row) != len(raw_header):
                result.setdefault("malformed_rows", 0)
                result["malformed_rows"] += 1
                continue
            for index, value in enumerate(row):
                if not value.strip():
                    result["missing_values"][raw_header[index]] += 1
            result["target_counts"][row[10]] += 1
            result["payment_formats"][row[9]] += 1
            result["payment_currencies"][row[8]] += 1
            result["receiving_currencies"][row[6]] += 1
            result["unique_source_accounts"].add((row[1], row[2]))
            result["unique_destination_accounts"].add((row[3], row[4]))
            result["unique_source_banks"].add(row[1])
            result["unique_destination_banks"].add(row[3])
            if len(result["amount_paid_sample"]) < sample_size:
                paid = _safe_float(row[7])
                received = _safe_float(row[5])
                if paid is not None:
                    result["amount_paid_sample"].append(paid)
                if received is not None:
                    result["amount_received_sample"].append(received)
            if len(result["timestamps"]) < sample_size:
                try:
                    result["timestamps"].append(datetime.strptime(row[0], "%Y/%m/%d %H:%M"))
                except ValueError:
                    pass

    rows = result["rows"]
    result["laundering_rate"] = (
        result["target_counts"].get("1", 0) / rows if rows else None
    )
    result["unique_source_accounts"] = len(result["unique_source_accounts"])
    result["unique_destination_accounts"] = len(result["unique_destination_accounts"])
    result["unique_source_banks"] = len(result["unique_source_banks"])
    result["unique_destination_banks"] = len(result["unique_destination_banks"])
    result["amount_paid_sample_stats"] = _quantiles(result.pop("amount_paid_sample"))
    result["amount_received_sample_stats"] = _quantiles(result.pop("amount_received_sample"))
    timestamps = result.pop("timestamps")
    result["sample_time_range"] = {
        "min": min(timestamps).isoformat() if timestamps else None,
        "max": max(timestamps).isoformat() if timestamps else None,
    }
    for key in ("missing_values", "target_counts", "payment_formats", "payment_currencies", "receiving_currencies"):
        result[key] = dict(result[key])
    return result


def analyze_patterns(path: Path) -> dict:
    line_count = 0
    nonempty_lines = 0
    first_lines: list[str] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            line_count += 1
            stripped = line.strip()
            if stripped:
                nonempty_lines += 1
                if len(first_lines) < 10:
                    first_lines.append(stripped)
    return {
        "path": str(path),
        "lines": line_count,
        "nonempty_lines": nonempty_lines,
        "first_nonempty_lines": first_lines,
        "note": "Patterns are structured text and require a dedicated parser before relational analysis.",
    }


def main() -> None:
    args = parse_args()
    report = {
        "transactions": analyze_transactions(args.data_dir / args.transactions),
        "patterns": analyze_patterns(args.data_dir / args.patterns),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote EDA report to {args.output}")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
