"""Load IBM AML transaction data from Kaggle.

The dataset is large, so this script loads one file at a time. Use
``--file all`` only when you explicitly want every file loaded into memory.
"""

from __future__ import annotations

import argparse

import kagglehub
from kagglehub import KaggleDatasetAdapter


DATASET_HANDLE = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
DATASET_FILES = {
    "patterns": "LI-Small_Patterns.txt",
    "transactions": "LI-Small_Trans.csv",
    "accounts": "LI-Small_accounts.csv",
}


def load_file(file_path: str):
    """Load one dataset file as a pandas DataFrame through KaggleHub."""
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_HANDLE,
        file_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        choices=[*DATASET_FILES, "all"],
        default="transactions",
        help="Dataset file to load (default: transactions).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_files = DATASET_FILES.values() if args.file == "all" else [DATASET_FILES[args.file]]
    for file_path in selected_files:
        df = load_file(file_path)
        print(f"Loaded {file_path}: {len(df):,} records, {len(df.columns):,} columns")
        print("First 5 records:")
        print(df.head())
        print()


if __name__ == "__main__":
    main()
