import sys
import unittest
from unittest.mock import patch

from scripts import load_kaggle_dataset


class ParseArgsTests(unittest.TestCase):
    def test_defaults_to_transactions(self):
        with patch.object(sys, "argv", ["load_kaggle_dataset.py"]):
            args = load_kaggle_dataset.parse_args()

        self.assertEqual(args.file, "transactions")

    def test_accepts_each_supported_file(self):
        for requested_file in ("patterns", "transactions", "accounts", "all"):
            with self.subTest(requested_file=requested_file), patch.object(
                sys, "argv", ["load_kaggle_dataset.py", "--file", requested_file]
            ):
                args = load_kaggle_dataset.parse_args()

            self.assertEqual(args.file, requested_file)

    def test_rejects_unknown_file(self):
        with patch.object(sys, "argv", ["load_kaggle_dataset.py", "--file", "unknown"]):
            with self.assertRaises(SystemExit) as error:
                load_kaggle_dataset.parse_args()

        self.assertEqual(error.exception.code, 2)


class MainTests(unittest.TestCase):
    @patch("scripts.load_kaggle_dataset.load_file")
    def test_all_loads_each_dataset_file(self, load_file):
        class FakeDataFrame:
            def __init__(self, row):
                self._row = row
                self.columns = ["row"]

            def __len__(self):
                return 1

            def head(self):
                return self._row

        load_file.side_effect = [
            FakeDataFrame({"row": 1}),
            FakeDataFrame({"row": 2}),
            FakeDataFrame({"row": 3}),
        ]
        with patch.object(sys, "argv", ["load_kaggle_dataset.py", "--file", "all"]):
            load_kaggle_dataset.main()

        self.assertEqual(
            load_file.call_args_list,
            [
                unittest.mock.call("LI-Small_Patterns.txt"),
                unittest.mock.call("LI-Small_Trans.csv"),
                unittest.mock.call("LI-Small_accounts.csv"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
