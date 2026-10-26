# AML Fraud Detection

## Load the Kaggle dataset

This project uses the [IBM Transactions for Anti-Money Laundering (AML)](
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
) dataset from Kaggle. The dataset is large and is intentionally not stored in
this repository. The loader downloads data through KaggleHub and loads one file
at a time.

### Create a virtual environment

Create and activate a local `.venv` environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The `.venv` directory is excluded from version control.

### Install dependencies manually

```bash
python -m pip install "kagglehub[pandas-datasets]"
```

Authenticate with Kaggle as described in the [KaggleHub documentation](https://github.com/Kaggle/kagglehub#authentication), if your environment is not already configured.

### Run the loader

From the repository root:

```bash
python scripts/load_kaggle_dataset.py
```

By default, this loads `LI-Small_Trans.csv`. Select another file with:

```bash
python scripts/load_kaggle_dataset.py --file patterns
python scripts/load_kaggle_dataset.py --file accounts
```

The available dataset files are:

- `LI-Small_Patterns.txt`
- `LI-Small_Trans.csv`
- `LI-Small_accounts.csv`

To load all three files sequentially, use `--file all`. Each file is held in
memory only while it is being processed:

```bash
python scripts/load_kaggle_dataset.py --file all
```

The core KaggleHub call is:

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
    "LI-Small_Trans.csv",
)

print("First 5 records:", df.head())
```
