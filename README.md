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

## Project direction

The next phase is AML fraud prediction. Because the data contains accounts and
transactions, a graph neural network (GNN) may be applicable: accounts can be
nodes and transfers can be directed, timestamped edges. This must be confirmed
after inspecting the schemas and labels; a tabular baseline should be built
first so the graph model has a meaningful comparison.

## Graph ingestion and initial GNN

The graph pipeline treats each bank/account pair as one node and each
transaction as a directed edge.  The node feature matrix contains sent and
received counts and amounts, laundering-count, and total degree.  The edge
index has shape `[2, number_of_transactions]`, with row 0 containing source
node IDs and row 1 containing destination IDs; edge labels are the transaction
`Is Laundering` values.

Build and train the initial two-layer mean-aggregation GCN with:

```bash
python scripts/train_gnn.py /path/to/LI-Small_Trans.csv --epochs 20
```

The model predicts laundering at the transaction (edge) level.  It uses
PyTorch directly, so no torch-geometric installation is required.

Run the CLI tests with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Exploratory data analysis

After downloading the files into a local directory, run the streaming EDA
report. It does not load the full transaction file into memory:

```bash
python scripts/explore_dataset.py /path/to/aml-data --output eda.json
```

The report covers row counts, the duplicated `Account` header, malformed rows,
missing values, laundering prevalence, categorical distributions, sampled
amount statistics, timestamp range, and account/bank cardinalities. It also
records that `LI-Small_Patterns.txt` is structured text and should be parsed
separately before joining it to transactions.
