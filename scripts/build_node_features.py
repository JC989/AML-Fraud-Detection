"""Build the account graph used by the AML GNN.

The transaction CSV contains two columns named ``Account``.  Pandas renames
the second one to ``Account.1``; the helpers below deliberately use the
renamed form and keep the bank in the node key so account numbers are not
mistaken for globally unique identifiers.
"""

from __future__ import annotations

import pandas as pd



def add_account_ids(df):
    df = df.copy()
    df["sender"] = df["From Bank"].astype(str) + "_" + df["Account"].astype(str)
    df["receiver"] = df["To Bank"].astype(str) + "_" + df["Account.1"].astype(str)
    return df

def build_account_index(df):
    if "sender" not in df or "receiver" not in df:
        df = add_account_ids(df)
    accounts = sorted(set(df["sender"]) | set(df["receiver"]))
    return {account: i for i, account in enumerate(accounts)}

def build_node_features(df, account_to_index, cutoff=None):
    """Return a numeric feature matrix indexed in node-id order.

    ``cutoff`` is retained for backwards compatibility and, when supplied,
    limits transactions by row position before aggregating features.
    """
    work = add_account_ids(df)
    if cutoff is not None:
        work = work.iloc[:cutoff]
    accounts = [account for account, _ in sorted(account_to_index.items(), key=lambda item: item[1])]
    sent = work.groupby("sender", sort=False)
    received = work.groupby("receiver", sort=False)
    paid = pd.to_numeric(work["Amount Paid"], errors="coerce").fillna(0.0)
    received_amount = pd.to_numeric(work["Amount Received"], errors="coerce").fillna(0.0)
    work = work.assign(_paid=paid, _received=received_amount)
    features = pd.DataFrame(index=accounts)
    features["sent_count"] = sent.size().reindex(accounts, fill_value=0)
    features["received_count"] = received.size().reindex(accounts, fill_value=0)
    features["sent_amount"] = work.groupby("sender")[["_paid"]].sum() ["_paid"].reindex(accounts, fill_value=0.0)
    features["received_amount"] = work.groupby("receiver")[["_received"]].sum() ["_received"].reindex(accounts, fill_value=0.0)
    laundering = pd.to_numeric(
        work["Is Laundering"] if "Is Laundering" in work else pd.Series(0, index=work.index),
        errors="coerce",
    ).fillna(0)
    work = work.assign(_laundering=laundering)
    features["laundering_sent_count"] = work.groupby("sender") ["_laundering"].sum().reindex(accounts, fill_value=0)
    features["degree"] = features["sent_count"] + features["received_count"]
    return features.astype(float)


def build_edge_index(df, account_to_index):
    """Create a directed ``[2, number_of_transactions]`` edge-index matrix."""
    work = add_account_ids(df)
    missing = set(work["sender"]) | set(work["receiver"])
    missing -= set(account_to_index)
    if missing:
        raise ValueError("account_to_index is missing accounts present in transactions")
    return pd.DataFrame({
        "source": work["sender"].map(account_to_index).astype("int64"),
        "target": work["receiver"].map(account_to_index).astype("int64"),
    }).to_numpy().T


def build_edge_labels(df):
    """Return transaction laundering labels in the same order as edge_index."""
    if "Is Laundering" not in df:
        raise ValueError("transactions must contain an 'Is Laundering' column")
    return pd.to_numeric(df["Is Laundering"], errors="raise").astype("int64").to_numpy()


    
