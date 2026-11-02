import pandas as pd



def add_account_ids(df):
    df = df.copy()
    df["sender"] = df["From Bank"].astype(str) + "_" + df["Account"].astype(str)
    df["receiver"] = df["To Bank"].astype(str) + "_" + df["Account.1"].astype(str)
    return df

def build_account_index(df):
    accounts = sorted(set(df["sender"]) | set(df["receiver"]))
    return {account: i for i, account in enumerate(accounts)}

def build_node_features(df, account_to_index, cutoff=None):
    accounts = list(account_to_index)
    features = {}

    features["sent_count"] = (
        df.groupby("sender").size().reindex(accounts, fill_value=0)
    )

    return pd.DataFrame(features, index=accounts)


    