from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import duckdb
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from ml.config import NODES_PATH, EDGES_PATH, PYG_DATA_PATH, NODE_FEATURE_COLS, NODE_LOG1P_COLS, EDGE_CONTINUOUS_COLS, EDGE_CATEGORICAL_COLS


def main() -> None:
    """Convert Parquet declarations into a PyTorch Geometric Data object."""
    p = argparse.ArgumentParser(); p.add_argument("--nodes", type=Path, default=NODES_PATH); p.add_argument("--edges", type=Path, default=EDGES_PATH); p.add_argument("--output", type=Path, default=PYG_DATA_PATH); args = p.parse_args()
    con = duckdb.connect()
    def table(path):
        q = con.execute("SELECT * FROM read_parquet(?)", [str(path)])
        return {name: values for name, values in q.fetchnumpy().items()}
    nodes, edges = table(args.nodes), table(args.edges)
    # DuckDB's NumPy fetch avoids loading the raw CSV or requiring pandas.
    x_raw = np.column_stack([nodes[col] for col in NODE_FEATURE_COLS]).astype(np.float32)
    node_train = (x_raw[:, 0] + x_raw[:, 1]) > 0
    # Compress heavy tails before scaling; log1p is monotonic, so ranking
    # information is unchanged and cold-start zeros stay zero.
    for i, col in enumerate(NODE_FEATURE_COLS):
        if col in NODE_LOG1P_COLS:
            x_raw[:, i] = np.log1p(x_raw[:, i])
    # Cold-start nodes have no train behavior; fit only on train-active nodes.
    node_scaler = StandardScaler().fit(x_raw[node_train] if node_train.any() else x_raw)
    x = torch.tensor(node_scaler.transform(x_raw), dtype=torch.float32)
    e_raw = np.column_stack([edges[col] for col in EDGE_CONTINUOUS_COLS]).astype(np.float32)
    train_mask = edges["split"] == "train"
    # Never fit continuous edge scaling on validation or test rows.
    edge_scaler = StandardScaler().fit(e_raw[train_mask] if train_mask.any() else e_raw)
    continuous = edge_scaler.transform(e_raw)
    # Encode categorical IDs as one-hot blocks, preserving their integer IDs in
    # the Parquet declaration while exposing dense model-ready attributes.
    categorical = []
    for col in EDGE_CATEGORICAL_COLS:
        ids = edges[col].astype(np.int64)
        one_hot = np.zeros((len(ids), int(ids.max()) + 1 if len(ids) else 0), dtype=np.float32)
        if len(ids): one_hot[np.arange(len(ids)), ids] = 1.0
        categorical.append(one_hot)
    edge_attr = torch.tensor(np.concatenate([continuous, *categorical], axis=1), dtype=torch.float32)
    edge_index = torch.tensor(np.vstack([edges["src_node_id"], edges["dst_node_id"]]), dtype=torch.long)
    y = torch.tensor(edges["y"], dtype=torch.float32)
    train = torch.tensor(train_mask, dtype=torch.bool); val = torch.tensor(edges["split"] == "val", dtype=torch.bool); test = torch.tensor(edges["split"] == "test", dtype=torch.bool)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train, val_mask=val, test_mask=test)
    args.output.parent.mkdir(parents=True, exist_ok=True); torch.save(data, args.output)
    orphan = bool(((edge_index < 0) | (edge_index >= x.shape[0])).any())
    print(f"x={tuple(x.shape)} {x.dtype}; edge_index={tuple(edge_index.shape)} {edge_index.dtype}; edge_attr={tuple(edge_attr.shape)} {edge_attr.dtype}; y={tuple(y.shape)} {y.dtype}")
    print(f"masks: train={train.sum().item():,}, val={val.sum().item():,}, test={test.sum().item():,}; nan_or_inf={not bool(torch.isfinite(x).all() and torch.isfinite(edge_attr).all())}; orphan_endpoints={orphan}")


if __name__ == "__main__": main()
