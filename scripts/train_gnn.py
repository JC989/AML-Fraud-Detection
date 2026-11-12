"""Build and train a small transaction-level GNN without torch-geometric."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

try:  # Works both with ``python scripts/train_gnn.py`` and ``python -m``.
    from scripts.build_node_features import (
        add_account_ids,
        build_account_index,
        build_edge_index,
        build_edge_labels,
        build_node_features,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI form.
    from build_node_features import (
        add_account_ids,
        build_account_index,
        build_edge_index,
        build_edge_labels,
        build_node_features,
    )


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_labels: torch.Tensor
    account_to_index: dict[str, int]


def build_graph(transactions: pd.DataFrame) -> GraphData:
    transactions = add_account_ids(transactions)
    account_to_index = build_account_index(transactions)
    features = build_node_features(transactions, account_to_index)
    edges = build_edge_index(transactions, account_to_index)
    return GraphData(
        x=torch.tensor(features.to_numpy(dtype=np.float32)),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_labels=torch.tensor(build_edge_labels(transactions), dtype=torch.float32),
        account_to_index=account_to_index,
    )


class GCNTransactionClassifier(nn.Module):
    """Two-layer mean-aggregation GCN with an edge classifier."""

    def __init__(self, in_features: int, hidden_features: int = 32):
        super().__init__()
        self.node_layer_1 = nn.Linear(in_features, hidden_features)
        self.node_layer_2 = nn.Linear(hidden_features, hidden_features)
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_features * 2, hidden_features), nn.ReLU(), nn.Linear(hidden_features, 1)
        )

    @staticmethod
    def _aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        messages = torch.zeros_like(x)
        messages.index_add_(0, target, x[source])
        degree = torch.zeros(x.size(0), device=x.device)
        degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
        return messages / degree.clamp_min(1).unsqueeze(1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.node_layer_1(x) + self._aggregate(x, edge_index))
        h = torch.relu(self.node_layer_2(h) + self._aggregate(h, edge_index))
        source, target = edge_index
        return self.edge_classifier(torch.cat([h[source], h[target]], dim=1)).squeeze(1)


def train(graph: GraphData, epochs: int = 20, learning_rate: float = 1e-2, seed: int = 7) -> list[float]:
    torch.manual_seed(seed)
    model = GCNTransactionClassifier(graph.x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive = graph.edge_labels.sum().item()
    negative = len(graph.edge_labels) - positive
    weight = torch.tensor([negative / max(positive, 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(graph.x, graph.edge_index)
        loss = loss_fn(logits, graph.edge_labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "transactions",
        type=Path,
        nargs="?",
        help="Optional local transaction CSV; omitted loads it through KaggleHub.",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Load LI-Small_Trans.csv through KaggleHub instead of a local file.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.kaggle or args.transactions is None:
        try:
            from scripts.load_kaggle_dataset import load_file
        except ModuleNotFoundError:  # direct script execution
            from load_kaggle_dataset import load_file

        transactions = load_file("LI-Small_Trans.csv")
    else:
        transactions = pd.read_csv(args.transactions)
    graph = build_graph(transactions)
    losses = train(graph, epochs=args.epochs)
    print(f"nodes={graph.x.size(0):,} features={graph.x.size(1)} edges={graph.edge_index.size(1):,}")
    print(f"initial_loss={losses[0]:.4f} final_loss={losses[-1]:.4f}")


if __name__ == "__main__":
    main()
