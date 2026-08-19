"""
Graph Neural Network for regression with iterative node removal based on correlation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
)
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops


class RegressionGNN(nn.Module):
    """GCN-based regression model with residual connection."""
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(RegressionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim * 2)
        self.proj = nn.Linear(input_dim, hidden_dim * 2)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        conv_out = self.conv1(x, edge_index)
        projected = self.proj(x)
        combined = F.relu(conv_out + projected)
        output = self.linear(combined)
        return output.squeeze()


def filter_edges(edge_index, node_count, nodes_to_remove):
    """Remove edges incident to specified nodes."""
    if not nodes_to_remove:
        return edge_index
    mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
    for node in nodes_to_remove:
        connected = (edge_index[0] % node_count == node) | (edge_index[1] % node_count == node)
        mask &= ~connected
    return edge_index[:, mask]


def run_regression_gnn(train_graphs, test_graphs, y_train, y_test,
                       edge_index, node_count, classifiers):
    """
    Main function: train GNN, evaluate, and iteratively remove worst nodes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine input dimension
    sample_key = next(iter(train_graphs))
    input_dim = train_graphs[sample_key].shape[1]

    model = RegressionGNN(input_dim, 128, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Prepare datasets
    def create_dataset(graph_dict, labels):
        graph_list = []
        for idx, (key, value) in enumerate(graph_dict.items()):
            x = value
            y = torch.tensor([labels[idx]], dtype=torch.float64)
            graph_list.append(Data(x=x, edge_index=edge_index, y=y))
        return graph_list

    train_data = create_dataset(train_graphs, y_train)
    test_data = create_dataset(test_graphs, y_test)

    batch_size = 200
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    nodes_to_remove = []
    epochs = 200

    for removal_step in range(node_count - 1):
        print(f"Removal step {removal_step + 1}/{node_count - 1}")
        print(f"Removed nodes: {nodes_to_remove}")

        # Train model
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                x = batch.x.to(device)
                edge_idx = batch.edge_index.to(device)
                y = batch.y.to(device)

                if nodes_to_remove:
                    edge_idx = filter_edges(edge_idx, node_count, nodes_to_remove)

                output = model(x, edge_idx)

                # Pool predictions across nodes
                batch_size_curr = y.size(0)
                pooled = torch.zeros(batch_size_curr, 1, device=device)
                for i in range(batch_size_curr):
                    start = i * node_count
                    end = start + node_count
                    pooled[i] = output[start:end].mean(dim=0)

                pooled = pooled.squeeze()
                loss = criterion(pooled, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

        # Evaluate and select next node to remove
        next_node = evaluate_and_select_node(
            model, test_loader, node_count, nodes_to_remove
        )
        if next_node is not None:
            nodes_to_remove.append(next_node)
        else:
            break

        torch.cuda.empty_cache()

    print(f"Final removed nodes: {nodes_to_remove}")
    return model, nodes_to_remove


def evaluate_and_select_node(model, test_loader, node_count, nodes_to_remove):
    """
    Evaluate model on test set, compute correlation between each node's output
    and the pooled prediction, and return the node with lowest correlation
    (excluding already removed nodes).
    """
    device = next(model.parameters()).device
    model.eval()

    all_node_outputs = []
    all_pooled_preds = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch.x.to(device)
            edge_idx = batch.edge_index.to(device)
            y = batch.y.to(device)  # not directly used for correlation

            if nodes_to_remove:
                edge_idx = filter_edges(edge_idx, node_count, nodes_to_remove)

            output = model(x, edge_idx)  # shape: (batch_size * node_count,)
            batch_size_curr = y.size(0)
            node_outputs = output.view(batch_size_curr, node_count)  # [B, N]

            # Pooled predictions (mean over nodes)
            pooled = node_outputs.mean(dim=1)  # [B]

            all_node_outputs.append(node_outputs.cpu().numpy())
            all_pooled_preds.append(pooled.cpu().numpy())

    # Concatenate across batches
    node_outs = np.concatenate(all_node_outputs, axis=0)   # [total_samples, N]
    pooled_outs = np.concatenate(all_pooled_preds, axis=0) # [total_samples]

    # Compute correlation of each node's output with the pooled prediction
    corr_list = []
    for node_idx in range(node_count):
        node_vals = node_outs[:, node_idx]
        corr, _ = pearsonr(node_vals, pooled_outs)
        # Treat NaN or negative as 0 (original behaviour)
        if np.isnan(corr) or corr < 0:
            corr = 0.0
        corr_list.append((node_idx, corr))

    # Exclude already removed nodes
    removed_set = set(nodes_to_remove)
    filtered = [(idx, corr) for idx, corr in corr_list if idx not in removed_set]

    if not filtered:
        print("No nodes left to remove.")
        return None

    # Choose node with smallest correlation
    worst_node = min(filtered, key=lambda x: x[1])[0]
    worst_corr = min(filtered, key=lambda x: x[1])[1]
    print(f"Node with smallest correlation: {worst_node} (corr={worst_corr:.4f})")

    return worst_node