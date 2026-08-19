"""
Main entry for Hierarchical Ensemble Regression (HERL).
"""
import pandas as pd
import numpy as np
import torch
import collections
from sklearn.model_selection import train_test_split

from base_layer_regression import train_base_regressors
from decision_network_regression import build_decision_network, construct_node_features, build_graph_data
from gnn_regression import run_regression_gnn
from tfpandsingle import tfp_averagingandsingle, tfp_blending, tfp_stacking


def main():
    print("============ Hierarchical Ensemble Regression (HERL) Start =============")

    # Load data (adjust path accordingly)
    DATA_PATH = "D:\\datasets\\regression_data.csv"
    data = pd.read_csv(DATA_PATH)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Step 1: Base layer training
    (node_train_preds, node_test_preds, perf_df, node_count,
     val_y, val_X, regressors) = train_base_regressors(
         X_train, X_test, y_train, y_test
     )

    # Convert to tensors
    y_val_tensor = torch.tensor(val_y.values)
    y_test_tensor = torch.tensor(y_test.values)

    # Step 2: Build decision network
    adjacency, edge_index = build_decision_network(perf_df, node_count)

    # Step 3: Construct node features (creates CSV files)
    construct_node_features(
        node_train_preds, node_test_preds, perf_df, adjacency,
        node_count, val_X, X_test
    )

    # Step 4: Build graph tensors from CSV files
    train_graphs, test_graphs = build_graph_data(node_count)

    # Step 5: Run GNN and iteratively remove nodes
    trained_model, removed_nodes = run_regression_gnn(
        train_graphs, test_graphs,
        y_val_tensor, y_test_tensor,
        edge_index, node_count, regressors
    )

    print(f"Final removed nodes: {removed_nodes}")

    # Step 6: Compare with traditional ensemble methods
    print("\n--- Comparison with Traditional Ensembles ---")
    tfp_averagingandsingle(X_train, X_test, y_train, y_test)
    tfp_blending(X_train, X_test, y_train, y_test)
    tfp_stacking(X_train, X_test, y_train, y_test)

    print("============ HERL Complete =============")


if __name__ == "__main__":
    main()