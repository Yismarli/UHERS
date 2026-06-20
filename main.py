import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from HERLbaselayer import train_base_layer
from HERLdecisionnetwrok import build_decision_network, build_node_features, load_graph_data
from DNP import train_regression_gnn
from tfpandsingle import (
    ensemble_averaging_regression,
    blending_regression,
    stacking_regression
)


def main():
    print("============ Hierarchical Ensemble Regression (HERL) Start =============")

    # Load data (example path)
    DATA_PATH = "D:\\datasets\\regression\\your_data.csv"
    data = pd.read_csv(DATA_PATH)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Step 1: Base layer training
    (node_predictions,
     node_test_predictions,
     performance_metrics,
     node_count,
     val_y,
     val_X,
     feature_importance_dict,
     regressors,
     important_train,
     important_test) = train_base_layer(X_train, X_test, y_train, y_test)

    # Convert labels to tensors
    y_val_tensor = torch.tensor(val_y.values)
    y_test_tensor = torch.tensor(y_test.values)

    # Step 2: Build decision network
    adjacency, edge_index = build_decision_network(performance_metrics, node_count)

    # Step 3: Build node features
    build_node_features(
        node_predictions,
        node_test_predictions,
        performance_metrics,
        adjacency,
        node_count,
        val_X,
        X_test,
        feature_importance_dict,
        regressors
    )

    # Step 4: Load graph data
    train_graphs, test_graphs = load_graph_data(node_count)

    # Step 5: Train GNN and remove nodes
    trained_model, removed_nodes = train_regression_gnn(
        train_graphs,
        test_graphs,
        y_val_tensor,
        y_test_tensor,
        edge_index,
        node_count,
        epochs=200
    )

    print(f"Final removed nodes: {removed_nodes}")

    # Step 6: Compare with traditional ensemble methods
    print("\n--- Comparison with Traditional Ensembles ---")
    ensemble_averaging_regression(X_train, X_test, y_train, y_test)
    blending_regression(X_train, X_test, y_train, y_test)
    stacking_regression(X_train, X_test, y_train, y_test)

    print("============ HERL Complete =============")


if __name__ == "__main__":
    main()
