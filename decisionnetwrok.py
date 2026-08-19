"""
Build decision network (adjacency matrix) based on correlation comparison,
and construct node feature files for graph data.
"""
import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import os


def build_decision_network(perf_df, node_count):
    """
    Build directed adjacency matrix: edge from higher correlation node to lower.
    """
    print("================== Building Decision Network (Regression) ==================")
    adj = np.zeros((node_count, node_count), dtype=int)

    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                corr_i = perf_df.loc[perf_df['Node'] == i, 'Correlation'].values[0]
                corr_j = perf_df.loc[perf_df['Node'] == j, 'Correlation'].values[0]
                if corr_i > corr_j:
                    adj[i, j] = 1

    print("Adjacency matrix:\n", adj)

    # Convert to sparse tensor
    coo = coo_matrix(adj)
    rows, cols = coo.nonzero()
    edge_index_np = np.array([rows, cols])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    print("Edge index:", edge_index)

    return adj, edge_index


def construct_node_features(node_train_preds, node_test_preds, perf_df, adjacency,
                            node_count, val_data, test_data):
    """
    Build node feature CSV files for training and testing.
    No feature importance/standardization needed (importances_features_num=0).
    """
    print("================== Constructing Node Features (Regression) ==================")

    # Compute normalized in/out degrees
    out_deg = np.sum(adjacency == 1, axis=1)
    in_deg = np.sum(adjacency == 1, axis=0)
    out_deg = out_deg / np.linalg.norm(out_deg) if np.linalg.norm(out_deg) > 0 else out_deg
    in_deg = in_deg / np.linalg.norm(in_deg) if np.linalg.norm(in_deg) > 0 else in_deg

    for i in range(node_count):
        print(f"Processing node {i}")

        # Training features
        col1 = node_train_preds.iloc[:, i]
        corr_val = perf_df.loc[perf_df['Node'] == i, 'Correlation'].values[0]
        evs_val = perf_df.loc[perf_df['Node'] == i, 'ExplainedVariance'].values[0]
        r2_val = perf_df.loc[perf_df['Node'] == i, 'R2'].values[0]

        train_df = pd.DataFrame({
            'NodePred': col1,
            'Correlation': corr_val,
            'ExplainedVariance': evs_val,
            'R2': r2_val,
            'OutDegree': out_deg[i],
            'InDegree': in_deg[i]
        })

        train_df.to_csv(f'node_train_features_reg_{i}.csv', index=False)

        # Testing features
        test_col = node_test_preds.iloc[:, i]

        test_df = pd.DataFrame({
            'NodePred': test_col,
            'Correlation': corr_val,
            'ExplainedVariance': evs_val,
            'R2': r2_val,
            'OutDegree': out_deg[i],
            'InDegree': in_deg[i]
        })

        test_df.to_csv(f'node_test_features_reg_{i}.csv', index=False)

    print("Node features saved.")


def load_and_process_files(file_names):
    """
    Load CSV files, combine rows across nodes into per-sample tensors.
    """
    dataframes = []
    for fname in file_names:
        df = pd.read_csv(fname + '.csv')
        dataframes.append(df)

    num_samples = len(dataframes[0])
    tensors = {}
    for sample_idx in range(num_samples):
        row_data = [df.iloc[sample_idx] for df in dataframes]
        merged = pd.concat(row_data, axis=1, ignore_index=True)
        tensors[f'G{sample_idx+1}'] = torch.tensor(merged.transpose().values, dtype=torch.float32)

    # Clean up CSV files
    for fname in file_names:
        try:
            os.remove(fname + '.csv')
        except FileNotFoundError:
            pass
    return tensors


def build_graph_data(node_count):
    """
    Build training and testing graph tensors from saved CSV files.
    """
    print("================== Building Graph Data (Regression) ==================")
    train_files = [f'node_train_features_reg_{i}' for i in range(node_count)]
    test_files = [f'node_test_features_reg_{i}' for i in range(node_count)]

    train_tensors = load_and_process_files(train_files)
    test_tensors = load_and_process_files(test_files)

    return train_tensors, test_tensors