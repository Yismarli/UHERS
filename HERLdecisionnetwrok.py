import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import os


def build_decision_network(performance_metrics, node_count):
    """
    Build directed graph based on correlation (or R²) comparison.
    Edge from better to worse performer.
    """
    print("================== Decision Network Construction (Regression) ==================")

    adjacency = np.zeros((node_count, node_count))

    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                corr_i = performance_metrics.loc[
                    performance_metrics['Node'] == i, 'Correlation'
                ].values[0]
                corr_j = performance_metrics.loc[
                    performance_metrics['Node'] == j, 'Correlation'
                ].values[0]
                if corr_i > corr_j:
                    adjacency[i, j] = 1

    print("Adjacency matrix:\n", adjacency)

    # Convert to sparse tensor
    coo = coo_matrix(adjacency)
    rows, cols = coo.nonzero()
    edge_index_np = np.array([rows, cols])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    print("Edge index tensor shape:", edge_index.shape)

    print("================== Decision Network Construction Complete ==================")
    return adjacency, edge_index


def build_node_features(
    node_predictions,
    node_test_predictions,
    performance_metrics,
    adjacency,
    node_count,
    validation_data,
    test_data,
    feature_importance_dict,
    regressors,
    top_n=0  # regression uses no additional features
):
    """
    Construct node feature files for each node (training and testing).
    """
    print("================== Node Feature Construction (Regression) ==================")

    out_degrees = np.sum(adjacency == 1, axis=1)
    in_degrees = np.sum(adjacency == 1, axis=0)

    norm_out = out_degrees / np.linalg.norm(out_degrees) if np.linalg.norm(out_degrees) > 0 else out_degrees
    norm_in = in_degrees / np.linalg.norm(in_degrees) if np.linalg.norm(in_degrees) > 0 else in_degrees

    for node_idx in range(node_count):
        print(f"Processing node {node_idx}")

        model_name, _ = regressors[node_idx]
        features_info = feature_importance_dict.get(model_name, {'feature_names': []})
        selected_features = validation_data[features_info['feature_names'][:top_n]] if features_info['feature_names'] else pd.DataFrame()

        # Training features
        node_col = node_predictions.iloc[:, node_idx]
        metrics_row = performance_metrics.iloc[node_idx]

        train_df = pd.DataFrame({
            'NodePredictions': node_col,
            'Correlation': metrics_row['Correlation'],
            'ExplainedVariance': metrics_row['ExplainedVariance'],
            'R2': metrics_row['R2'],
            'OutDegree': norm_out[node_idx],
            'InDegree': norm_in[node_idx]
        })

        # Add selected features if any
        if not selected_features.empty:
            selected_reset = selected_features.reset_index(drop=True)
            selected_reset.columns = [f'Feat_{i}' for i in range(selected_reset.shape[1])]
            train_df = pd.concat([train_df, selected_reset], axis=1)

        train_filename = f"node_train_features_reg_{node_idx}.csv"
        train_df.to_csv(train_filename, index=False)

        # Testing features
        test_node_col = node_test_predictions.iloc[:, node_idx]
        test_selected = test_data[features_info['feature_names'][:top_n]] if features_info['feature_names'] else pd.DataFrame()

        test_df = pd.DataFrame({
            'NodePredictions': test_node_col,
            'Correlation': metrics_row['Correlation'],
            'ExplainedVariance': metrics_row['ExplainedVariance'],
            'R2': metrics_row['R2'],
            'OutDegree': norm_out[node_idx],
            'InDegree': norm_in[node_idx]
        })

        if not test_selected.empty:
            test_selected_reset = test_selected.reset_index(drop=True)
            test_selected_reset.columns = [f'Feat_{i}' for i in range(test_selected_reset.shape[1])]
            test_df = pd.concat([test_df, test_selected_reset], axis=1)

        test_filename = f"node_test_features_reg_{node_idx}.csv"
        test_df.to_csv(test_filename, index=False)

    print("================== Node Feature Construction Complete ==================")


def load_graph_data(node_count):
    """
    Load node feature CSV files and combine into graph tensors.
    """
    print("================== Loading Graph Data (Regression) ==================")

    train_files = [f"node_train_features_reg_{i}.csv" for i in range(node_count)]
    test_files = [f"node_test_features_reg_{i}.csv" for i in range(node_count)]

    train_tensors = load_and_process_files(train_files, node_count)
    test_tensors = load_and_process_files(test_files, node_count)

    # Cleanup temporary files
    for f in train_files + test_files:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    print("================== Graph Data Loading Complete ==================")
    return train_tensors, test_tensors


def load_and_process_files(file_list, node_count):
    """
    Read CSV files, combine rows across nodes into sample-wise tensors.
    """
    dataframes = []
    for fname in file_list:
        if os.path.exists(fname):
            dataframes.append(pd.read_csv(fname))
        else:
            raise FileNotFoundError(f"File {fname} not found")

    num_samples = len(dataframes[0])
    tensors = {}

    for sample_idx in range(num_samples):
        sample_data = []
        for df in dataframes:
            sample_data.append(df.iloc[sample_idx])
        merged = pd.concat(sample_data, axis=1, ignore_index=True)
        tensors[f"Sample_{sample_idx+1}"] = torch.tensor(
            merged.transpose().values, dtype=torch.float32
        )

    return tensors