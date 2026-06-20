import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, explained_variance_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor,
    RandomForestRegressor, HistGradientBoostingRegressor,
    ExtraTreesRegressor, BaggingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor


def train_base_layer(X_train, X_test, y_train, y_test):
    """
    Train base layer regressors using 5-fold cross-validation.
    Returns predictions, performance metrics, node count, and feature importances.
    """
    print("================== Base Layer Training (Regression) Start ==================")

    regressors = [
        ('XGB', XGBRegressor()),
        ('CatBoost', CatBoostRegressor(verbose=False)),
        ('LightGBM', LGBMRegressor()),
        ('AdaBoost', AdaBoostRegressor()),
        ('GradientBoost', GradientBoostingRegressor()),
        ('RandomForest', RandomForestRegressor()),
        ('ExtraTrees', ExtraTreesRegressor()),
        ('Linear', LinearRegression()),
        ('DecisionTree', DecisionTreeRegressor()),
        ('HistGradient', HistGradientBoostingRegressor()),
        ('Bagging', BaggingRegressor()),
        ('KNeighbors', KNeighborsRegressor()),
        ('SVR', svm.SVR()),
        ('MLP', MLPRegressor(max_iter=500)),
        ('Ridge', Ridge(alpha=1.0)),
        ('Lasso', Lasso(alpha=0.1)),
        ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5))
    ]

    node_predictions = pd.DataFrame()
    node_test_predictions = pd.DataFrame()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Metrics storage
    performance_metrics = pd.DataFrame(columns=['Node', 'Correlation', 'ExplainedVariance', 'R2'])
    predictions_dict = {name: [] for name, _ in regressors}
    val_X_list = []
    val_y_list = []

    corr_scores, evs_scores, r2_scores = [], [], []

    print("Starting cross-validation folds...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        for model_idx, (name, reg) in enumerate(regressors):
            reg.fit(X_fold_train, y_fold_train)
            y_val_pred = reg.predict(X_fold_val)

            corr, _ = pearsonr(y_fold_val, y_val_pred)
            evs = explained_variance_score(y_fold_val, y_val_pred)
            r2 = r2_score(y_fold_val, y_val_pred)

            # Clip negative values to zero (for correlation if negative, treat as zero)
            corr = max(0, corr)
            evs = max(0, evs)
            r2 = max(0, r2)

            corr_scores.append((model_idx, corr))
            evs_scores.append((model_idx, evs))
            r2_scores.append((model_idx, r2))

            predictions_dict[name].append(y_val_pred)

        val_X_list.append(X_fold_val)
        val_y_list.append(y_fold_val)

    # Test set predictions
    for name, reg in regressors:
        test_pred = reg.predict(X_test)
        test_pred_df = pd.DataFrame(test_pred, columns=[name])
        node_test_predictions = pd.concat([node_test_predictions, test_pred_df], axis=1)

    # Average metrics over folds
    avg_corr = defaultdict(list)
    avg_evs = defaultdict(list)
    avg_r2 = defaultdict(list)

    for idx, corr in corr_scores:
        avg_corr[idx].append(corr)
    for idx, evs in evs_scores:
        avg_evs[idx].append(evs)
    for idx, r2 in r2_scores:
        avg_r2[idx].append(r2)

    for idx in range(len(regressors)):
        mean_corr = np.mean(avg_corr[idx]) if idx in avg_corr else np.nan
        mean_evs = np.mean(avg_evs[idx]) if idx in avg_evs else np.nan
        mean_r2 = np.mean(avg_r2[idx]) if idx in avg_r2 else np.nan
        performance_metrics.loc[idx] = {
            'Node': idx,
            'Correlation': mean_corr,
            'ExplainedVariance': mean_evs,
            'R2': mean_r2
        }

    print("Performance metrics per node:")
    print(performance_metrics)

    # Combine validation predictions
    for name, pred_list in predictions_dict.items():
        combined_preds = np.concatenate(pred_list)
        node_predictions[name] = combined_preds

    combined_val_X = pd.concat(val_X_list, ignore_index=True)
    combined_val_y = pd.concat(val_y_list, ignore_index=True)

    # Extract feature importances (simplified: use coefficients or importances)
    feature_importance_dict = extract_feature_importances(regressors, X_train)

    # Important features extraction (if needed, but we set top_n=0 for regression as per original)
    # In original code, importances_features_num was set to 0, so we skip feature extraction.
    # But we still return empty dataframes to match expected outputs.
    train_important_features = pd.DataFrame()
    test_important_features = pd.DataFrame()

    node_count = len(regressors)

    print("================== Base Layer Training Complete ==================")

    return (
        node_predictions,
        node_test_predictions,
        performance_metrics,
        node_count,
        combined_val_y,
        combined_val_X,
        feature_importance_dict,
        regressors,
        train_important_features,
        test_important_features
    )


def extract_feature_importances(regressors, X_data):
    """
    Extract feature importances or coefficients from regressors.
    """
    feature_importance_dict = {}
    feature_counts = defaultdict(int)

    for name, reg in regressors:
        try:
            if hasattr(reg, 'feature_importances_'):
                importances = reg.feature_importances_
            elif hasattr(reg, 'coef_'):
                importances = np.abs(reg.coef_)
            else:
                importances = None
        except AttributeError:
            importances = None

        if importances is not None:
            # Get top 10 features (though in regression we set top_n=0, so no features used)
            top_indices = np.argsort(importances)[::-1][:0]  # zero features
            top_features = X_data.columns[top_indices].tolist()
            for feat in top_features:
                feature_counts[feat] += 1
            feature_importance_dict[name] = {
                'indices': top_indices,
                'feature_names': top_features
            }
        else:
            # Use common features from other models
            common = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:0]
            common_names = [feat for feat, _ in common]
            common_indices = [list(X_data.columns).index(feat) for feat in common_names]
            feature_importance_dict[name] = {
                'indices': common_indices,
                'feature_names': common_names
            }

    return feature_importance_dict