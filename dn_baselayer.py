"""
Base layer training for regression: train multiple regressors with cross-validation,
collect out-of-fold predictions and performance metrics (correlation, EVS, R2).
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import math
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score, r2_score
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


def train_base_regressors(X_train, X_test, y_train, y_test):
    """
    Train base regressors with 5-fold CV. Returns predictions, performance metrics,
    and node count.
    """
    print("================== Base Layer Training (Regression) ==================")

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

    node_train_preds = pd.DataFrame()
    node_test_preds = pd.DataFrame()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Storage
    predictions = {name: [] for name, _ in regressors}
    val_X_list, val_y_list = [], []
    corr_scores, evs_scores, r2_scores = [], [], []

    print("Starting cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        for model_idx, (name, reg) in enumerate(regressors):
            reg.fit(X_fold_train, y_fold_train)
            y_val_pred = reg.predict(X_fold_val)

            corr, _ = pearsonr(y_fold_val, y_val_pred)
            evs = explained_variance_score(y_fold_val, y_val_pred)
            r2 = r2_score(y_fold_val, y_val_pred)

            # Clip negative values to zero (as in original)
            corr = max(0, corr if not math.isnan(corr) else 0)
            evs = max(0, evs)
            r2 = max(0, r2)

            corr_scores.append((model_idx, corr))
            evs_scores.append((model_idx, evs))
            r2_scores.append((model_idx, r2))

            predictions[name].append(y_val_pred)

        val_X_list.append(X_fold_val)
        val_y_list.append(y_fold_val)

    # Generate test predictions
    for name, reg in regressors:
        test_pred = reg.predict(X_test)
        test_pred_df = pd.DataFrame(test_pred, columns=[name])
        node_test_preds = pd.concat([node_test_preds, test_pred_df], axis=1)

    # Combine validation predictions
    for name, preds in predictions.items():
        combined = np.concatenate(preds)
        node_train_preds[name] = combined

    # Compute average metrics per model
    avg_corr = defaultdict(list)
    avg_evs = defaultdict(list)
    avg_r2 = defaultdict(list)
    for idx, corr in corr_scores:
        avg_corr[idx].append(corr)
    for idx, evs in evs_scores:
        avg_evs[idx].append(evs)
    for idx, r2 in r2_scores:
        avg_r2[idx].append(r2)

    perf_df = pd.DataFrame(columns=['Node', 'Correlation', 'ExplainedVariance', 'R2'])
    for idx in range(len(regressors)):
        perf_df.loc[idx] = {
            'Node': idx,
            'Correlation': np.mean(avg_corr[idx]) if idx in avg_corr else np.nan,
            'ExplainedVariance': np.mean(avg_evs[idx]) if idx in avg_evs else np.nan,
            'R2': np.mean(avg_r2[idx]) if idx in avg_r2 else np.nan
        }
    print("Performance metrics per node:")
    print(perf_df)

    # Combine validation data
    val_X_combined = pd.concat(val_X_list, ignore_index=True)
    val_y_combined = pd.concat(val_y_list, ignore_index=True)
    val_y_combined.columns = ['val_y']

    node_count = len(regressors)

    return (node_train_preds, node_test_preds, perf_df, node_count,
            val_y_combined, val_X_combined, regressors)