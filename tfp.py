"""
Traditional ensemble methods for regression: averaging, blending, stacking.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
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


def evaluate_individual_regressors(X_train, X_test, y_train, y_test):
    """
    Train and evaluate each regressor individually, return predictions and R2 scores.
    """
    regressors = [
        XGBRegressor(),
        CatBoostRegressor(verbose=False),
        LGBMRegressor(),
        AdaBoostRegressor(),
        GradientBoostingRegressor(),
        RandomForestRegressor(),
        HistGradientBoostingRegressor(),
        ExtraTreesRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
        BaggingRegressor(),
        KNeighborsRegressor(),
        svm.SVR(),
        MLPRegressor(max_iter=500, solver='adam', hidden_layer_sizes=128),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1),
        ElasticNet(alpha=0.1, l1_ratio=0.5)
    ]

    y_preds = np.zeros((len(X_test), len(regressors)))
    r2_scores = {}

    print("================== Individual Regressor Performance ==================")
    for idx, reg in enumerate(regressors):
        print(f"Round {idx+1}: {type(reg).__name__}")
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        y_preds[:, idx] = y_pred

        r2 = r2_score(y_test, y_pred)
        r2_scores[idx] = r2
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        corr, p = pearsonr(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)

        print(f"  R2: {r2:.6f}, MAE: {mae:.6f}, MSE: {mse:.6f}, "
              f"Corr: {corr:.6f}, p: {p:.6f}, EVS: {evs:.6f}")

    return regressors, y_preds, r2_scores


def remove_worst_regressor(regressors, y_val_preds, y_test_preds, y_val, y_test):
    """
    Remove the worst performing regressor based on sub-layer evaluation.
    """
    sorted_results = evaluate_sub_layer(regressors, y_val_preds, y_test_preds, y_val, y_test)
    worst_name = sorted_results[-1][0]
    worst_idx = next(i for i, r in enumerate(regressors) if type(r).__name__ == worst_name)

    regressors.pop(worst_idx)
    y_val_preds = np.delete(y_val_preds, worst_idx, axis=1)
    y_test_preds = np.delete(y_test_preds, worst_idx, axis=1)

    return regressors, y_val_preds, y_test_preds


def evaluate_sub_layer(regressors, y_val_preds, y_test_preds, y_val, y_test):
    """
    Train sub-layer regressors on validation predictions and sort by R2.
    """
    results = []
    for reg in regressors:
        name = type(reg).__name__
        reg.fit(y_val_preds, y_val)
        y_pred = reg.predict(y_test_preds)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        corr, _ = pearsonr(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)

        results.append((name, r2, mae, mse, corr, evs))

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    print("Sub-layer ranking (by R2):", [(n, r2) for n, r2, *_ in sorted_results])
    return sorted_results


def tfp_averagingandsingle(X_train, X_test, y_train, y_test):
    """
    Ensemble averaging (mean of predictions) with iterative removal of worst models.
    """
    regressors, y_preds, r2_scores = evaluate_individual_regressors(
        X_train, X_test, y_train, y_test
    )
    sorted_indices = sorted(r2_scores, key=r2_scores.get)

    best_r2 = -float('inf')
    deletion_info = []

    print("================== Ensemble Averaging ==================")
    for i in range(len(regressors) - 1):
        remaining = sorted_indices[i:]
        mean_pred = y_preds[:, remaining].mean(axis=1)
        current_r2 = r2_score(y_test, mean_pred)

        if current_r2 > best_r2:
            best_r2 = current_r2
            deletion_info = [
                f"Model {type(regressors[sorted_indices[k]]).__name__} "
                f"(R2: {r2_scores[sorted_indices[k]]:.4f})"
                for k in range(i)
            ]

        print(f"Removed {i} models, Current R2: {current_r2:.6f}, Deleted: {deletion_info}")
    print("================== End Averaging ==================")


def tfp_blending(X_train, X_test, y_train, y_test):
    """
    Blending ensemble with hold-out validation set.
    """
    print("================== Blending Ensemble ==================")
    X_base, X_hold, y_base, y_hold = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )

    regressors = [
        XGBRegressor(),
        CatBoostRegressor(verbose=False),
        LGBMRegressor(),
        AdaBoostRegressor(),
        GradientBoostingRegressor(),
        RandomForestRegressor(),
        HistGradientBoostingRegressor(),
        ExtraTreesRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
        BaggingRegressor(),
        KNeighborsRegressor(),
        svm.SVR(),
        MLPRegressor(max_iter=500, solver='adam', hidden_layer_sizes=128),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1),
        ElasticNet(alpha=0.1, l1_ratio=0.5)
    ]

    y_hold_preds = np.zeros((len(X_hold), len(regressors)))
    y_test_preds = np.zeros((len(X_test), len(regressors)))

    for idx, reg in enumerate(regressors):
        reg.fit(X_base, y_base)
        y_hold_preds[:, idx] = reg.predict(X_hold)
        y_test_preds[:, idx] = reg.predict(X_test)

    while len(regressors) > 0:
        regressors, y_hold_preds, y_test_preds = remove_worst_regressor(
            regressors, y_hold_preds, y_test_preds, y_hold, y_test
        )
    print("================== Blending Complete ==================")


def tfp_stacking(X_train, X_test, y_train, y_test):
    """
    Stacking ensemble with 5-fold cross-validation.
    """
    print("================== Stacking Ensemble ==================")
    regressors = [
        XGBRegressor(),
        CatBoostRegressor(verbose=False),
        LGBMRegressor(),
        AdaBoostRegressor(),
        GradientBoostingRegressor(),
        RandomForestRegressor(),
        HistGradientBoostingRegressor(),
        ExtraTreesRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
        BaggingRegressor(),
        KNeighborsRegressor(),
        svm.SVR(),
        MLPRegressor(max_iter=500, solver='adam', hidden_layer_sizes=128),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1),
        ElasticNet(alpha=0.1, l1_ratio=0.5)
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    y_oof_preds = np.zeros((len(X_train), len(regressors)))  # out-of-fold
    y_test_preds = np.zeros((len(X_test), len(regressors)))

    for idx, reg in enumerate(regressors):
        oof_preds = cross_val_predict(reg, X_train, y_train, cv=kf)
        y_oof_preds[:, idx] = oof_preds
        reg.fit(X_train, y_train)
        y_test_preds[:, idx] = reg.predict(X_test)

    while len(regressors) > 0:
        regressors, y_oof_preds, y_test_preds = remove_worst_regressor(
            regressors, y_oof_preds, y_test_preds, y_train, y_test
        )
    print("================== Stacking Complete ==================")