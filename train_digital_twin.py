"""
Digital Twin: stacked ensemble regressor predicting Reactor Pressure
from all other TEP process variables, trained on normal operation only.

Architecture: Random Forest + Gradient Boosting (base models) ->
Linear Regression (meta-learner), via sklearn's StackingRegressor
(which internally uses cross-validated base predictions to avoid leakage).

Reads tep_train_renamed.parquet / tep_test_renamed.parquet.
Saves the trained twin (joblib) and a residuals file for later use
by the fault classifier / agent layer.

Usage:
    python train_digital_twin.py /path/to/data/folder
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "reactor_pressure"

# Columns that are metadata, not process variables -> exclude from features
TAG_COLUMNS = ["fault_number", "split", "run_id", "sample_index"]


def load_normal_data(data_dir: Path):
    train_df = pd.read_parquet(data_dir / "tep_train_renamed.parquet")
    test_df = pd.read_parquet(data_dir / "tep_test_renamed.parquet")

    # Twin only learns from normal operation (fault_number == 0)
    train_normal = train_df[train_df["fault_number"] == 0].reset_index(drop=True)
    test_normal = test_df[test_df["fault_number"] == 0].reset_index(drop=True)

    return train_normal, test_normal


def split_features_target(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in TAG_COLUMNS + [TARGET]]
    X = df[feature_cols]
    y = df[TARGET]
    return X, y, feature_cols


def build_twin() -> StackingRegressor:
    base_models = [
        ("random_forest", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)),
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)),
    ]
    meta_learner = LinearRegression()
    twin = StackingRegressor(estimators=base_models, final_estimator=meta_learner, cv=5, n_jobs=-1)
    return twin


def evaluate(y_true, y_pred, label: str):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return rmse, mae, r2


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_digital_twin.py /path/to/data/folder")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    train_normal, test_normal = load_normal_data(data_dir)

    print(f"Training rows (normal only): {len(train_normal)}")
    print(f"Testing rows (normal only): {len(test_normal)}")

    X_train, y_train, feature_cols = split_features_target(train_normal)
    X_test, y_test, _ = split_features_target(test_normal)

    print(f"\nTraining stacked ensemble to predict '{TARGET}' from {len(feature_cols)} features...")
    twin = build_twin()
    twin.fit(X_train, y_train)

    train_pred = twin.predict(X_train)
    test_pred = twin.predict(X_test)

    print("\n--- Performance ---")
    evaluate(y_train, train_pred, "Train")
    evaluate(y_test, test_pred, "Test")

    # Save residuals (actual - predicted) for the test set -> this is what
    # the agent layer will later query to flag anomalies
    residuals_df = test_normal[TAG_COLUMNS].copy()
    residuals_df["actual"] = y_test.values
    residuals_df["predicted"] = test_pred
    residuals_df["residual"] = residuals_df["actual"] - residuals_df["predicted"]

    out_dir = data_dir
    joblib.dump(twin, out_dir / "digital_twin_reactor_pressure.joblib")
    residuals_df.to_parquet(out_dir / "twin_residuals_normal_test.parquet", index=False)

    print(f"\nSaved model to {out_dir / 'digital_twin_reactor_pressure.joblib'}")
    print(f"Saved residuals to {out_dir / 'twin_residuals_normal_test.parquet'}")
    print(f"\nResidual stats: mean={residuals_df['residual'].mean():.4f}, "
          f"std={residuals_df['residual'].std():.4f}")
