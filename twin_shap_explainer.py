"""
XGBoost Digital Twin + SHAP Root-Cause Explainer

Trains an XGBoost regressor to predict Reactor Pressure from normal
operation data (same target as the stacked ensemble twin), then uses
SHAP to explain, for any given fault, which input variables are driving
the twin's prediction away from what "healthy" looks like.

This mirrors the Holcim approach: regression + SHAP for root-cause
explanation, rather than a fixed fault-category classifier.

Usage:
    python twin_shap_explainer.py /path/to/data/folder [fault_number]

If fault_number is omitted, defaults to fault 4 (Reactor Cooling Water
Inlet Temperature Step) as a clear, well-documented example.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

TARGET = "reactor_pressure"
TAG_COLUMNS = ["fault_number", "split", "run_id", "sample_index"]

# Faults are introduced partway through each _te test file (~sample 160 of 960).
# Rows before this are still behaving normally, so we skip them when analyzing
# a specific fault's root cause -- otherwise we'd be "explaining" normal behavior.
FAULT_ONSET_SAMPLE = 160


def load_data(data_dir: Path):
    train_df = pd.read_parquet(data_dir / "tep_train_renamed.parquet")
    test_df = pd.read_parquet(data_dir / "tep_test_renamed.parquet")
    return train_df, test_df


def get_features(df: pd.DataFrame):
    return [c for c in df.columns if c not in TAG_COLUMNS + [TARGET]]


def train_twin(train_df: pd.DataFrame, feature_cols: list) -> XGBRegressor:
    normal_train = train_df[train_df["fault_number"] == 0]
    X_train = normal_train[feature_cols]
    y_train = normal_train[TARGET]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_twin(model, test_df, feature_cols):
    normal_test = test_df[test_df["fault_number"] == 0]
    X_test = normal_test[feature_cols]
    y_test = normal_test[TARGET]
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"Twin performance on normal test data -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


def explain_fault(model, test_df, feature_cols, fault_number: int):
    """
    Compares SHAP feature contributions between normal operation and a
    specific fault (post-onset rows only), ranking features by how much
    their contribution shifted -- these are the root-cause candidates.
    """
    explainer = shap.TreeExplainer(model)

    normal_test = test_df[test_df["fault_number"] == 0]
    X_normal = normal_test[feature_cols]
    shap_normal = explainer.shap_values(X_normal)
    baseline_mean_shap = np.mean(shap_normal, axis=0)

    fault_test = test_df[
        (test_df["fault_number"] == fault_number)
        & (test_df["sample_index"] >= FAULT_ONSET_SAMPLE)
    ]
    if fault_test.empty:
        print(f"No post-onset rows found for fault {fault_number}.")
        return

    X_fault = fault_test[feature_cols]
    shap_fault = explainer.shap_values(X_fault)
    fault_mean_shap = np.mean(shap_fault, axis=0)

    predicted = model.predict(X_fault)
    actual = fault_test[TARGET].values
    residual = actual - predicted

    shift = fault_mean_shap - baseline_mean_shap
    shift_series = pd.Series(shift, index=feature_cols).sort_values(key=abs, ascending=False)

    print(f"\n=== Fault {fault_number}: root-cause candidates (post-onset, sample >= {FAULT_ONSET_SAMPLE}) ===")
    print(f"Mean actual reactor pressure: {actual.mean():.2f}")
    print(f"Mean predicted reactor pressure: {predicted.mean():.2f}")
    print(f"Mean residual (actual - predicted): {residual.mean():.4f}")
    print(f"\nTop 10 variables by SHAP contribution shift vs. normal baseline:")
    print(shift_series.head(10).to_string())

    return shift_series


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python twin_shap_explainer.py /path/to/data/folder [fault_number]")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    fault_number = int(sys.argv[2]) if len(sys.argv) == 3 else 4

    train_df, test_df = load_data(data_dir)
    feature_cols = get_features(train_df)

    print(f"Training XGBoost twin on normal data ({(train_df['fault_number'] == 0).sum()} rows)...")
    model = train_twin(train_df, feature_cols)
    evaluate_twin(model, test_df, feature_cols)

    joblib.dump(model, data_dir / "digital_twin_xgb_reactor_pressure.joblib")
    print(f"Saved XGBoost twin to {data_dir / 'digital_twin_xgb_reactor_pressure.joblib'}")

    shift_series = explain_fault(model, test_df, feature_cols, fault_number)
    if shift_series is not None:
        shift_series.to_csv(data_dir / f"shap_root_cause_fault_{fault_number}.csv")
        print(f"\nSaved full SHAP shift ranking to shap_root_cause_fault_{fault_number}.csv")
