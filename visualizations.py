import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAULT_ONSET_SAMPLE = 160


def get_pressure_trend_figure(test_df: pd.DataFrame, model, feature_cols: list,
                               target: str, fault_number: int) -> plt.Figure:
    """
    Line chart: actual vs. predicted reactor pressure over time for a given
    fault's test run, with a vertical line marking fault onset.

    Args:
        test_df: the full test dataframe (all fault classes).
        model: trained twin model with a .predict(X) method.
        feature_cols: list of feature column names used by the model.
        target: name of the target column (e.g. "reactor_pressure").
        fault_number: which fault's test run to plot (0 = normal).

    Returns:
        A matplotlib Figure.
    """
    subset = test_df[test_df["fault_number"] == fault_number].sort_values("sample_index")
    if subset.empty:
        raise ValueError(f"No data found for fault_number={fault_number}")

    X = subset[feature_cols]
    actual = subset[target].values
    predicted = model.predict(X)
    sample_index = subset["sample_index"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sample_index, actual, label="Actual", color="black", linewidth=1.2)
    ax.plot(sample_index, predicted, label="Twin Predicted", color="tab:blue",
            linewidth=1.2, linestyle="--")

    if fault_number != 0:
        ax.axvline(FAULT_ONSET_SAMPLE, color="red", linestyle=":", linewidth=1.5,
                   label="Fault onset")

    ax.set_xlabel("Sample index (time)")
    ax.set_ylabel(target.replace("_", " ").title())
    ax.set_title(f"Fault {fault_number}: Actual vs. Twin-Predicted {target.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def get_shap_contributions_figure(shift_series: pd.Series, fault_number: int,
                                   top_n: int = 10) -> plt.Figure:
    """
    Horizontal bar chart of the top SHAP contribution shifts vs. normal
    baseline for a given fault -- the root-cause ranking.

    Args:
        shift_series: pandas Series indexed by feature name, values = SHAP
            contribution shift vs. baseline (as produced by explain_root_cause
            logic). Should already be sorted by absolute magnitude.
        fault_number: which fault this ranking belongs to (for the title).
        top_n: how many top variables to show.

    Returns:
        A matplotlib Figure.
    """
    top = shift_series.head(top_n).sort_values()  # ascending for horizontal bar order

    colors = ["tab:red" if v < 0 else "tab:blue" for v in top.values]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution shift vs. normal baseline")
    ax.set_title(f"Fault {fault_number}: Top {top_n} Root-Cause Variables")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig
