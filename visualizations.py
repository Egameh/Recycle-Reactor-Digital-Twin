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


def get_pid_convergence_figure(pressure_history: list, valve_history: list,
                                setpoint: float) -> plt.Figure:
    """
    Two-panel chart showing a PID control loop converging: predicted reactor
    pressure approaching the setpoint over iterations (top), and the
    manipulated valve position driving that convergence (bottom).

    Args:
        pressure_history: twin-predicted pressure at each iteration.
        valve_history: manipulated variable (valve %) at each iteration.
        setpoint: target pressure value.

    Returns:
        A matplotlib Figure.
    """
    iterations = np.arange(len(pressure_history))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(iterations, pressure_history, color="tab:blue", linewidth=1.5, label="Twin-predicted pressure")
    ax1.axhline(setpoint, color="black", linestyle="--", linewidth=1.2, label="Setpoint")
    ax1.set_ylabel("Reactor Pressure")
    ax1.set_title("PID Control Loop: Pressure Convergence")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(iterations, valve_history, color="tab:orange", linewidth=1.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("A Feed Flow Valve (%)")
    ax2.set_title("Manipulated Variable")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def get_cascade_pid_figure(pressure_history: list, inner_var_history: list,
                            inner_setpoint_history: list, pressure_setpoint: float,
                            inner_var_name: str) -> plt.Figure:
    """
    Two-panel chart for a cascade PID loop: outer loop (reactor pressure vs.
    its setpoint) and inner loop (the intermediate variable vs. the moving
    setpoint the outer loop is feeding it).

    Args:
        pressure_history: twin-predicted reactor pressure at each iteration.
        inner_var_history: the intermediate variable's value at each iteration.
        inner_setpoint_history: the moving setpoint the outer loop generated
            for the intermediate variable, at each iteration.
        pressure_setpoint: the outer loop's fixed target.
        inner_var_name: display name of the intermediate variable.

    Returns:
        A matplotlib Figure.
    """
    iterations = np.arange(len(pressure_history))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(iterations, pressure_history, color="tab:blue", linewidth=1.5,
             label="Twin-predicted reactor pressure")
    ax1.axhline(pressure_setpoint, color="black", linestyle="--", linewidth=1.2,
                label="Reactor pressure setpoint")
    ax1.set_ylabel("Reactor Pressure")
    ax1.set_title("Outer Loop: Reactor Pressure")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(iterations, inner_var_history, color="tab:orange", linewidth=1.5,
             label=f"{inner_var_name} (actual)")
    ax2.plot(iterations, inner_setpoint_history, color="tab:green", linestyle="--",
             linewidth=1.2, label=f"{inner_var_name} (setpoint from outer loop)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(inner_var_name.replace("_", " ").title())
    ax2.set_title("Inner Loop: Intermediate Variable Tracking")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def get_strategy_comparison_figure(single_var_history: list, multi_var_history: list,
                                    setpoint: float) -> plt.Figure:
    """
    Overlays two control strategies' predicted-pressure trajectories on one
    chart: a single-variable PID (limited by that variable's local range)
    vs. a joint multi-variable proportional restoration toward the
    fault-free baseline.

    Args:
        single_var_history: predicted pressure at each iteration, single-variable strategy.
        multi_var_history: predicted pressure at each iteration, multi-variable strategy.
        setpoint: target reactor pressure.

    Returns:
        A matplotlib Figure.
    """
    iterations = np.arange(max(len(single_var_history), len(multi_var_history)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(single_var_history)), single_var_history, color="tab:red",
            linewidth=1.5, label="Single-variable PID (stripper pressure)")
    ax.plot(np.arange(len(multi_var_history)), multi_var_history, color="tab:green",
            linewidth=1.5, label="Multi-variable joint restoration")
    ax.axhline(setpoint, color="black", linestyle="--", linewidth=1.2, label="Setpoint")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reactor Pressure")
    ax.set_title("Control Strategy Comparison: Single-Variable vs. Multi-Variable Correction")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
