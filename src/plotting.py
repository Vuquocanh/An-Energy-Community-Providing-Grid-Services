"""Matplotlib figures for the input data and the optimisation results.

Every function returns the ``Figure`` and optionally saves it, so the same code works in a
script (``python main.py``) and in a notebook (``plot_schedule(results, data);``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import InputData
from .model import Results


def _finish(fig: plt.Figure, save_to: Path | str | None) -> plt.Figure:
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_inputs(data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly prices (with tariffs) and available PV / load preferences, side by side."""
    h = data.hours
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.step(h, data.energy_price, where="mid",
             label="energy price", color="k")
    ax1.step(h, data.energy_price + data.import_tariff,
             where="mid", ls="--", label="price + import tariff")
    ax1.step(h, data.energy_price - data.export_tariff,
             where="mid", ls=":", label="price - export tariff")
    ax1.set(xlabel="hour", ylabel="DKK/kWh", title="Electricity prices")
    ax1.legend(fontsize=8)

    ax2.fill_between(h, data.pv_available, step="mid",
                     alpha=0.4, color="orange", label="PV available")
    ax2.axhline(data.load_max_kWh, color="C3", ls="--", label="max load")
    if data.load_min_kWh > 0:
        ax2.axhline(data.load_min_kWh, color="C3", ls=":", label="min load")
    if data.reference_load is not None:
        ax2.step(h, data.reference_load, where="mid",
                 color="C0", label="reference load")
    ax2.set(xlabel="hour", ylabel="kWh/h", title="PV and load preferences")
    ax2.legend(fontsize=8)
    fig.suptitle(f"Input data - {data.question}", fontsize=11)
    return _finish(fig, save_to)


def plot_schedule(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Optimal schedule: load, PV used, import/export, with prices on a second axis."""
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.2))

    width = 0.8
    if "load" in hr:
        ax.bar(h, hr["load"], width, color="C0", alpha=0.7, label="load")
    if "pv" in hr:
        ax.bar(h, -hr["pv"], width, color="orange", alpha=0.7,
               label="PV used (negative = generation)")
    if "pv_available" in hr and "pv" in hr:
        ax.step(h, -hr["pv_available"], where="mid",
                color="orange", ls="--", lw=1, label="PV available")
    if "import" in hr and "export" in hr:
        ax.plot(h, hr["import"] - hr["export"], "k.-",
                label="net import (+) / export (-)")
    if "reference_load" in hr:
        ax.step(h, hr["reference_load"], where="mid",
                color="C0", ls=":", label="reference load")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="hour", ylabel="kWh/h",
           title=f"Optimal schedule - {results.question} (cost {results.objective:.1f} DKK)")

    ax2 = ax.twinx()
    ax2.step(h, hr["price"], where="mid", color="C3",
             lw=1.2, label="energy price")
    ax2.set_ylabel("DKK/kWh", color="C3")

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)


def plot_duals(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly dual variables (all ``dual_*`` columns) against the price signals."""
    hr = results.hourly
    dual_cols = [c for c in hr.columns if c.startswith("dual_")]
    fig, ax = plt.subplots(figsize=(11, 4))
    h = hr.index.to_numpy()
    for c in dual_cols:
        ax.step(h, hr[c], where="mid", label=c.removeprefix("dual_"))
    ax.step(h, data.energy_price + data.import_tariff, where="mid",
            color="grey", ls="--", lw=1, label="price + import tariff")
    ax.step(h, data.energy_price - data.export_tariff, where="mid",
            color="grey", ls=":", lw=1, label="price - export tariff")
    ax.set(xlabel="hour", ylabel="DKK/kWh",
           title=f"Dual variables - {results.question}")
    ax.legend(fontsize=8, ncol=3)
    return _finish(fig, save_to)


def plot_disutility_sweep(
        df, price_import=(), price_export=(), base_value: float | None = None,
        xlabel: str = 'disutility coefficient', logx: bool = False,
        save_to: Path | str | None = None,
) -> plt.Figure:
    '''Metrics of a disutility-coefficient sweep. For 2(b) and 2(c), iv.

    Top: total absolute deviation (left axis, kWh) and number of deviating hours (right).
    Bottom: cost components (DKK). Vertical grey lines mark the distinct hourly effective
    import (dashed) and export (dotted) prices - the values where the hypotheses of
    2.(b).ii predict the steps of the linear model.
    '''
    x = df.index.to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)

    for ax in (ax1, ax2):
        for v in np.unique(price_import):
            ax.axvline(v, color="grey", ls="--", lw=0.5, alpha=0.6)
        for v in np.unique(price_export):
            ax.axvline(v, color="grey", ls=":", lw=0.5, alpha=0.6)
        if base_value is not None:
            ax.axvline(base_value, color="C3", lw=1.0, alpha=0.8)

    ax1.plot(x, df["abs_deviation_kWh"], ".-",
             color="C0", label="total |deviation| [kWh]")
    ax1.plot(x, df["energy_kWh"], ".-", color="C2",
             label="daily energy consumed [kWh]")
    ax1.set_ylabel("kWh")
    ax1b = ax1.twinx()
    ax1b.plot(x, df["hours_deviating"], ".-",
              color="C1", label="# deviating hours")
    ax1b.set_ylabel("hours", color="C1")
    lines, labels = ax1.get_legend_handles_labels()
    l2, lb2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lb2, fontsize=8, loc="upper right")
    title = "Sweep of the disutility coefficient"
    if len(np.atleast_1d(price_import)) or len(np.atleast_1d(price_export)):
        title += " (grey lines: hourly effective prices)"
    if base_value is not None:
        title += " - red: base value"
    ax1.set_title(title, fontsize=10)

    ax2.plot(x, df["procurement_DKK"], ".-", label="procurement cost")
    ax2.plot(x, df["disutility_DKK"], ".-", label="total disutility")
    ax2.plot(x, df["total_cost_DKK"], "k.-", label="total cost (objective)")
    ax2.set(xlabel=xlabel, ylabel="DKK")
    ax2.legend(fontsize=8)
    if logx:
        ax1.set_xscale("log")
    return _finish(fig, save_to)


def plot_load_comparison(runs, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """For Question 2(d) --> the optimal load of several models on one shared axis, versus the
    reference profile and the available PV, with prices on a secondary axis.

    ``runs`` maps a label (e.g. "Q1 (linear utility)") to a :class:`Results`.
    """
    fig, ax = plt.subplots(figsize=(11, 4.4))
    h = data.hours
    ax.step(h, data.reference_load, where="mid", color="k",
            ls=":", lw=1.4, label="reference profile")
    ax.step(h, data.pv_available, where="mid", color="orange",
            ls="--", lw=1.1, label="PV available")
    for (label, res), color in zip(runs.items(), ("C0", "C2", "C3", "C4")):
        ax.step(h, res.hourly["load"], where="mid",
                color=color, lw=1.6, label=f"load - {label}")
    ax.set(xlabel="hour", ylabel="kWh/h",
           title="Optimal load profiles across objective functions (base cases)")

    ax2 = ax.twinx()
    ax2.step(h, data.energy_price, where="mid", color="grey",
             lw=0.9, alpha=0.7, label="energy price")
    ax2.set_ylabel("DKK/kWh", color="grey")

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)


def plot_scenario_comparison(
    runs: dict[str, Results], metric: str = "objective", save_to: Path | str | None = None
) -> plt.Figure:
    """Bar chart of one metric across scenarios. ``metric`` is ``"objective"`` or the name of an
    hourly column whose daily sum is compared (e.g. ``"import"``, ``"export"``, ``"load"``)."""
    names = list(runs)
    if metric == "objective":
        values = [r.objective for r in runs.values()]
        ylabel = "daily cost [DKK]"
    else:
        values = [r.hourly[metric].sum() for r in runs.values()]
        ylabel = f"daily {metric} [kWh]"
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names)), 3.8))
    ax.bar(names, values, color="C0")
    ax.set(ylabel=ylabel, title=f"Scenario comparison - {metric}")
    ax.tick_params(axis="x", rotation=20)
    return _finish(fig, save_to)

def plot_schedule_q3_comparison(results_q3, results_q2c, data, save_to=None):
    fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=150)
    
    h = results_q3.hourly          
    hours = h.index

    ax1.bar(hours, h["load"], width=0.4, label="Q3 load (with $E_{\min}$)", color="#5B9BD5", alpha=0.7)

    if "reference_load" in h.columns:
        ax1.plot(hours, h["reference_load"], linestyle=":", color="#1F4E79", linewidth=2, label="reference load ($\ell_t^{ref}$)")

    if results_q2c is not None:
        ax1.plot(
            hours, 
            results_q2c.hourly["load"], 
            linestyle="--", 
            color="#ED7D31", 
            linewidth=2, 
            marker="s", 
            markersize=3.5, 
            label="Q2.(c) load (unconstrained)"
        )

    if "import" in h.columns and "export" in h.columns:
        ax1.plot(hours, h["import"] - h["export"], marker="o", color="black", linewidth=1.2, label="net import (+) / export (-)")

    ax2 = ax1.twinx()
    ax2.plot(hours, data.energy_price, color="#C00000", linestyle="-", label="energy price")
    ax2.set_ylabel("DKK/kWh", color="#C00000")

    ax1.set_xlabel("hour")
    ax1.set_ylabel("kWh/h")
    ax1.set_title(f"Optimal schedule - {results_q3.question} (cost {results_q3.objective:.1f} DKK)")  

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    return _finish(fig, save_to)


def plot_duals_q3(results_q3, data, save_to=None):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    h = results_q3.hourly          
    hours = h.index

    if "dual_balance" in h.columns:
        ax.plot(hours, h["dual_balance"], label="dual balance ($\lambda_t$)", color="#1F77B4", linewidth=1.5)
    if "dual_pv_max" in h.columns:
        ax.plot(hours, h["dual_pv_max"], label="dual pv_max", color="#FF7F0E", linewidth=1.5)

    p_imp = data.energy_price + data.import_tariff
    p_exp = data.energy_price - data.export_tariff
    ax.plot(hours, p_imp, linestyle="--", color="gray", alpha=0.6, label="price + import tariff")
    ax.plot(hours, p_exp, linestyle=":", color="gray", alpha=0.6, label="price - export tariff")

    if "min_daily_energy" in results_q3.duals:
        mu_val = abs(results_q3.duals["min_daily_energy"]) 
        ax.axhline(y=mu_val, color="red", linestyle="--", linewidth=1.5, label=f"$\mu$ (min energy dual = {mu_val:.2f} DKK/kWh)")

    ax.set_xlabel("hour")
    ax.set_ylabel("DKK/kWh")
    ax.set_title(f"Dual variables - {results_q3.question}") 
    ax.legend(loc="upper right", fontsize=8.5)

    return _finish(fig, save_to)

import pandas as pd

def plot_q3_emin_sensitivity(df_emin: pd.DataFrame, save_to: Path | str = None, show: bool = False):
    """Q3f:  E_min Monitoring - the impact of the dual multiplier mu_daily 
        and the total daily cost"""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "#1f77b4"  # Blue - Duality multiplier
    color2 = "#d62728"  # Red - Objective Function / Total Costs

    # left y-axis: mu^daily
    ax1.set_xlabel("Minimum Daily Energy Requirement $E_{\\mathrm{min}}$ (kWh)", fontsize=11)
    ax1.set_ylabel("Dual Variable $\\mu^{\\mathrm{daily}}$ (DKK/kWh)", color=color1, fontsize=11)
    line1 = ax1.step(
        df_emin["E_min"], df_emin["mu_daily"], where="post", color=color1, linewidth=2, label="Dual $\\mu^{\\mathrm{daily}}$"
    )
    ax1.scatter(df_emin["E_min"], df_emin["mu_daily"], color=color1, zorder=5)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # right y-axis: Objective
    ax2 = ax1.twinx()
    ax2.set_ylabel("Total Daily Cost / Objective (DKK)", color=color2, fontsize=11)
    line2 = ax2.plot(
        df_emin["E_min"], df_emin["objective"], color=color2, linewidth=2, linestyle="-.", label="Total Cost"
    )
    ax2.scatter(df_emin["E_min"], df_emin["objective"], color=color2, zorder=5)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Mark Slack Intervals and Binding Intervals
    slack_mask = df_emin["mu_daily"] == 0
    if slack_mask.any():
        knee_point = df_emin[slack_mask]["E_min"].max()
        ax1.axvline(x=knee_point, color="gray", linestyle=":", linewidth=1.5)
        ax1.text(
            knee_point * 0.5,
            df_emin["mu_daily"].max() * 0.5,
            "Inactive Zone\n$(\\mu = 0)$",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    plt.title("Sensitivity to Minimum Daily Energy Requirement ($E_{\\mathrm{min}}$)", fontsize=12, fontweight="bold")
    fig.tight_layout()

    plt.savefig(save_to, dpi=300)
    plt.close()


def plot_q3_cq_profiles(cq_results: dict[float, pd.DataFrame], base_data, save_to: Path | str = None, show: bool = False):
    """Q3f:  Monitoring c_Q - 24-hour Load Profile and Electricity Price Comparison
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    hours = range(base_data.n_hours)
    ref_load = getattr(base_data, "reference_load", getattr(base_data, "l_ref", None))

    # fig 1: Actual load under base load with different c_Q  (L_t)
    if ref_load is not None:
        ax1.plot(hours, ref_load, color="black", linestyle="--", linewidth=2, label="Reference Load $\\ell_t^{\\mathrm{ref}}$")

    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#1f77b4"]
    for i, (cq, hourly_df) in enumerate(cq_results.items()):
        c = colors[i % len(colors)]
        ax1.plot(hours, hourly_df["load"], marker="o", markersize=4, linewidth=2, color=c, label=f"$c_Q = {cq}$ DKK/kWh$^2$")

    ax1.set_ylabel("Hourly Load (kWh)", fontsize=11)
    ax1.set_title("Optimal Load Profiles under Different Quadratic Disutility $c_Q$", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # fig 2: Import Electricity Price (pi_imp)
    pi_imp = base_data.energy_price + base_data.import_tariff
    ax2.bar(hours, pi_imp, color="#7f7f7f", alpha=0.4, width=0.6, label="Effective Import Price $\\pi_t^+$")
    ax2.set_xlabel("Hour of Day ($t$)", fontsize=11)
    ax2.set_ylabel("Price (DKK/kWh)", fontsize=10)
    ax2.set_xticks(hours)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    plt.savefig(save_to, dpi=300)
    plt.close()

def plot_q3_battery_schedule(
    results: Results, data: InputData, save_to: Path | str | None = None
) -> plt.Figure:
    """Optimal schedule with battery storage for Question 3.(g).

    Plots:
    - Top panel: Flexible load, reference load, net import/export, and electricity price.
    - Bottom panel: Battery charging/discharging power and State of Charge (SoC).
    """
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, dpi=150)

    width = 0.35

    # ==================== Top Panel: Power Flow & Load ====================
    if "load" in hr:
        ax1.plot(h, hr["load"], "o-", color="#1f77b4", linewidth=2, label="Flexible load $L_t$")
    if "reference_load" in hr:
        ax1.plot(h, hr["reference_load"], "k--", linewidth=1.5, label="Reference load $\ell_t^{\mathrm{ref}}$")
    elif data.reference_load is not None:
        ax1.plot(h, data.reference_load, "k--", linewidth=1.5, label="Reference load $\ell_t^{\mathrm{ref}}$")

    if "import" in hr and "export" in hr:
        ax1.plot(h, hr["import"] - hr["export"], "s-", color="black", linewidth=1.2, alpha=0.7, label="Net import (+) / export (-)")

    if "pv" in hr and (hr["pv"] > 1e-4).any():
        ax1.plot(h, hr["pv"], "^:", color="orange", linewidth=1.5, label="PV used")

    ax1.set_ylabel("Power / Energy (kWh/h)", fontsize=10)
    ax1.set_title(
        f"Optimal Schedule with Battery - {results.question} (Total Cost: {results.objective:.2f} DKK)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Top Panel Right Axis: Price
    ax1_twin = ax1.twinx()
    p_imp = data.energy_price + data.import_tariff
    ax1_twin.step(h, p_imp, where="mid", color="#d62728", linestyle="-", linewidth=1.2, alpha=0.8, label="Import price $\pi_t^+$")
    ax1_twin.set_ylabel("DKK/kWh", color="#d62728", fontsize=10)
    ax1_twin.tick_params(axis="y", labelcolor="#d62728")

    # Combine legends for top panel
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1_t, labels1_t = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines1_t, labels1 + labels1_t, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8.5)

    # ==================== Bottom Panel: Battery Dynamics ====================
    p_ch = hr["p_ch"] if "p_ch" in hr else np.zeros(len(h))
    p_dis = hr["p_dis"] if "p_dis" in hr else np.zeros(len(h))

    ax2.bar(h - width / 2, p_ch, width, label="Charging $P^{\mathrm{ch}}_t$", color="#2ca02c", alpha=0.8)
    ax2.bar(h + width / 2, p_dis, width, label="Discharging $P^{\mathrm{dis}}_t$", color="#ff7f0e", alpha=0.8)
    ax2.set_ylabel("Battery Power (kW)", fontsize=10)
    ax2.set_xlabel("Hour of Day ($t$)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Bottom Panel Right Axis: State of Charge (SoC)
    ax2_twin = ax2.twinx()
    if "soc" in hr:
        soc_vals = hr["soc"]
        ax2_twin.plot(h, soc_vals, "D-", color="#9467bd", linewidth=2, markersize=4, label="Battery SoC (kWh)")

    if data.battery_capacity_kWh is not None:
        ax2_twin.axhline(data.battery_capacity_kWh, color="#9467bd", linestyle="--", linewidth=1, alpha=0.7, label="Capacity limit")

    ax2_twin.set_ylabel("SoC (kWh)", color="#9467bd", fontsize=10)
    ax2_twin.tick_params(axis="y", labelcolor="#9467bd")

    # Combine legends for bottom panel
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2_t, labels2_t = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines2 + lines2_t, labels2 + labels2_t, loc="upper right", fontsize=8.5)

    ax2.set_xticks(h)

    return _finish(fig, save_to)

def plot_q3g_sensitivity(df_spread: pd.DataFrame, df_cap: pd.DataFrame, save_to: Path | str | None = None) -> plt.Figure:
    """Plot battery value sensitivity analysis for Q3.(g).v."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    # Left: Environment Sensitivity (Price Spread Scale)
    ax1.plot(df_spread["price_spread_scale"], df_spread["value_of_battery"], "o-", color="#1f77b4", linewidth=2, markersize=6)
    ax1.set_xlabel("Price Spread Scale Factor ($\\alpha_{\\mathrm{spread}}$)", fontsize=10)
    ax1.set_ylabel("Value of Battery (DKK/day)", fontsize=10)
    ax1.set_title("Sensitivity to Market Environment\n(Price Volatility / Spread)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right: Technical Sensitivity (Battery Capacity)
    ax2.plot(df_cap["capacity_kWh"], df_cap["value_of_battery"], "s-", color="#2ca02c", linewidth=2, markersize=6)
    ax2.axvline(4.0, color="gray", linestyle=":", label="Base Capacity (4 kWh)")
    ax2.set_xlabel("Battery Capacity $E_{\\mathrm{bat}}^{\\mathrm{cap}}$ (kWh)", fontsize=10)
    ax2.set_ylabel("Value of Battery (DKK/day)", fontsize=10)
    ax2.set_title("Sensitivity to Technical Spec\n(Battery Storage Capacity)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    return _finish(fig, save_to)
