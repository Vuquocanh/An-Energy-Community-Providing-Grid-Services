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

# ==============================================================================
# Assignment 2 - Question 2: Dynamic Pricing & Contract Analysis
# ==============================================================================

def plot_q2_posted_prices(
    results_pricing: Results,
    com_data,
    results_MC: Results | None = None,
    save_to: Path | str | None = None,
) -> plt.Figure:
    """Question 2: 24-hour posted internal prices vs market benchmarks and Model C shadow prices."""
    hr = results_pricing.hourly
    h = hr.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)

    p = getattr(com_data, "energy_price", hr["price"].to_numpy())
    beta = getattr(getattr(com_data, "contract", None), "tariff_discount_ratio", 0.75)
    tau_imp = getattr(com_data, "import_tariff", 0.5)
    tau_exp = getattr(com_data, "export_tariff", 0.4)

    p_imp_disc = p + (1.0 - beta) * tau_imp
    p_exp_disc = p - (1.0 - beta) * tau_exp

    # Shade cap binding hours
    first_shade = True
    if "exceedance" in hr and "cap" in hr:
        is_cap = (hr["import"] >= hr["cap"] - 1e-3) & (hr["import"] > 1e-3)
        for t in h:
            if is_cap.iloc[t] if isinstance(is_cap, pd.Series) else is_cap[t]:
                ax.axvspan(
                    t - 0.4,
                    t + 0.4,
                    color="orange",
                    alpha=0.15,
                    label="Cap Binding Hour" if first_shade else "",
                )
                first_shade = False

    ax.step(h, p, where="mid", color="gray", linestyle=":", label="Day-ahead Price ($p_t$)", linewidth=1.2)
    ax.step(h, p_imp_disc, where="mid", color="darkred", linestyle="--", label=r"Discounted Import ($p_t + (1-\beta)\tau^{\mathrm{imp}}$)", alpha=0.7)
    ax.step(h, p_exp_disc, where="mid", color="darkgreen", linestyle="--", label=r"Discounted Export ($p_t - (1-\beta)\tau^{\mathrm{exp}}$)", alpha=0.7)

    # Posted prices: always the model's own x. Never substitute the C duals here -- doing so once
    # hid a broken KKT reformulation (all posted prices were zero).
    posted_vals = hr["posted_price"].to_numpy()

    ax.step(h, posted_vals, where="mid", color="#1f77b4", linewidth=2.2, label=r"Posted Internal Price ($x_t$)", zorder=4)

    if results_MC is not None and "dual_balance" in results_MC.hourly:
        ax.plot(h, results_MC.hourly["dual_balance"], "o", color="#d62728", markersize=5, label=r"Model C Dual Price ($\lambda_t^C$)", zorder=5)

    ax.set_xlabel("Hour of Day ($t$)", fontsize=10)
    ax.set_ylabel("Price (DKK/kWh)", fontsize=10)
    ax.set_title(f"Question 2: Posted Price Signals vs. Market Benchmarks ({results_pricing.question})", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.5)

    return _finish(fig, save_to)


def plot_q2_3step_decomposition(
    df_3step: pd.DataFrame, save_to: Path | str | None = None
) -> plt.Figure:
    """Question 2: Comparison across 3-step requirements (cap_only -> BB -> BB_IR)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    variants = df_3step["Variant"].tolist()
    x = np.arange(len(variants))
    width = 0.35

    revs = df_3step["Manager_Revenues_DKK"].values
    dso_costs = df_3step["DSO_Costs_DKK"].values
    residuals = df_3step["Budget_Residual_DKK"].values

    ax1.bar(x - width / 2, revs, width, label="Manager Payments Collected", color="#1f77b4", alpha=0.85)
    ax1.bar(x + width / 2, dso_costs, width, label="Manager DSO Grid Costs", color="#ff7f0e", alpha=0.85)

    ymin = min(np.min(revs), np.min(dso_costs), np.min(residuals))
    ymax = max(np.max(revs), np.max(dso_costs))
    span = ymax - ymin if ymax != ymin else 100.0
    ax1.set_ylim(ymin - 0.25 * span, ymax + 0.25 * span)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    for i in range(len(variants)):
        res = residuals[i]
        val_pos = max(revs[i], dso_costs[i], 0)
        ax1.annotate(
            f"Net Residual:\n{res:+.1f} DKK",
            xy=(x[i], val_pos + 0.05 * span),
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontweight="bold")
    ax1.set_ylabel("Amount (DKK)", fontsize=10)
    ax1.set_title("Manager Financial Balance & Budget Residual", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    costs = df_3step["Objective_Cost_DKK"].values
    ir_viols = df_3step["IR_Violations_Count"].values

    ax2_color = "#2ca02c"
    ax2.bar(x, costs, width=0.4, color=ax2_color, alpha=0.85, label="Total Objective Cost (DKK)")
    ax2.set_ylabel("Total Objective Cost (DKK)", color=ax2_color, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=ax2_color)
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, fontweight="bold")
    ax2.set_ylim(0, max(costs) * 1.25)

    for i in range(len(variants)):
        # 优化柱内文字排版，防止边缘裁切
        ax2.annotate(
            f"{costs[i]:.1f} DKK\n(Rec: {df_3step['Welfare_Recovery_Pct'].iloc[i]:.0f}%)",
            xy=(x[i], costs[i] * 0.5),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=7.5,
        )

    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, ir_viols, "D--", color="#d62728", linewidth=2, markersize=7, label="IR Violation Count")
    ax2_twin.set_ylabel("Number of Non-Rational Members", color="#d62728", fontsize=10)
    ax2_twin.tick_params(axis="y", labelcolor="#d62728")
    ax2_twin.set_ylim(-0.2, max(max(ir_viols) + 1.5, 3))

    ax2.set_title("Community Objective Cost & IR Violations", fontsize=11, fontweight="bold")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    return _finish(fig, save_to)


def plot_q2_member_gains_and_blocking(
    member_costs_df: pd.DataFrame,
    title_suffix: str = "Model BB",
    benchmark_costs: dict[str, float] | None = None,
    save_to: Path | str | None = None,
) -> plt.Figure:
    """Question 2: Individual Rationality (IR) member-by-member gain & blocking identification."""
    df = member_costs_df.copy()

    t_col = "total_cost" if "total_cost" in df.columns else ("total" if "total" in df.columns else None)
    
    if "benchmark_cost" not in df.columns and benchmark_costs is not None:
        df["benchmark_cost"] = [benchmark_costs.get(m, 0.0) for m in df.index]

    if t_col and "benchmark_cost" in df.columns:
        df["net_gain"] = df["benchmark_cost"] - df[t_col]
    elif "ir_violation_or_gain" in df.columns:
        df["net_gain"] = -df["ir_violation_or_gain"]
    elif t_col:
        df["net_gain"] = 0.0
    else:
        raise KeyError("DataFrame must contain 'total_cost' or 'total' column.")

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)

    members = df.index.tolist()
    gains = df["net_gain"].values
    x = np.arange(len(members))

    colors = ["#2ca02c" if g >= -1e-4 else "#d62728" for g in gains]
    ax.bar(x, gains, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)

    # Overlay the IR compensation of the relaxation (soft-IR slack or extended transfer)
    comp_col = next((c for c in ("ir_slack", "delta_transfer")
                     if c in df.columns and (df[c] > 1e-4).any()), None)
    if comp_col:
        ax.bar(x, df[comp_col].values, color="#1f77b4", alpha=0.5, hatch="//",
               label=r"IR compensation ($s_i$)" if comp_col == "ir_slack" else r"Transfer ($\delta_i$)")

    ax.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax.set_xticks(x)
    labels = [
        f"{m}\n({df['archetype'].iloc[i]})" if "archetype" in df.columns else str(m)
        for i, m in enumerate(members)
    ]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)

    ax.set_ylabel("Net Savings / Benefit (DKK)", fontsize=10)
    ax.set_title(f"Individual Member Rationality (IR) & Benefit Distribution ({title_suffix})", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Annotations
    for i, g in enumerate(gains):
        if g < -1e-4:
            ax.annotate(
                f"Blocking\n({g:.2f} DKK)",
                xy=(x[i], g),
                xytext=(x[i], g - 2.5),
                ha="center",
                va="top",
                fontsize=7.5,
                fontweight="bold",
                color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
            )
        else:
            ax.annotate(
                f"+{g:.1f}",
                xy=(x[i], max(g, 0)),
                xytext=(x[i], max(g, 0) + 0.3),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#2ca02c",
            )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", alpha=0.85, label="IR Satisfied (Net Gain > 0)"),
        Patch(facecolor="#d62728", alpha=0.85, label="IR Violated / Blocking Member (Gain < 0)"),
    ]
    if comp_col:
        legend_elements.append(
            Patch(facecolor="#1f77b4", alpha=0.5, hatch="//",
                  label=r"Required IR compensation ($s_i$)" if comp_col == "ir_slack"
                  else r"Required lump-sum transfer ($\delta_i$)")
        )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=8.5)

    # 扩大顶部 Y 轴边界，防止图例遮挡数字
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 4, ymax + 20)

    return _finish(fig, save_to)

def plot_q2_cap_sensitivity(
    df_cap_sens: pd.DataFrame, save_to: Path | str | None = None
) -> plt.Figure:
    """Question 2: Sensitivity study on feeder capacity limit (cap tightness f_cap).
    
    Panel 1: Objective Costs (Model M^C, Model M^R, M_{cap}^P, M_{BB}^P) vs Cap Factor.
    Panel 2: Number of Blocking Members & Total Compensation Transfer sum(delta_i).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=150)

    x = df_cap_sens["cap_factor"].values

    ax1.plot(x, df_cap_sens["cost_MR_DKK"], "k--", label=r"Reference Baseline ($M^R$)", linewidth=1.5)
    ax1.plot(x, df_cap_sens["cost_MC_DKK"], "o-", color="#1f77b4", label=r"Direct Control Benchmark ($M^C$)", linewidth=2.0)
    ax1.plot(x, df_cap_sens["cost_cap_only_DKK"], "s:", color="#2ca02c", label=r"Pricing Cap Only ($M_{cap}^P$)", linewidth=1.8)

    if "cost_BB_DKK" in df_cap_sens.columns:
        ax1.plot(x, df_cap_sens["cost_BB_DKK"], "^-.", color="#ff7f0e",
                 label=r"Pricing Budget Balance ($M_{BB}^P$)", linewidth=1.5)
        # Points whose MILP hit its time limit are upper bounds: mark them hollow.
        if "bb_status" in df_cap_sens.columns:
            limited = df_cap_sens["bb_status"].astype(str) != "OPTIMAL"
            if limited.any():
                ax1.plot(x[limited.values], df_cap_sens["cost_BB_DKK"].values[limited.values], "^",
                         markersize=9, markerfacecolor="white", markeredgecolor="#ff7f0e",
                         linestyle="none", label="BB time-limited (upper bound)")

    if "cost_soft_ir_DKK" in df_cap_sens.columns:
        ax1.plot(x, df_cap_sens["cost_soft_ir_DKK"], "d:", color="#9467bd",
                 label=r"Soft IR, minimum compensation ($M_{\mathrm{soft}}^P$)", linewidth=1.6)

    infeasible_mask = ~df_cap_sens["is_BB_IR_feasible"].astype(bool)
    if infeasible_mask.any():
        x_inf_max = df_cap_sens[infeasible_mask]["cap_factor"].max()
        ax1.axvspan(min(x) - 0.05, x_inf_max + 0.05, color="#d62728", alpha=0.12,
                    label="BB+IR not attained under uniform prices (proved infeasible or not found)")

    ax1.set_ylabel("Total Community Cost (DKK)", fontsize=10)
    ax1.set_title(r"Question 2: Sensitivity to Feeder Cap Tightness ($f_{\mathrm{cap}}$)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5)

    color_block = "#d62728"
    ax2.plot(x, df_cap_sens["num_blocking_members"], "D-", color=color_block, linewidth=2, markersize=6, label="Blocking Members Count")
    ax2.set_ylabel("Number of Blocking Members", color=color_block, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color_block)
    ax2.set_xlabel(r"Cap factor $f_{\mathrm{cap}}$ ($1.0$ = shipped cap, 9.39 kW on d1)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    if "total_ir_slack_DKK" in df_cap_sens.columns:
        ax2_twin = ax2.twinx()
        delta_vals = df_cap_sens["total_ir_slack_DKK"].fillna(0.0).values
        ax2_twin.bar(x, delta_vals, width=0.04, color="#9467bd", alpha=0.4,
                     label=r"Required IR compensation ($\sum_i s_i$)")
        ax2_twin.set_ylabel(r"IR compensation $\sum_i s_i$ (DKK)", color="#9467bd", fontsize=10)
        ax2_twin.tick_params(axis="y", labelcolor="#9467bd")

        max_delta = np.nanmax(delta_vals) if len(delta_vals) > 0 else 0.0

        top_lim = 1.0 if max_delta <= 1e-4 else max_delta * 1.15
        ax2_twin.set_ylim(bottom=0, top=top_lim)
        # ---------------------------------------------------------------------

        lines2, labels2 = ax2.get_legend_handles_labels()
        lines2_t, labels2_t = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines2 + lines2_t, labels2 + labels2_t, loc="upper right", fontsize=8.5)
    else:
        ax2.legend(loc="upper right", fontsize=8.5)

    return _finish(fig, save_to)


def plot_q2_indifference_risk(
    df_indiff: pd.DataFrame, save_to: Path | str | None = None
) -> plt.Figure:
    """Question 2: how wide is the members' best response at the posted prices?

    With linear disutility a member's optimal response is a set, not a point.  The bars show the
    widest hourly span of that set (kWh/h) and the line the number of hours in which the member
    can move while staying exactly indifferent.  The manager is optimistic if it anticipates one
    element of the set; nothing in the price guarantees that the member picks it.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)

    df = df_indiff.copy()
    members = df.index.tolist()
    x = np.arange(len(members))

    ranges = df["max_hourly_range_kWh"].astype(float).values
    hours = df["num_indifferent_hours"].astype(int).values

    colors = ["#d62728" if h > 0 else "#2ca02c" for h in hours]
    ax.bar(x, ranges, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6,
           label="Widest hourly range of a best response (kWh/h)")
    ax.set_ylabel("Range of net import (kWh/h)", fontsize=10)
    ax.set_xticks(x)
    labels = [
        f"{m}\n({df['archetype'].iloc[i]})" if "archetype" in df.columns else str(m)
        for i, m in enumerate(members)
    ]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_title("Question 2: Consumer Indifference - the Best Response is a Set, not a Point",
                 fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    ax2 = ax.twinx()
    ax2.plot(x, hours, "o--", color="#1f77b4", linewidth=1.6, markersize=6,
             label="Hours with several optimal loads")
    ax2.set_ylabel("Indifferent hours (of 24)", color="#1f77b4", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(-0.2, 24.2)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8.5)

    return _finish(fig, save_to)