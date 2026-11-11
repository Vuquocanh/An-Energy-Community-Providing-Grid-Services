"""plot_import_vs_cap(res, save_to): import at the connection point of R / I / C against the cap.
plot_shift_heatmap(changes, save_to): change of import against R by source and hour, one block per setup.
plot_day_data(com, save_to): input data of one day - prices, reference load, PV available and cap.
plot_internal_price(res, com, save_to): internal price lambda of C against the prices at the meter.
plot_member_gains(table, save_to): gain of every member from joining C, against R and against I.
plot_sensitivity(sweeps, base, save_to): the Q1 metrics against each swept contract term, one column per term.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from .data_loader import CommunityData
from .model import Results

INK, MUTED, GRID, BASELINE, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# a setup keeps its colour in every figure (R = reference grey); the load classes L1-L4 are one blue ramp,
# light = most flexible; diverging blue - grey - red for changes (palette checked for colour-vision deficiency)
SETUP_COLOR = {"R": MUTED, "I": "#eb6834", "C": "#1baf7a"}
SOURCE_COLOR = {"L1": "#86b6ef", "L2": "#3987e5", "L3": "#1c5cab", "L4": "#0d366b",
                "pv": "#eda100", "battery": "#6250d6", "export / netting": BASELINE}
SOURCE_LABEL = {"L1": "load L1 (c_L 0.25)", "L2": "load L2 (0.5)", "L3": "load L3 (1.0)", "L4": "load L4 (2.0)",
                "pv": "PV", "battery": "battery (charge - discharge)", "export / netting": "export / netting"}
HEAT_ROWS = ("L1", "L2", "L3", "L4", "battery", "pv")
SWEEP_LABEL = {"beta": "tariff discount β", "cap_factor": "cap energy factor f (cap = f E_R / 24)"}
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#2a78d6", "#f0efec", "#e34948"])


def _finish(fig: plt.Figure, save_to: Path | str | None) -> plt.Figure:
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def _style(ax: plt.Axes) -> None:
    """Recessive chrome: horizontal hairline grid, no top/right spines, muted ticks."""
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plot_import_vs_cap(res: dict[str, Results], save_to: Path | str | None = None) -> plt.Figure:
    """Import at the connection point (kW) of every setup in ``res`` against the cap, the exceedance shaded."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    h = res["R"].hourly.index.to_numpy()
    cap = res["R"].hourly["cap"].to_numpy()
    for s, r in res.items():
        imp = r.hourly["import"].to_numpy()
        ax.step(h, imp, where="mid", color=SETUP_COLOR[s], lw=2, label=s)
        ax.fill_between(h, cap, imp, where=imp > cap + 1e-6, step="mid", color=SETUP_COLOR[s], alpha=0.25, lw=0)
    ax.step(h, cap, where="mid", color=INK, lw=1.2, ls="--", label="cap")
    ax.set(xlabel="hour", ylabel="import (kW)", xticks=range(0, 24, 2))
    ax.legend(fontsize=9, loc="upper left")
    _style(ax)
    return _finish(fig, save_to)


def plot_shift_heatmap(changes: dict[str, pd.DataFrame], save_to: Path | str | None = None) -> plt.Figure:
    """Change of import against R by source (rows L1-L4, battery, PV) and hour, kWh/h.

    ``changes`` = {setup: q1.import_change_by_source(X, R)}, one block per setup, one colour scale for all:
    red = more import than R (load moved in, charging, less PV), blue = less (load moved out, discharging).
    """
    rows = [r for r in HEAT_ROWS if all(r in c for c in changes.values())]
    vmax = max(np.abs(c[rows].to_numpy()).max() for c in changes.values())
    fig, axes = plt.subplots(len(changes), 1, figsize=(10, 1 + 0.4 * len(rows) * len(changes)),
                             sharex=True, squeeze=False, layout="constrained")
    for ax, (s, c) in zip(axes[:, 0], changes.items()):
        im = ax.imshow(c[rows].T.to_numpy(), aspect="auto", cmap=DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(len(rows)), [SOURCE_LABEL[r] for r in rows])
        ax.set_ylabel(f"{s} minus R")
        # surface-coloured gaps between the cells
        ax.set_xticks(np.arange(-0.5, len(c), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=1.5)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(colors=MUTED, labelcolor=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1, 0].set(xlabel="hour", xticks=range(0, 24, 2))
    fig.colorbar(im, ax=axes[:, 0].tolist(), label="change of import against R (kWh/h)", shrink=0.9)
    return _finish(fig, save_to)


def plot_day_data(com: CommunityData, save_to: Path | str | None = None) -> plt.Figure:
    """Input data of one day at the community level.

    (a) day-ahead price p_t and the import price without contract (p + tau_imp) and under the contract below
        the cap (p + (1 - beta) tau_imp), DKK/kWh; (b) the members' reference load and the PV available,
        summed over the community, against the cap, kW.
    """
    h = com.members[0].hours
    p, tau, beta = com.energy_price, com.import_tariff, com.contract.tariff_discount_ratio
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.step(h, p, where="mid", color=INK, lw=2, label="day-ahead price p")
    ax1.step(h, p + tau, where="mid", color=SETUP_COLOR["R"], lw=2, label="import price, no contract")
    ax1.step(h, p + (1 - beta) * tau, where="mid", color=SETUP_COLOR["C"], lw=2,
             label="import price below the cap, contract")
    ax1.set(ylabel="price (DKK/kWh)")
    ax1.legend(fontsize=8, loc="upper left")
    _style(ax1)

    ax2.step(h, sum(m.reference_load for m in com.members), where="mid", color=SOURCE_COLOR["L2"], lw=2,
             label="reference load, all members")
    ax2.step(h, sum(m.pv_available for m in com.members), where="mid", color=SOURCE_COLOR["pv"], lw=2,
             label="PV available, all systems")
    ax2.step(h, com.contract.capacity_limit_kW, where="mid", color=INK, lw=1.2, ls="--", label="cap")
    ax2.set(xlabel="hour", ylabel="power (kW)", xticks=range(0, 24, 2))
    ax2.legend(fontsize=8, loc="upper left")
    _style(ax2)
    return _finish(fig, save_to)


def plot_internal_price(res: Results, com: CommunityData, save_to: Path | str | None = None) -> plt.Figure:
    """Internal price lambda_t of C (``res``: the dual of the community balance) against the day-ahead price and
    the prices at the meter, import below the cap p + (1 - beta) tau_imp and export p - (1 - beta) tau_exp
    (DKK/kWh); the hours with the import at the cap are shaded."""
    h = res.hourly
    hours = h.index.to_numpy()
    f = 1 - com.contract.tariff_discount_ratio
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for t in hours[(h["import"] >= h["cap"] - 1e-6).to_numpy()]:
        ax.axvspan(t - 0.5, t + 0.5, color=GRID, alpha=0.6, lw=0)
    ax.step(hours, h["price"], where="mid", color=INK, lw=1.5, label="day-ahead price p")
    ax.step(hours, h["price"] + f * com.import_tariff, where="mid", color=MUTED, lw=1.2, ls="--",
            label="import price below the cap")
    ax.step(hours, h["price"] - f * com.export_tariff, where="mid", color=MUTED, lw=1.2, ls=":",
            label="export price")
    ax.step(hours, h["dual_balance"], where="mid", color=SETUP_COLOR["C"], lw=2, label="internal price λ (C)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [Patch(color=GRID, alpha=0.6)], labels + ["import at the cap"], fontsize=8, loc="upper left")
    ax.set(xlabel="hour", ylabel="price (DKK/kWh)", xticks=range(0, 24, 2))
    _style(ax)
    return _finish(fig, save_to)


def plot_member_gains(table: pd.DataFrame, save_to: Path | str | None = None) -> plt.Figure:
    """Gain of every member from joining C under the dual settlement, against R and against I (DKK/day; ``table`` =
    q1.settlement), the members in the order of the table (slope class, then assets); > 0: better off in C."""
    x = np.arange(len(table))
    w = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, table["gain_vs_R_DKK"], w, color=SETUP_COLOR["R"], edgecolor=SURFACE, lw=0.8, label="against R")
    ax.bar(x + w / 2, table["gain_vs_I_DKK"], w, color=SETUP_COLOR["I"], edgecolor=SURFACE, lw=0.8, label="against I")
    ax.axhline(0, color=BASELINE, lw=1)
    slope = table["archetype"].str[:2].to_numpy()
    for k in np.flatnonzero(slope[1:] != slope[:-1]):        # separators between the slope classes
        ax.axvline(k + 0.5, color=GRID, lw=1)
    ax.set_xticks(x, table.index, rotation=45, ha="right")
    ax.set(ylabel="gain from joining C (DKK/day)")
    ax.legend(fontsize=9, loc="upper left")
    _style(ax)
    return _finish(fig, save_to)


def plot_sensitivity(sweeps: dict[str, pd.DataFrame], base: dict[str, float],
                     rows: tuple[str, ...] = ("value_DKK", "max_exceedance_kW", "hours_above_cap"),
                     save_to: Path | str | None = None) -> plt.Figure:
    """The Q1 metrics ``rows`` against each swept contract term, one column per term (``sweeps`` = {term:
    q1.sweep(...)}): value of I and C against R (DKK/day, community), max exceedance (kW) and hours above the cap
    of R, I and C (DSO); the base value of the term (``base``) dashed."""
    spec = {"value_DKK": ("value against R (DKK/day)", ("I", "C")),
            "max_exceedance_kW": ("max exceedance (kW)", ("R", "I", "C")),
            "hours_above_cap": ("hours above the cap", ("R", "I", "C"))}
    fig, axes = plt.subplots(len(rows), len(sweeps), figsize=(max(8, 5.5 * len(sweeps)), 2.8 * len(rows)),
                             sharex="col", squeeze=False)
    for j, (term, t) in enumerate(sweeps.items()):
        for i, col in enumerate(rows):
            ylabel, setups = spec[col]
            ax = axes[i, j]
            for s in setups:
                d = t[t["setup"] == s]
                ax.plot(d[term], d[col], "o-", color=SETUP_COLOR[s], lw=2, ms=6, label=s)
            ax.axvline(base[term], color=INK, lw=1, ls="--", label="base case")
            if col == "value_DKK":
                ax.axhline(0, color=BASELINE, lw=1)
            if j == 0:
                ax.set_ylabel(ylabel)
            _style(ax)
        axes[-1, j].set_xlabel(SWEEP_LABEL[term])
    axes[min(1, len(rows) - 1), 0].legend(fontsize=8, loc="best")     # a panel with R, I and C
    return _finish(fig, save_to)
