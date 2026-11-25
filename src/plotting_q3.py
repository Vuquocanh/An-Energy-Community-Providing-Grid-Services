"""Question 3 figures (``python main.py --q3-figures``), in the house style of ``plotting.py``.

    plot_q3_models_vs_baseline(summary, oos_dir, save_to)   Fig. 1: realised daily cost per model (box, in-sample target as a
                                                            marker) and realised violation frequency against the target, both
                                                            readings, N = 5 and 10 side by side.
    plot_q3_eps_frontier(summary, save_to)                  Fig. 2: realised violation (x) against out-of-sample mean cost (y),
                                                            one point per eps, joint and hourly; inset targeted vs realised.
    plot_q3_knife_edge(hourly, save_to)                     the posted prices (steps) and, per hour, the range of the community
                                                            import over the members' best-response sets against the cap.

Conventions (W12/W13 notes): one colour per model in every figure, DP in the reference grey; DKK/day and kWh; steps for
hourly quantities; the same out-of-sample days for every model; a title that states the claim (passed by the caller).
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import BASELINE, GRID, INK, MUTED, _finish, _style

MODEL_COLOR = {"DP": MUTED, "SDP_E": "#1baf7a", "SDP_CC": "#2a78d6", "SDP_CVaR": "#9b51e0",
               "SDP_CC_joint": "#2a78d6", "SDP_CC_hourly": "#eb6834"}
MODEL_NAME = {"DP": r"$\mathcal{DP}$ (θ = 1)", "SDP_E": r"$\mathcal{SDP}^{E}$", "SDP_CC": r"$\mathcal{SDP}^{CC}$",
              "SDP_CVaR": r"$\mathcal{SDP}^{CVaR}$", "SDP_CC_joint": "joint", "SDP_CC_hourly": "hourly"}


def _label_name(row) -> str:
    base = MODEL_NAME.get(row["model"], row["model"])
    if row["model"] == "DP":
        return base
    extra = f"\nN = {int(row['N'])}"
    if row["model"] == "SDP_CC" and not pd.isna(row.get("eps", np.nan)):
        base = base.replace("}$", "}_{" + f"{row['eps']:g}" + "}$")
    return base + extra


def plot_q3_models_vs_baseline(summary: pd.DataFrame, oos_dir: Path | str, save_to: Path | str | None = None,
                               title: str = "Pricing for the average member breaks the cap on half the days; "
                                            "the stochastic models trade a few DKK/day for it") -> plt.Figure:
    oos_dir = Path(oos_dir)
    opt = summary[summary["reading"] == "optimistic"].copy()
    opt["order"] = opt["model"].map({"DP": 0, "SDP_E": 1, "SDP_CC": 2, "SDP_CVaR": 3}) + opt["N"].fillna(1) / 100
    opt = opt.sort_values("order").reset_index(drop=True)
    labels = [_label_name(r) for _, r in opt.iterrows()]
    colors = [MODEL_COLOR.get(r["model"], INK) for _, r in opt.iterrows()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    # --- left: realised daily cost, optimistic reading, with the in-sample target
    data = []
    for _, r in opt.iterrows():
        df = pd.read_csv(oos_dir / f"q3_base_oos_{r['label']}_optimistic.csv")
        data.append(df["community_cost"].to_numpy())
    pos = np.arange(len(opt))
    bp = ax1.boxplot(data, positions=pos, widths=0.55, patch_artist=True, showfliers=False,
                     medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    ax1.scatter(pos, opt["cost_mean"], marker="_", s=300, color=colors, lw=2, zorder=3, label="realised mean")
    ax1.scatter(pos, opt["target_cost"], marker="D", s=28, facecolor="white", edgecolor=INK, zorder=4, label="in-sample objective (target)")
    ax1.set_xticks(pos); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("daily community cost, 100 out-of-sample days [DKK/day]")
    ax1.set_title("Cost: targeted vs realised (optimistic reading)", fontsize=10, loc="left")
    lo, hi = min(d.min() for d in data), max(d.max() for d in data)
    ax1.set_ylim(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
    ax1.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _style(ax1)
    # --- right: violation frequency, both readings, against the target
    pess = summary[summary["reading"] == "pessimistic"].set_index("label")
    w = 0.38
    ax2.bar(pos - w / 2, opt["violation_frequency"], w, color=colors, label="optimistic reading")
    ax2.bar(pos + w / 2, [pess.loc[l, "violation_frequency"] for l in opt["label"]], w, color=colors, alpha=0.45, hatch="//",
            edgecolor=colors, label="pessimistic reading")
    for k, (_, r) in enumerate(opt.iterrows()):
        if r["model"] == "SDP_CC" and not pd.isna(r.get("eps", np.nan)):
            ax2.plot([k - 0.45, k + 0.45], [r["eps"], r["eps"]], color=INK, lw=1.2, ls="--")
            ax2.text(k, r["eps"] + 0.02, f"target ε = {r['eps']:g}", ha="center", fontsize=7, color=INK)
    ax2.set_xticks(pos); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(0, 1.08); ax2.set_ylabel("share of days with any cap exceedance")
    ax2.set_title("Protection: realised against target", fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _style(ax2)
    fig.suptitle(textwrap.fill(title, 120), fontsize=11, ha="left", x=0.01)
    return _finish(fig, save_to)


def plot_q3_eps_frontier(summary: pd.DataFrame, save_to: Path | str | None = None,
                         title: str = "A chance constraint buys cap reliability in coarse steps and keeps less than it promises") -> plt.Figure:
    opt = summary[summary["reading"] == "optimistic"].copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    for model, grp in opt[opt["model"].str.startswith("SDP_CC")].groupby("model"):
        grp = grp.sort_values("eps")
        c = MODEL_COLOR.get(model, INK)
        ax.plot(grp["violation_frequency"], grp["cost_mean"], "-o", color=c, ms=5, label=f"$\\mathcal{{SDP}}^{{CC}}$ {MODEL_NAME.get(model, model)}")
        for _, r in grp.iterrows():
            ax.annotate(f"ε = {r['eps']:g}", (r["violation_frequency"], r["cost_mean"]), textcoords="offset points", xytext=(5, 4), fontsize=7, color=c)
    for model in ("DP", "SDP_E"):
        sub = opt[opt["model"] == model]
        if len(sub):
            ax.scatter(sub["violation_frequency"], sub["cost_mean"], marker="s", s=45, color=MODEL_COLOR[model], zorder=4, label=MODEL_NAME[model])
    ax.set_xlabel("realised share of days with cap exceedance (100 out-of-sample days, optimistic reading)")
    ax.set_ylabel("realised mean daily cost [DKK/day]")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    # inset: targeted vs realised
    cc = opt[opt["model"].str.startswith("SDP_CC")]
    if len(cc):
        ins = ax.inset_axes([0.62, 0.55, 0.33, 0.38])
        for model, grp in cc.groupby("model"):
            ins.plot(grp["eps"], grp["violation_frequency"], "o-", ms=3, color=MODEL_COLOR.get(model, INK))
        lim = max(1e-3, cc["eps"].max(), cc["violation_frequency"].max())
        ins.plot([0, lim], [0, lim], color=BASELINE, lw=1)
        ins.set_xlabel("ε targeted", fontsize=7); ins.set_ylabel("realised", fontsize=7); ins.tick_params(labelsize=6)
        _style(ins)
    ax.set_title(textwrap.fill(title, 80), fontsize=10, loc="left")
    return _finish(fig, save_to)


def plot_q3_knife_edge(hourly: pd.DataFrame, save_to: Path | str | None = None,
                       title: str = "Every welfare-optimal price sits on a member's indifference threshold: the cap holds only if "
                                    "members break ties the manager's way") -> plt.Figure:
    """``hourly`` from ``q3_evaluate.evaluate_prices(...)['hourly']`` with import_min / import_max (with_range=True).
    The band is the NET import at the meter over the members' best-response sets (negative = export)."""
    T = len(hourly)
    edges = np.arange(T + 1)
    post = lambda v: np.append(np.asarray(v, dtype=float), np.asarray(v, dtype=float)[-1])   # step="post" needs T+1 values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]}, layout="constrained")
    ax1.step(edges, post(hourly["price"]), where="post", color=MODEL_COLOR["DP"], lw=1.6, label="posted price $x_t$")
    ax1.set_ylabel("DKK/kWh"); ax1.set_ylim(0, hourly["price"].max() * 1.12)
    ax1.legend(frameon=False, fontsize=8, loc="upper left"); _style(ax1)
    ax2.fill_between(edges, post(hourly["import_min"]), post(hourly["import_max"]), step="post", color="#2a78d6", alpha=0.2, lw=0,
                     label="range of the net import over the members' best-response sets")
    ax2.step(edges, post(hourly["import"]), where="post", color="#2a78d6", lw=1.6, label="optimistic reading (what the model assumes)")
    ax2.step(edges, post(hourly["cap"]), where="post", color="#e34948", lw=1.4, ls="--", label="cap")
    ax2.axhline(0, color=BASELINE, lw=0.8)
    lo, hi = float(hourly["import_min"].min()), float(hourly["import_max"].max())
    ax2.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax2.set_ylabel("net import at the meter [kWh/h]  (negative = export)"); ax2.set_xlabel("hour")
    ax2.set_xticks(range(0, 25, 3)); ax2.set_xlim(0, T)
    ax2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1); _style(ax2)
    fig.suptitle(textwrap.fill(title, 95), fontsize=10, ha="left", x=0.01)
    return _finish(fig, save_to)


def run_q3_figures(out: Path | str, community: str = "base", day: str = "d1") -> None:
    """All Q3 figures from the CSVs of --q3-base / --q3-eps, plus the knife-edge figure computed from the DP prices."""
    out = Path(out)
    if (out / "q3_base_summary.csv").exists():
        s = pd.read_csv(out / "q3_base_summary.csv")
        plot_q3_models_vs_baseline(s, out, save_to=out / "q3_fig1_models_vs_baseline.png")
        print("  wrote q3_fig1_models_vs_baseline.png")
    if (out / "q3_eps_summary.csv").exists():
        s = pd.read_csv(out / "q3_eps_summary.csv")
        plot_q3_eps_frontier(s, save_to=out / "q3_fig2_eps_frontier.png")
        print("  wrote q3_fig2_eps_frontier.png")
    prices_file = out / "q3_base_prices.json"
    if prices_file.exists():
        from .data_loader import load_community
        from .q3_evaluate import evaluate_prices
        from .q3_scenarios import expected_scenario
        com = load_community(community, day=day)
        x = np.array(json.load(open(prices_file))["DP"])
        hourly = evaluate_prices(x, expected_scenario(com), None, tie_break="optimistic", with_range=True)["hourly"]
        hourly.to_csv(out / "q3_knife_edge_hourly.csv", index=False)
        plot_q3_knife_edge(hourly, save_to=out / "q3_fig3_knife_edge.png")
        print("  wrote q3_fig3_knife_edge.png")