"""Question 3 experiments: pricing for uncertain members. One command per experiment (README rule).

    python main.py --q3-base              # DP baseline vs SDP^E, SDP^CC (eps), SDP^CVaR at N = 5 and 10; out-of-sample on 100 days
    python main.py --q3-eps               # the eps frontier of SDP^CC at N = 5 (joint and hourly), same 100 days
    python main.py --q3-base --q3-n 5     # only N = 5 (quick)

Every price vector is evaluated on the SAME out-of-sample scenarios (common random numbers, seed 47750) with the
protocol of ``q3_evaluate`` in two readings of the members' responses: optimistic (what the single-level models
assume) and pessimistic (the attainable adversarial element). Results in results/<community>/q3/.

Outputs of --q3-base
    q3_base_models.csv        one row per (model, N): in-sample objective, violations, sizes, solve time, gap, price deviation
    q3_base_summary.csv       one row per (model, N, reading): the F2 metrics on the 100 out-of-sample days
    q3_base_prices.json       the price vectors
    q3_base_oos_<model>_N<N>_<reading>.csv   one row per out-of-sample day
Outputs of --q3-eps
    q3_eps_models.csv, q3_eps_summary.csv, q3_eps_prices.json, q3_eps_oos_<...>.csv   (same layout, keyed by eps and joint/hourly)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import CommunityData, load_community
from .model import CommunityModel
from .model_q2_pricing import DynamicPricingModel, NoPricingSolution
from .model_q3_sdp import SDPModel
from .q3_evaluate import evaluate_on_set, summarize
from .q3_scenarios import Scenario, scenario_sets, validate_scenarios

READINGS = ("optimistic", "pessimistic")
SUMMARY_KEYS = ("cost_mean", "cost_p10", "cost_p50", "cost_p90", "cost_max", "violation_frequency", "exceedance_mean_kWh",
                "exceedance_max_kWh", "penalty_mean", "penalty_max", "budget_mean", "budget_p10", "budget_p90",
                "share_days_deficit", "ir_violation_share", "import_range_mean_kWh", "hours_indifferent_mean")


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------
def _benchmarks_at_theta1(com: CommunityData) -> dict[str, float]:
    R = CommunityModel(com, setup="R").build().solve().member_costs["total"].to_dict()
    I = CommunityModel(com, setup="I").build().solve().member_costs["total"].to_dict()
    return {i: min(R[i], I[i]) for i in com.member_ids}


def solve_baseline_dp(com: CommunityData, mode: str = "BB", mip_gap: float = 1e-3, time_limit: float = 600) -> dict:
    """The deterministic baseline: DP at theta = 1 with the shipped bounds and the minmax price rule."""
    t0 = time.perf_counter()
    m = DynamicPricingModel(com, mode=mode, benchmark_costs=_benchmarks_at_theta1(com),
                            price_bounds=DynamicPricingModel.default_price_bounds(com), mip_gap=mip_gap, time_limit=time_limit).build()
    r = m.solve()
    row = {"model": "DP", "N": 1, "status": r.status, "objective": float(r.objective), "expected_penalty": r.meta["objective_terms"]["penalty_cost"],
           "in_sample_violations": int(r.hourly["exceedance"].max() > 1e-6), "n_binaries": m.m.NumBinVars, "n_constraints": m.m.NumConstrs,
           "solve_s": time.perf_counter() - t0, "gap": r.meta["solver_gap"], "max_price_deviation": r.meta["max_price_deviation"],
           "eps": np.nan, "cc_joint": np.nan, "cvar_beta": np.nan, "cvar_gamma": np.nan}
    prices = r.hourly["posted_price"].to_numpy()
    m.m.dispose()
    return {"row": row, "prices": prices}


def solve_sdp(com: CommunityData, scenarios: list[Scenario], label: str, warm_prices=None, mode: str = "BB",
              cc_eps: float | None = None, cc_joint: bool = True, cvar_beta: float | None = None, cvar_gamma: float = 0.5,
              mip_gap: float = 1e-3, time_limit: float = 1200, mip_focus: int = 0, verbose: bool = False) -> dict:
    """One SDP variant on one in-sample set; infeasible is a result (status recorded, no prices)."""
    t0 = time.perf_counter()
    m = SDPModel(com, scenarios, mode=mode, cc_eps=cc_eps, cc_joint=cc_joint, cvar_beta=cvar_beta, cvar_gamma=cvar_gamma,
                 price_bounds=DynamicPricingModel.default_price_bounds(com), mip_gap=mip_gap, time_limit=time_limit,
                 mip_focus=mip_focus, verbose=verbose).build()
    if warm_prices is not None:
        m.warm_start_from_prices(warm_prices)
    row = {"model": label, "N": len(scenarios), "eps": cc_eps if cc_eps is not None else np.nan, "cc_joint": cc_joint if cc_eps is not None else np.nan,
           "cvar_beta": cvar_beta if cvar_beta is not None else np.nan, "cvar_gamma": cvar_gamma if cvar_beta is not None else np.nan,
           "n_binaries": m.m.NumBinVars, "n_constraints": m.m.NumConstrs}
    try:
        r = m.solve()
    except NoPricingSolution as exc:
        row.update({"status": "infeasible" if exc.status_code == 3 else f"no solution (status {exc.status_code})", "objective": np.nan,
                    "expected_penalty": np.nan, "in_sample_violations": np.nan, "solve_s": time.perf_counter() - t0, "gap": np.nan, "max_price_deviation": np.nan})
        print(f"  {label} N={len(scenarios)}: {row['status']} after {row['solve_s']:.0f} s")
        m.m.dispose()
        return {"row": row, "prices": None, "scenarios": None, "members": None}
    row.update({"status": r.status, "objective": float(r.objective), "expected_penalty": r.meta["expected_penalty"],
                "in_sample_violations": r.meta["in_sample_violations"], "solve_s": time.perf_counter() - t0, "gap": r.meta["solver_gap"],
                "max_price_deviation": r.meta["max_price_deviation"], "cvar": r.meta.get("cvar"),
                "total_ir_slack": r.meta.get("total_ir_slack", 0.0)})
    print(f"  {label} N={len(scenarios)}: objective {r.objective:.3f}, violations {r.meta['in_sample_violations']}/{len(scenarios)}, "
          f"gap {r.meta['solver_gap']:.1e}, {row['solve_s']:.0f} s")
    m.m.dispose()
    return {"row": row, "prices": r.prices, "scenarios": r.scenarios, "members": r.members}


def evaluate_all(prices: dict[str, np.ndarray], oos: list[Scenario], targets: dict[str, float], out: Path, prefix: str,
                 keys: dict[str, dict]) -> pd.DataFrame:
    """Every price vector on the common out-of-sample set, both readings; one summary row per (label, reading)."""
    rows = []
    base_cost = {}
    for label, x in prices.items():
        if x is None:
            continue
        for tb in READINGS:
            df, dm = evaluate_on_set(x, oos, label=f"{label}_{tb}", tie_break=tb, with_range=(tb == "optimistic"), verbose=False)
            df.to_csv(out / f"{prefix}_oos_{label}_{tb}.csv", index=False)
            s = summarize(df, dm, target_cost=targets.get(label))
            row = {"label": label, "reading": tb, **keys.get(label, {}), "target_cost": targets.get(label, np.nan),
                   "optimism_gap": s.get("optimism_gap", np.nan)}
            row.update({k: s.get(k, np.nan) for k in SUMMARY_KEYS})
            rows.append(row)
            if label.startswith("DP"):
                base_cost[tb] = s["cost_mean"]
    summary = pd.DataFrame(rows)
    if base_cost:
        summary["VSS_vs_DP"] = [base_cost.get(r, np.nan) - c for r, c in zip(summary["reading"], summary["cost_mean"])]
    return summary


# ----------------------------------------------------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------------------------------------------------
def run_q3_base(out: Path | str, community: str = "base", day: str = "d1", n_values: tuple[int, ...] = (5, 10),
                eps: float = 0.1, cvar_beta: float = 0.9, cvar_gamma: float = 0.5, mip_gap: float = 1e-3,
                time_limit: float = 1200, n_oos: int = 100) -> dict:
    """DP baseline vs SDP^E, SDP^CC_eps (joint) and SDP^CVaR at every N, all evaluated on the common out-of-sample set."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    com = load_community(community, day=day)
    sets = scenario_sets(com, n_in=tuple(n_values), n_oos=n_oos)
    for scen in sets.values():
        validate_scenarios(com, scen)
    from .q3_scenarios import theta_table
    for key, scen in sets.items():                      # the drawn sets, with their seeds, as the deliverable asks
        theta_table(scen).to_csv(out / f"q3_scenarios_{key}.csv", index=False)
    json.dump({key: {"n": len(scen), "seed": scen[0].seed} for key, scen in sets.items()},
              open(out / "q3_scenarios_seeds.json", "w"), indent=1)
    print(f"Q3 base study: {community} {day}, N in {list(n_values)}, {n_oos} out-of-sample days (seed {sets['oos'][0].seed})")

    print("[baseline] DP at theta = 1")
    base = solve_baseline_dp(com, mip_gap=1e-4)
    rows, prices, targets, keys = [base["row"]], {"DP": base["prices"]}, {"DP": base["row"]["objective"]}, {"DP": {"model": "DP", "N": 1}}
    variants = [("SDP_CVaR", {"cvar_beta": cvar_beta, "cvar_gamma": cvar_gamma}), ("SDP_E", {}),
                ("SDP_CC", {"cc_eps": eps, "cc_joint": True, "mip_focus": 1}),
                ("SDP_E_softIR", {"mode": "soft_ir"})]      # IR in expectation with one slack per member (Formative 2, E5b-c)
    warm = {"default": base["prices"]}          # carried across N: N=10 starts from the N=5 CVaR prices
    for N in n_values:
        scen = sets[f"in_{N}"]
        for name, kw in variants:
            label = f"{name}_N{N}"
            # start = warm.get("SDP_CVaR", warm["default"]) if name == "SDP_CC" else warm["default"]
            start = warm.get("SDP_CVaR", warm["default"])
            res = solve_sdp(com, scen, name, warm_prices=start, mip_gap=mip_gap, time_limit=time_limit, **kw)
            if res["prices"] is not None:
                warm[name] = res["prices"]
            rows.append(res["row"]); prices[label] = res["prices"]; targets[label] = res["row"].get("objective", np.nan)
            keys[label] = {"model": name, "N": N, **{k: res["row"][k] for k in ("eps", "cc_joint", "cvar_beta", "cvar_gamma")}}
            if res["scenarios"] is not None:
                res["scenarios"].to_csv(out / f"q3_base_insample_{label}.csv", index=False)
            if res.get("members") is not None:
                res["members"].to_csv(out / f"q3_base_members_{label}.csv", index=False)
    models = pd.DataFrame(rows)
    models.to_csv(out / "q3_base_models.csv", index=False)
    json.dump({k: (v.tolist() if v is not None else None) for k, v in prices.items()}, open(out / "q3_base_prices.json", "w"), indent=1)

    print("[evaluation] every price vector on the common out-of-sample set, both readings")
    summary = evaluate_all(prices, sets["oos"], targets, out, "q3_base", keys)
    summary.to_csv(out / "q3_base_summary.csv", index=False)
    cols = ["label", "reading", "target_cost", "cost_mean", "cost_p90", "violation_frequency", "exceedance_mean_kWh", "budget_mean", "share_days_deficit", "VSS_vs_DP"]
    print(summary[cols].round(3).to_string(index=False))
    return {"models": models, "summary": summary, "prices": prices}


def run_q3_eps(out: Path | str, community: str = "base", day: str = "d1", N: int = 5,
               eps_values: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6), both_variants: bool = True,
               mip_gap: float = 1e-3, time_limit: float = 1200, n_oos: int = 100) -> dict:
    """The eps frontier of SDP^CC: in-sample cost and realised violation against the target, joint and hourly."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    com = load_community(community, day=day)
    sets = scenario_sets(com, n_in=(N,), n_oos=n_oos)
    scen = sets[f"in_{N}"]
    print(f"Q3 eps study: N = {N}, eps in {list(eps_values)}, joint{' and hourly' if both_variants else ''}")
    base = solve_baseline_dp(com, mip_gap=1e-4)
    rows, prices, targets, keys = [base["row"]], {"DP": base["prices"]}, {"DP": base["row"]["objective"]}, {"DP": {"model": "DP", "N": 1}}
    # warm start: the SDP^CVaR prices of the base study at this N when available (CC finds no incumbent from DP prices), else DP
    warm0 = base["prices"]
    base_prices = out / "q3_base_prices.json"
    if base_prices.exists():
        saved = json.load(open(base_prices)).get(f"SDP_CVaR_N{N}")
        if saved is not None:
            warm0 = np.asarray(saved)
            print(f"  warm start from q3_base_prices.json: SDP_CVaR_N{N}")
    for joint in ((True, False) if both_variants else (True,)):
        start = warm0
        for e in eps_values:
            tag = "joint" if joint else "hourly"
            label = f"SDP_CC_{tag}_eps{e:g}_N{N}"
            res = solve_sdp(com, scen, f"SDP_CC_{tag}", warm_prices=start, cc_eps=e, cc_joint=joint, mip_gap=mip_gap,
                            time_limit=time_limit, mip_focus=1)
            if res["prices"] is not None:
                start = res["prices"]
            rows.append(res["row"]); prices[label] = res["prices"]; targets[label] = res["row"].get("objective", np.nan)
            keys[label] = {"model": f"SDP_CC_{tag}", "N": N, "eps": e, "cc_joint": joint}
            if res["scenarios"] is not None:
                res["scenarios"].to_csv(out / f"q3_eps_insample_{label}.csv", index=False)
    models = pd.DataFrame(rows)
    models.to_csv(out / "q3_eps_models.csv", index=False)
    json.dump({k: (v.tolist() if v is not None else None) for k, v in prices.items()}, open(out / "q3_eps_prices.json", "w"), indent=1)
    summary = evaluate_all(prices, sets["oos"], targets, out, "q3_eps", keys)
    summary.to_csv(out / "q3_eps_summary.csv", index=False)
    cols = ["label", "reading", "target_cost", "cost_mean", "violation_frequency", "exceedance_mean_kWh", "VSS_vs_DP"]
    print(summary[cols].round(3).to_string(index=False))
    return {"models": models, "summary": summary, "prices": prices}