from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data_loader import DAYS, load_community
from .model import CommunityModel

TOL = 1e-6


def run_q1_checks() -> bool:
    """Check R, I and C on the base community; print PASS/FAIL per check and return True if all pass.

    R against the reference values of dso_contract_params.json (cost, d1-d7), the limit cases
    (joint model = member by member; I with beta = 0 and no cap = R), the exceedance variables,
    the balance at the community meter, C <= I, and the 18 hours at the cap on d1 (assignment text).
    """
    ok = True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= bool(passed)
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    def solve(com, setup):
        return CommunityModel(com, setup=setup).build().solve()

    for day in DAYS:
        com = load_community("base", day=day)
        r, ref = solve(com, "R"), com.contract.reference_model_R
        check(f"R cost {day} = file", abs(r.objective - ref["cost_DKK"]) < 0.01,
              f"{r.objective:.2f} vs {ref['cost_DKK']} DKK (import {r.hourly['import'].sum():.1f} vs "
              f"{ref['daily_import_kWh']} kWh may differ: R has alternative optima)")

    com = load_community("base", day="d1")
    c = com.contract
    R, I, C = (solve(com, s) for s in ("R", "I", "C"))
    for s, res in (("R", R), ("I", I)):
        alone = sum(solve(replace(com, members=[d]), s).objective for d in com.members)
        check(f"{s} joint model = member by member", abs(res.objective - alone) < TOL,
              f"{res.objective:.4f} vs {alone:.4f}")
    loose = replace(c, tariff_discount_ratio=0.0,
                    individual_capacity_limit_kW={i: v * 1e6 for i, v in c.individual_capacity_limit_kW.items()})
    I0 = solve(replace(com, contract=loose), "I")
    check("I with beta = 0 and no cap = R", abs(I0.objective - R.objective) < TOL,
          f"{I0.objective:.4f} vs {R.objective:.4f}")
    cap_i = np.concatenate([c.individual_capacity_limit_kW[i] for i in com.member_ids])
    gap = np.abs(I.members["exceedance"] - np.maximum(I.members["import"] - cap_i, 0)).max()
    check("I exceedance = max(0, import - own cap)", gap < TOL, f"max gap {gap:.1e}")

    h = C.hourly
    gap = np.abs(h["exceedance_billed"] - h["exceedance"]).max()
    check("C exceedance = max(0, import - cap)", gap < TOL, f"max gap {gap:.1e}")
    net = C.members["net"].groupby(level="hour").sum()
    gap = np.abs(h["import"] - h["export"] - net).max()
    check("C balance at the meter", gap < TOL, f"max residual {gap:.1e}")
    check("C <= I (same contract)", C.objective <= I.objective + TOL, f"{C.objective:.2f} vs {I.objective:.2f}")
    at_cap = int(np.isclose(h["import"], h["cap"]).sum())
    check("C at the cap in 18 hours of d1 (assignment)", at_cap == 18, f"{at_cap} hours")
    return ok


# ----------------------------------------------------------------------------------------------
# Question 2 validation
# ----------------------------------------------------------------------------------------------
"""Three checks that a Q2 pricing solution must pass:

1. the C dispatch is recovered (objective; the dispatch itself may be non-unique);
2. the posted prices are really a best response for every member and the dual-derived payments
   equal the actual payments (``DynamicPricingModel.validate_solution``, stored in ``meta``);
3. the manager's optimistic / pessimistic exposure when the members' best response is not unique
   (indifference), which is the ``short discussion about indifference`` of Question 2.
"""

from typing import Any

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from .data_loader import CommunityData
from .model import Results


def validate_c_dispatch_recovery(results_pricing: Results, results_MC: Results,
                                 tol: float = 1e-2) -> dict[str, Any]:
    """Compare a pricing run with model C.

    The *objective* must match (the price can always support the C optimum in the optimistic
    sense).  The hourly import profile and the distance to the C duals are reported as
    information only: the C optimum -- and the price vector that supports it -- need not be
    unique, so requiring ``x == lambda`` would be wrong.
    """
    gap_obj = float(results_pricing.objective - results_MC.objective)
    gap_imp = float(np.max(np.abs(results_pricing.hourly["import"].to_numpy()
                                  - results_MC.hourly["import"].to_numpy())))
    info = {}
    if "dual_balance" in results_MC.hourly.columns:
        diff = np.abs(results_pricing.hourly["posted_price"].to_numpy()
                      - results_MC.hourly["dual_balance"].to_numpy())
        info["max_price_minus_lambda"] = float(diff.max())
        info["median_price_minus_lambda"] = float(np.median(diff))
    passed = abs(gap_obj) <= tol
    print(f"[{'PASS' if passed else 'FAIL'}] C dispatch recovered: objective gap = {gap_obj:+.6f} DKK"
          f" | import profile gap = {gap_imp:.4f} kWh"
          + (f" | max |x - lambda_C| = {info['max_price_minus_lambda']:.4f} DKK/kWh (informative)"
             if info else ""))

    return {"passed": passed, "objective_gap_DKK": gap_obj,
            "import_profile_gap_kWh": gap_imp, **info}


def summarize_model_validation(results_pricing: Results, tol: float = 1e-4) -> dict[str, Any]:
    """Report the independent checks that ``DynamicPricingModel.validate_solution`` stored."""
    checks = results_pricing.meta.get("validation")
    if not checks:
        raise KeyError("No 'validation' entry in meta: was the model solved by DynamicPricingModel?")
    worst = max(checks.values())
    passed = worst <= tol
    print(f"[{'PASS' if passed else 'FAIL'}] KKT / best response / payment identity: "
          f"worst residual = {worst:.2e}")
    return {"passed": passed, "max_residual": float(worst), "checks": checks}


def analyze_indifference_risk(com_data: CommunityData, posted_prices: np.ndarray,
                              end_soc_mode: str = "constraint",
                              tol: float = 1e-5) -> pd.DataFrame:
    """How wide is every member's best response at the posted prices?

    With linear disutility a member's best response is a *set*, not a point.  For each member we
    solve its LP, then -- keeping its cost within ``tol`` of the optimum -- minimise and maximise
    its net import **in every hour**.  ``max_hourly_range_kWh`` is the widest hourly span of that
    set and ``num_indifferent_hours`` the number of hours in which the member can move while
    remaining exactly indifferent.  The manager is optimistic if it anticipates one element of the
    set; the pessimistic alternative is any other element.
    """
    from .model import CommunityModel

    records = []

    for d in com_data.members:
        i = d.consumer_id
        follower = CommunityModel(com_data, "C", end_soc_mode=end_soc_mode)
        follower._add_member(d)
        T = list(follower.T)
        net = follower.net[i]

        cost = gp.quicksum(v[i] for v in follower.mexpr.values()) \
            + gp.quicksum(float(posted_prices[t]) * net[t] for t in T)
        follower.m.setObjective(cost, GRB.MINIMIZE)
        follower.m.Params.OutputFlag = 0
        follower.m.optimize()
        if follower.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Member LP failed for {i}")
        min_cost = follower.m.ObjVal
        follower.m.addConstr(cost <= min_cost + tol, name="opt_cost_bound")

        lo, hi = [], []
        for t in T:
            follower.m.setObjective(net[t], GRB.MINIMIZE)
            follower.m.optimize()
            lo.append(net[t].getValue())
            follower.m.setObjective(net[t], GRB.MAXIMIZE)
            follower.m.optimize()
            hi.append(net[t].getValue())
        lo, hi = np.array(lo), np.array(hi)
        spread = hi - lo

        records.append(
            {
                "member": i,
                "archetype": d.archetype,
                "disutility_slope": d.linear_disutility,
                "min_cost_DKK": round(min_cost, 4),
                "max_hourly_range_kWh": round(float(np.abs(spread).max()), 4),
                "num_indifferent_hours": int((np.abs(spread) > 1e-3).sum()),
                "has_indifference": bool(np.abs(spread).max() > 1e-3),
            }
        )
        follower.m.dispose()

    df = pd.DataFrame(records).set_index("member")
    n = int(df["has_indifference"].sum())
    worst = df["max_hourly_range_kWh"].max()
    print(f"[INFO] Indifference: {n}/{len(df)} members have multiple optimal responses; "
          f"the largest hourly range of a best response is {worst:.2f} kWh/h. "
          f"The manager is optimistic if it anticipates one element of that set.")
    return df


def run_q2_validation_pipeline(com_data: CommunityData, results_cap: Results,
                               results_MC: Results, model_pricing: Any) -> dict[str, Any]:
    """Full Q2 validation: C recovery, independent KKT checks, indifference analysis."""
    print("\n" + "=" * 60)
    print(" Running Question 2 validation pipeline")
    print("=" * 60)

    res_duals = validate_c_dispatch_recovery(results_cap, results_MC)
    res_kkt = summarize_model_validation(results_cap)

    posted_x = results_cap.hourly["posted_price"].to_numpy()
    df_indiff = analyze_indifference_risk(com_data, posted_x)

    return {
        "c_recovery": res_duals,
        "kkt_residuals": res_kkt,
        "indifference_analysis": df_indiff,
    }
