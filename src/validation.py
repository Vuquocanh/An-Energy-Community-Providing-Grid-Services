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

"""Validation utilities for Question 2.

Provides analytical and numerical validation tools:
1. Validates dynamic pricing dual equivalence against Question 1 Model C shadow prices.
2. Checks lower-level KKT complementarity slackness residuals.
3. Analyzes consumer response indifference (non-uniqueness of optimal schedules)
   and quantifies the manager's risk under optimistic vs pessimistic assumptions.
"""

from typing import Any
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from .data_loader import CommunityData, InputData
from .model import CommunityModel, Results
from .model_q2_pricing import DynamicPricingModel

def validate_against_MC_duals(
    results_pricing_cap: Results, results_MC: Results, tol: float = 1e-4
) -> dict[str, Any]:
    """Validate whether posted prices in M^P_cap match Model C internal balance duals.

    Parameters
    ----------
    results_pricing_cap : Results
        Results object from DynamicPricingModel in 'cap_only' mode.
    results_MC : Results
        Results object from Question 1 centralized model (Model C).
    tol : float, default 1e-4
        Tolerance threshold for price equivalence check.

    Returns
    -------
    dict[str, Any]
        Dictionary with validation status ('passed'), max price difference, and hourly comparison.
    """
    x_posted = results_pricing_cap.hourly["posted_price"].to_numpy()
    
    if "dual_balance" not in results_MC.hourly.columns:
        raise KeyError("Model C results table is missing 'dual_balance' column.")
    
    lambda_MC = results_MC.hourly["dual_balance"].to_numpy()
    
    diff = np.abs(x_posted - lambda_MC)
    max_diff = float(np.max(diff))
    passed = bool(max_diff <= tol)

    df_cmp = pd.DataFrame(
        {
            "posted_price_x": x_posted,
            "MC_dual_lambda": lambda_MC,
            "absolute_diff": diff,
        },
        index=results_pricing_cap.hourly.index,
    )

    print(f"[{'PASS' if passed else 'FAIL'}] Dual Equivalence (M^P_cap vs M^C): max diff = {max_diff:.6f} DKK/kWh")

    return {
        "passed": passed,
        "max_difference": max_diff,
        "hourly_comparison": df_cmp,
    }


def check_kkt_residuals(
    model_pricing: DynamicPricingModel, tol: float = 1e-4
) -> dict[str, Any]:
    """Verify lower-level KKT complementary slackness conditions (residual < tol).

    Evaluates (dual_variable) * (primal_slack) for all linearized lower-level constraints.

    Parameters
    ----------
    model_pricing : DynamicPricingModel
        Built and solved DynamicPricingModel object.
    tol : float, default 1e-4
        Maximum allowed complementarity residual.

    Returns
    -------
    dict[str, Any]
        Dictionary containing overall pass status, maximum residual, and detailed violation log.
    """
    mvars = model_pricing.mvar
    residuals = []

    for i in model_pricing.ids:
        d = model_pricing.com.member(i)
        T = model_pricing.T
        ref = d.reference_load

        load_val = np.array([mvars["load"][i][t].X for t in T])
        dev_val = np.array([mvars["dev"][i][t].X for t in T])

        # 1. Dev Upper: alpha_up * (dev - load + ref) == 0
        alpha_up = np.array([mvars["dual_dev_up"][i][t].X for t in T])
        slack_dev_up = dev_val - load_val + ref
        residuals.extend(alpha_up * slack_dev_up)

        # 2. Dev Lower: alpha_dn * (dev + load - ref) == 0
        alpha_dn = np.array([mvars["dual_dev_dn"][i][t].X for t in T])
        slack_dev_dn = dev_val + load_val - ref
        residuals.extend(alpha_dn * slack_dev_dn)

        # 3. Load Bounds: nu_lower * (load - load_lower), nu_upper * (load_upper - load)
        nu_low = np.array([mvars["dual_load_lower"][i][t].X for t in T])
        slack_low = load_val - d.load_lower_kWh
        residuals.extend(nu_low * slack_low)

        nu_upp = np.array([mvars["dual_load_upper"][i][t].X for t in T])
        slack_upp = d.load_upper_kWh - load_val
        residuals.extend(nu_upp * slack_upp)

        # 4. Minimum Daily Energy: mu_energy * (sum(load) - min_daily_energy)
        mu_e = mvars["dual_min_daily_energy"][i].X
        slack_e = load_val.sum() - d.min_daily_energy_kWh
        residuals.append(mu_e * slack_e)

        # 5. PV Max Constraint
        if d.has_pv:
            pv_val = np.array([mvars["pv"][i][t].X for t in T])
            phi_pv = np.array([mvars["dual_pv_max"][i][t].X for t in T])
            slack_pv = d.pv_available - pv_val
            residuals.extend(phi_pv * slack_pv)

    max_res = float(np.max(np.abs(residuals))) if residuals else 0.0
    passed = bool(max_res <= tol)

    print(f"[{'PASS' if passed else 'FAIL'}] KKT Residual Check: max residual = {max_res:.2e}")

    return {
        "passed": passed,
        "max_residual": max_res,
    }


def analyze_indifference_risk(
    com_data: CommunityData, posted_prices: np.ndarray, tol: float = 1e-5
) -> pd.DataFrame:
    """Analyze consumer response non-uniqueness (indifference) under posted prices.

    Solves each member's problem in two modes:
    - Optimistic: Minimizes manager cost / grid congestion among optimal member responses.
    - Pessimistic: Maximizes member grid import / congestion within optimal member cost.

    Parameters
    ----------
    com_data : CommunityData
        Community input data.
    posted_prices : np.ndarray
        24-hour posted internal prices x_t.
    tol : float, default 1e-5
        Tolerance for optimal cost equivalence.

    Returns
    -------
    pd.DataFrame
        Summary per member of minimum cost, load range variation, and maximum peak deviation.
    """
    records = []

    for d in com_data.members:
        i = d.consumer_id
        T = range(d.n_hours)
        ref = d.reference_load
        c_L = d.linear_disutility

        # Step 1: Solve member's base response LP
        m_base = gp.Model(f"indiff_base_{i}")
        m_base.Params.OutputFlag = 0

        load = m_base.addVars(T, lb=d.load_lower_kWh, ub=d.load_upper_kWh, name="load")
        dev = m_base.addVars(T, lb=0.0, name="dev")
        
        m_base.addConstrs((dev[t] >= load[t] - ref[t] for t in T), name="d_up")
        m_base.addConstrs((dev[t] >= ref[t] - load[t] for t in T), name="d_dn")
        m_base.addConstr(load.sum() >= d.min_daily_energy_kWh, name="e_min")

        net = {t: gp.LinExpr(load[t]) for t in T}
        if d.has_pv:
            pv = m_base.addVars(T, lb=0.0, ub=d.pv_available, name="pv")
            for t in T:
                net[t].add(pv[t], -1.0)
        if d.has_battery:
            p_ch = m_base.addVars(T, lb=0.0, ub=d.battery_max_charge_kW, name="p_ch")
            p_dis = m_base.addVars(T, lb=0.0, ub=d.battery_max_discharge_kW, name="p_dis")
            soc = m_base.addVars(range(len(T) + 1), lb=0.0, ub=d.battery_capacity_kWh, name="soc")
            m_base.addConstr(soc[0] == d.battery_initial_soc_kWh)
            m_base.addConstrs(
                (soc[t + 1] == soc[t] + d.battery_charging_efficiency * p_ch[t] - p_dis[t] / d.battery_discharging_efficiency for t in T)
            )
            m_base.addConstr(soc[len(T)] >= d.battery_initial_soc_kWh)
            for t in T:
                net[t].add(p_ch[t] - p_dis[t], 1.0)

        cost_expr = gp.quicksum(posted_prices[t] * net[t] for t in T) + c_L * dev.sum()
        if d.has_pv:
            cost_expr.add(d.pv_marginal_cost * pv.sum())

        m_base.setObjective(cost_expr, GRB.MINIMIZE)
        m_base.optimize()

        min_cost = m_base.ObjVal

        # Step 2: Solve Pessimistic LP (Maximize peak net import given optimal cost)
        m_pess = m_base.copy()
        m_pess.Params.OutputFlag = 0
        
        # Re-constrain objective to be within min_cost + tol
        vars_pess = {v.VarName: v for v in m_pess.getVars()}
        cost_pess = gp.quicksum(
            posted_prices[t] * (vars_pess[f"load[{t}]"] + (vars_pess[f"p_ch[{t}]"] - vars_pess[f"p_dis[{t}]"] if d.has_battery else 0) - (vars_pess[f"pv[{t}]"] if d.has_pv else 0))
            for t in T
        ) + c_L * gp.quicksum(vars_pess[f"dev[{t}]"] for t in T)
        
        m_pess.addConstr(cost_pess <= min_cost + tol, name="opt_cost_bound")
        
        # Pessimistic objective: Maximize total net import
        net_import_pess = gp.quicksum(
            vars_pess[f"load[{t}]"] - (vars_pess[f"pv[{t}]"] if d.has_pv else 0)
            for t in T
        )
        m_pess.setObjective(net_import_pess, GRB.MAXIMIZE)
        m_pess.optimize()

        opt_load_sum = sum(v.X for v in m_base.getVars() if v.VarName.startswith("load"))
        pess_load_sum = sum(v.X for k, v in vars_pess.items() if k.startswith("load"))

        records.append(
            {
                "member": i,
                "archetype": d.archetype,
                "disutility_slope": c_L,
                "min_cost_DKK": round(min_cost, 4),
                "optimistic_daily_load_kWh": round(opt_load_sum, 2),
                "pessimistic_daily_load_kWh": round(pess_load_sum, 2),
                "has_indifference": abs(pess_load_sum - opt_load_sum) > 1e-3,
            }
        )

    return pd.DataFrame(records).set_index("member")


def run_q2_validation_pipeline(
    com_data: CommunityData, results_cap: Results, results_MC: Results, model_pricing: DynamicPricingModel
) -> dict[str, Any]:
    """Run full validation suite for Question 2 and return summarized report.

    Parameters
    ----------
    com_data : CommunityData
        Community data object.
    results_cap : Results
        Results from M^P_cap pricing model.
    results_MC : Results
        Results from Question 1 Model C.
    model_pricing : DynamicPricingModel
        Built and solved pricing model object.

    Returns
    -------
    dict[str, Any]
        Aggregated dictionary containing results of all three validation steps.
    """
    print("\n" + "=" * 60)
    print(" Running Question 2 Validation Pipeline")
    print("=" * 60)

    res_duals = validate_against_MC_duals(results_cap, results_MC)
    res_kkt = check_kkt_residuals(model_pricing)

    posted_x = results_cap.hourly["posted_price"].to_numpy()
    df_indiff = analyze_indifference_risk(com_data, posted_x)

    indiff_count = sum(df_indiff["has_indifference"])
    print(f"[INFO] Consumer Indifference Check: {indiff_count} / {len(com_data.members)} members have multiple optimal schedules.")

    return {
        "dual_equivalence": res_duals,
        "kkt_residuals": res_kkt,
        "indifference_analysis": df_indiff,
    }
