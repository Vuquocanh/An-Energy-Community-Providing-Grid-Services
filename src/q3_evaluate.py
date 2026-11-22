"""Question 3 -- evaluation of a posted price vector on scenarios (the out-of-sample protocol, F1(b)).

Given the prices x_t and one scenario (a :class:`CommunityData` from ``q3_scenarios``), every member solves its own
LP under its realised reference profile, daily minimum and bounds (the same member block as ``CommunityModel``, so the
responses are those of Questions 1-2); the meter aggregates the responses; the DSO bill, the exceedance, the payments
and the budget follow; individual rationality is checked against the member's outside option in that scenario.

Tie-break (feedback 1.5). A member's best response is a set in the hours where it is indifferent (|x_t - mu_i| = c_L,
x_t = c_PV, battery spreads), and with linear costs the welfare-optimal prices sit exactly on those thresholds. Three
readings of one price vector are therefore reported:
    "vertex"      the schedule the LP solver happens to return (neither reading; arbitrary);
    "optimistic"  the element of the best-response sets with the lowest community cost (what the single-level DP/SDP
                  assumes): members' costs fixed at their optima, then the DP objective minimised, one LP;
    "pessimistic" an attainable adversarial element: from the optimistic reading, maximise the joint import in the hours
                  where the cap binds, one more LP;
    import range  per hour, the smallest and largest community import over the best-response sets (24 x 2 LPs); the
                  pessimistic exceedance bound is sum_t (max import_t - cap_t)^+.

    member_benchmarks(scenario)                       -> {consumer_id: min(cost_R, cost_I)} under the scenario's theta
    evaluate_prices(prices, scenario, bench, ...)     -> dict: community totals + per-member DataFrame of one scenario
    evaluate_on_set(prices, scenarios, label, ...)    -> (DataFrame one row per scenario, DataFrame one row per member x scenario)
    summarize(df_scen, df_mem, target_cost)           -> dict of the metrics of F2 (targeted vs achieved)
"""
from __future__ import annotations

import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from .data_loader import CommunityData
from .model import CommunityModel
from .q3_scenarios import Scenario

TIE_BREAKS = ("vertex", "optimistic", "pessimistic")


# ----------------------------------------------------------------------------------------------------------------------
# outside options
# ----------------------------------------------------------------------------------------------------------------------
def member_benchmarks(scenario: Scenario, end_soc_mode: str = "constraint") -> dict[str, float]:
    """Outside option of every member in this scenario: the cheaper of standing alone without a contract (R) and with
    an individual contract (I), both solved under the scenario's reference profile (two LPs per member)."""
    com = scenario.community
    out = {}
    for setup in ("R", "I"):
        mdl = CommunityModel(com, setup=setup, end_soc_mode=end_soc_mode).build()
        res = mdl.solve()
        out[setup] = res.member_costs["total"].to_dict()
        mdl.m.dispose()
    return {i: min(out["R"][i], out["I"][i]) for i in com.member_ids}


# ----------------------------------------------------------------------------------------------------------------------
# the members' best responses to fixed prices, with the tie-break
# ----------------------------------------------------------------------------------------------------------------------
def _members_best_response(prices: np.ndarray, com: CommunityData, end_soc_mode: str = "constraint",
                           tie_break: str = "vertex", with_range: bool = False, tol: float = 1e-6) -> dict:
    """Every member's optimal response to the posted prices (one LP with all members: they are separable).

    Returns q[i][t] (net consumption), J[i] (private cost) and opt[i] (the member's optimum) for the requested
    tie-break, plus, with ``with_range``, the hourly min/max community import over the best-response sets."""
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"tie_break must be one of {TIE_BREAKS}")
    follower = CommunityModel(com, "C", end_soc_mode=end_soc_mode)   # the setup only names the model; no meter is added
    for d in com.members:
        follower._add_member(d)
    m, T, ids = follower.m, follower.T, com.member_ids
    m.Params.OutputFlag = 0

    obj = {}
    for i in ids:
        e = gp.LinExpr()
        for per_member in follower.mexpr.values():
            if i in per_member:
                e.add(per_member[i])
        e.add(gp.quicksum(float(prices[t]) * follower.net[i][t] for t in T))
        obj[i] = e
    m.setObjective(gp.quicksum(obj.values()), GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"member re-solve failed with status {m.Status}")

    def _read():
        q = {i: np.array([follower.net[i][t].getValue() for t in T]) for i in ids}
        J = {i: sum(pm[i].getValue() for pm in follower.mexpr.values() if i in pm) for i in ids}
        return q, J

    opt = {i: obj[i].getValue() for i in ids}
    q, J = _read()
    out = {"q": q, "J": J, "opt": opt, "tie_break": "vertex"}

    if tie_break in ("optimistic", "pessimistic") or with_range:
        # stay inside every member's best-response set
        for i in ids:
            m.addConstr(obj[i] <= opt[i] + tol * max(1.0, abs(opt[i])), name=f"best_response_{i}")
        q_tot = {t: gp.quicksum(follower.net[i][t] for i in ids) for t in T}

    if tie_break in ("optimistic", "pessimistic"):
        c = com.contract
        cap = np.asarray(c.capacity_limit_kW, dtype=float)
        p = np.asarray(com.energy_price, dtype=float)
        f = 1.0 - c.tariff_discount_ratio
        exc = m.addVars(T, lb=0.0, name="exc")
        imp = m.addVars(T, lb=0.0, name="imp")
        exp = m.addVars(T, lb=0.0, name="exp")
        for t in T:
            m.addConstr(imp[t] - exp[t] == q_tot[t], name=f"meter_{t}")
            m.addConstr(exc[t] >= imp[t] - cap[t], name=f"cap_{t}")
        # the community cost of the schedule (the DP objective), evaluated on the members' best responses
        community_cost = gp.quicksum(p[t] * (imp[t] - exp[t]) + f * com.import_tariff * (imp[t] - exc[t])
                                     + com.import_tariff * exc[t] + c.penalty_DKK_per_kWh * exc[t]
                                     + f * com.export_tariff * exp[t] for t in T)
        for per_member in follower.mexpr.values():
            community_cost.add(gp.quicksum(per_member.values()))
        # optimistic: the element of the best-response sets the single-level model assumes (lowest community cost)
        m.setObjective(community_cost, GRB.MINIMIZE)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"optimistic tie-break failed with status {m.Status}")
        q, J = _read()
        out.update({"q": q, "J": J, "tie_break": "optimistic"})
        if tie_break == "pessimistic":
            # pessimistic: one joint schedule that maximises the import in the hours where the cap binds (or is
            # exceeded) under the optimistic reading; an attainable adversarial element, not the per-hour bound
            q_opt_tot = sum(q.values())
            binding = [t for t in T if q_opt_tot[t] >= cap[t] - 1e-6]
            if binding:
                m.setObjective(gp.quicksum(q_tot[t] for t in binding), GRB.MAXIMIZE)
                m.optimize()
                if m.Status != GRB.OPTIMAL:
                    raise RuntimeError(f"pessimistic tie-break failed with status {m.Status}")
                q, J = _read()
            out.update({"q": q, "J": J, "tie_break": "pessimistic", "binding_hours": binding})
        # the meter variables stay in the model: they do not restrict q_tot, so the range stage is unaffected

    if with_range:
        lo, hi = np.zeros(len(T)), np.zeros(len(T))
        for t in T:
            m.setObjective(q_tot[t], GRB.MAXIMIZE); m.optimize(); hi[t] = m.ObjVal
            m.setObjective(q_tot[t], GRB.MINIMIZE); m.optimize(); lo[t] = m.ObjVal
        out["import_min"], out["import_max"] = lo, hi

    m.dispose()
    return out


# ----------------------------------------------------------------------------------------------------------------------
# one scenario
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_prices(prices, scenario: Scenario, benchmarks: dict[str, float] | None = None,
                    end_soc_mode: str = "constraint", tie_break: str = "optimistic", with_range: bool = True) -> dict:
    """Evaluate the prices on one scenario: re-solve (with the tie-break), aggregate at the meter, settle, check IR."""
    com = scenario.community
    c = com.contract
    x = np.asarray(prices, dtype=float)
    p = np.asarray(com.energy_price, dtype=float)
    f = 1.0 - c.tariff_discount_ratio
    tau_imp, tau_exp, alpha = com.import_tariff, com.export_tariff, c.penalty_DKK_per_kWh
    cap = np.asarray(c.capacity_limit_kW, dtype=float)

    t0 = time.perf_counter()
    br = _members_best_response(x, com, end_soc_mode, tie_break=tie_break, with_range=with_range)
    q_tot = sum(br["q"].values())
    imp, exp = np.maximum(q_tot, 0.0), np.maximum(-q_tot, 0.0)
    exc = np.maximum(imp - cap, 0.0)
    energy = float(np.sum(p * (imp - exp)))
    tariffs = float(np.sum(f * tau_imp * (imp - exc) + tau_imp * exc + f * tau_exp * exp))
    penalty = float(alpha * exc.sum())
    bill = energy + tariffs + penalty

    rows = []
    for d in com.members:
        i = d.consumer_id
        pay = float(np.sum(x * br["q"][i]))
        cost = pay + br["J"][i]
        bench = benchmarks.get(i, np.nan) if benchmarks else np.nan
        rows.append({"scenario": scenario.id, "member": i, "archetype": d.archetype, "theta": scenario.theta_profile[i],
                     "payment": pay, "private_cost": br["J"][i], "cost": cost, "member_optimum": br["opt"][i],
                     "benchmark": bench, "ir_violation": max(0.0, cost - bench) if benchmarks else np.nan})
    members = pd.DataFrame(rows)
    payments = float(members["payment"].sum())
    private = float(members["private_cost"].sum())
    totals = {
        "scenario": scenario.id, "probability": scenario.probability, "seed": scenario.seed, "tie_break": br["tie_break"],
        "community_cost": bill + private, "dso_bill": bill, "energy": energy, "tariffs": tariffs, "penalty": penalty,
        "private_cost": private, "payments": payments, "budget": payments - bill,
        "exceedance_kWh": float(exc.sum()), "max_exceedance_kW": float(exc.max()), "hours_above_cap": int((exc > 1e-6).sum()),
        "violated": bool(exc.max() > 1e-6), "import_kWh": float(imp.sum()), "peak_import_kW": float(imp.max()),
        "ref_energy_kWh": float(sum(d.reference_load.sum() for d in com.members)),
        "n_ir_violations": int((members["ir_violation"] > 1e-4).sum()) if benchmarks else -1,
    }
    hourly = pd.DataFrame({"price": x, "import": imp, "export": exp, "cap": cap, "exceedance": exc})
    if with_range:
        pess_exc = np.maximum(br["import_max"] - cap, 0.0)
        totals.update({"exceedance_pess_bound_kWh": float(pess_exc.sum()),
                       "violated_pess_bound": bool(pess_exc.max() > 1e-6),
                       "import_range_kWh": float((br["import_max"] - br["import_min"]).sum()),
                       "hours_indifferent": int(((br["import_max"] - br["import_min"]) > 1e-6).sum())})
        hourly["import_min"], hourly["import_max"] = br["import_min"], br["import_max"]
    totals["solve_s"] = time.perf_counter() - t0
    return {"totals": totals, "members": members, "hourly": hourly}


# ----------------------------------------------------------------------------------------------------------------------
# a set of scenarios, and the F2 metrics
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_on_set(prices, scenarios: list[Scenario], label: str = "", with_benchmarks: bool = True,
                    end_soc_mode: str = "constraint", tie_break: str = "optimistic", with_range: bool = True,
                    verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the prices on every scenario of a set. Returns one row per scenario and one row per (member, scenario)."""
    tot, mem = [], []
    t0 = time.perf_counter()
    for k, s in enumerate(scenarios):
        bench = member_benchmarks(s, end_soc_mode) if with_benchmarks else None
        r = evaluate_prices(prices, s, bench, end_soc_mode, tie_break=tie_break, with_range=with_range)
        r["totals"]["label"] = label
        tot.append(r["totals"])
        mem.append(r["members"].assign(label=label))
        if verbose and (k + 1) % 20 == 0:
            print(f"  [{label}] {k + 1}/{len(scenarios)} scenarios evaluated ({time.perf_counter() - t0:.0f} s)")
    return pd.DataFrame(tot), pd.concat(mem, ignore_index=True)


def summarize(df_scen: pd.DataFrame, df_mem: pd.DataFrame | None = None, target_cost: float | None = None) -> dict:
    """The F2 metrics: targeted vs achieved cost (distribution), violation frequency and exceedance, budget, IR."""
    w = df_scen["probability"].to_numpy()
    w = w / w.sum()
    cost = df_scen["community_cost"].to_numpy()
    out = {
        "n_scenarios": len(df_scen), "tie_break": str(df_scen["tie_break"].iloc[0]),
        "cost_mean": float(np.dot(w, cost)), "cost_p10": float(np.percentile(cost, 10)),
        "cost_p50": float(np.percentile(cost, 50)), "cost_p90": float(np.percentile(cost, 90)),
        "cost_min": float(cost.min()), "cost_max": float(cost.max()),
        "violation_frequency": float(np.dot(w, df_scen["violated"].to_numpy(dtype=float))),
        "exceedance_mean_kWh": float(np.dot(w, df_scen["exceedance_kWh"])),
        "exceedance_max_kWh": float(df_scen["exceedance_kWh"].max()),
        "penalty_mean": float(np.dot(w, df_scen["penalty"])), "penalty_max": float(df_scen["penalty"].max()),
        "budget_mean": float(np.dot(w, df_scen["budget"])), "budget_p10": float(np.percentile(df_scen["budget"], 10)),
        "budget_p90": float(np.percentile(df_scen["budget"], 90)),
        "share_days_deficit": float(np.dot(w, (df_scen["budget"] < -1e-4).to_numpy(dtype=float))),
        "solve_s_total": float(df_scen["solve_s"].sum()),
    }
    if "exceedance_pess_bound_kWh" in df_scen:
        out.update({"violation_frequency_pess_bound": float(np.dot(w, df_scen["violated_pess_bound"].to_numpy(dtype=float))),
                    "exceedance_pess_bound_mean_kWh": float(np.dot(w, df_scen["exceedance_pess_bound_kWh"])),
                    "import_range_mean_kWh": float(np.dot(w, df_scen["import_range_kWh"])),
                    "hours_indifferent_mean": float(np.dot(w, df_scen["hours_indifferent"]))})
    if target_cost is not None:
        out["target_cost"] = float(target_cost)
        out["optimism_gap"] = out["cost_mean"] - float(target_cost)
    if df_mem is not None and df_mem["benchmark"].notna().any():
        viol = df_mem["ir_violation"] > 1e-4
        out["ir_violation_share"] = float(viol.mean())
        out["ir_violation_by_archetype"] = df_mem.assign(v=viol).groupby("archetype")["v"].mean().round(3).to_dict()
        out["ir_violation_by_member"] = df_mem.assign(v=viol).groupby("member")["v"].mean().round(3).to_dict()
        out["ir_expected_shortfall_by_member"] = df_mem.groupby("member")["ir_violation"].mean().round(3).to_dict()
    return out
