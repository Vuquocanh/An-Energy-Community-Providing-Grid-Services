"""Question 1 experiments: is the community worth forming? R, I and C on one day.

    python main.py --q1-base            # R, I and C on the base community: metrics table, CSVs and figures in results/q1_base/
    python main.py --q1-sensitivity     # sweeps of the discount beta and of the cap: CSVs and figure in results/q1_sensitivity/

Metrics, every one against the reference R (members alone, no contract):
    value_DKK            community   Cost_R - Cost_X, what the contract is worth to the community (> 0: worth signing)
    max_exceedance_kW    DSO         max_t (import_t - cap_t)+ at the connection point: is the feeder still overloaded?
    hours_above_cap      DSO         number of hours with import_t > cap_t: how often it is
The value of pooling is value_C - value_I = Cost_I - Cost_C (same contract, with and without the community).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .data_loader import CommunityData, load_community
from .model import CommunityModel, Results
from .plotting import (plot_import_vs_cap, plot_internal_price, plot_member_gains, plot_sensitivity,
                       plot_shift_heatmap)

TOL = 1e-6      # kW: an import above the cap by less than this is counted as at the cap
SETUPS_Q1 = ("R", "I", "C")
# values of every contract term swept, one term at a time (base: beta 0.75, cap factor 1.3)
SWEEPS = {
    "beta": [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    "cap_factor": [0.8, 0.9, 1.0, 1.1, 1.15, 1.3, 1.6, 1.8, 1.9, 2.0, 2.5, 3.0, 3.5, 4.0],
}


def with_discount(com: CommunityData, beta: float) -> CommunityData:
    """Copy of ``com`` with the tariff discount ``beta`` of the contract (models I and C)."""
    return replace(com, contract=replace(com.contract, tariff_discount_ratio=beta))


def with_cap_factor(com: CommunityData, f: float) -> CommunityData:
    """Copy of ``com`` with the cap at f x E_R / 24 (the cap rule of the contract file; f = 1.3 in the base case).

    The community's cap and every individual cap of I (kappa_i x cap) are scaled by the same factor, so that I
    and C keep facing the same contract.
    """
    c = com.contract
    k = f / c.capacity_limit_daily_energy_factor
    return replace(com, contract=replace(
        c, capacity_limit_kW=c.capacity_limit_kW * k, capacity_limit_daily_energy_kWh=c.capacity_limit_daily_energy_kWh * k,
        capacity_limit_daily_energy_factor=f,
        individual_capacity_limit_kW={i: v * k for i, v in c.individual_capacity_limit_kW.items()}))


VARIANTS = {"beta": with_discount, "cap_factor": with_cap_factor}


def solve_setups(com: CommunityData, setups: tuple[str, ...] = SETUPS_Q1) -> dict[str, Results]:
    """Solve every set-up on the same data: {setup: Results}."""
    return {s: CommunityModel(com, setup=s).build().solve() for s in setups}


def metrics(res: Results, ref: Results) -> dict[str, float]:
    """The three Q1 metrics of one run against the reference run ``ref`` (R), and the totals behind them.

    value_DKK = Cost_ref - Cost_res (DKK/day, community); max_exceedance_kW = largest hourly import above the
    cap (kW, DSO); hours_above_cap (DSO). Also: cost_DKK, exceedance_kWh (sum over the day), peak_import_kW and
    energy_shifted_kWh = sum |load - ref| / 2 (every kWh moved leaves one hour and enters another, the daily
    minimum being met with equality).
    """
    h = res.hourly
    return {
        "value_DKK": ref.objective - res.objective,
        "max_exceedance_kW": h["exceedance"].max(),
        "hours_above_cap": int((h["import"] > h["cap"] + TOL).sum()),
        "cost_DKK": res.objective,
        "exceedance_kWh": h["exceedance"].sum(),
        "peak_import_kW": h["import"].max(),
        "energy_shifted_kWh": res.members["deviation"].sum() / 2,
    }


def import_change_by_source(res: Results, ref: Results) -> pd.DataFrame:
    """Hourly change of the import at the connection point, ``res`` minus ``ref`` (kWh/h), split by source.

    At every meter import = sum_i q_i + export, with q_i = load + charge - pv - discharge, so
        d(import) = d(load) of each slope class L1-L4 + d(charge - discharge) - d(pv) + d(export).
    The last column, 'export / netting', is the change of what crosses the meter outwards: negative when
    surplus that was exported (R, I) is used behind the meter instead (C). Also returns 'import' = the sum.
    """
    def by_hour(r: Results, col: str, ids=None):
        m = r.members if ids is None else r.members.loc[list(ids)]
        return m[col].groupby(level="hour").sum() if col in m else 0.0

    out = pd.DataFrame(index=res.hourly.index)
    slope = res.member_costs["archetype"].str[:2]          # "L1" .. "L4"
    for k in sorted(slope.unique()):
        ids = slope.index[slope == k]
        out[k] = by_hour(res, "load", ids) - by_hour(ref, "load", ids)
    out["battery"] = (by_hour(res, "p_ch") - by_hour(res, "p_dis")) - (by_hour(ref, "p_ch") - by_hour(ref, "p_dis"))
    out["pv"] = -(by_hour(res, "pv") - by_hour(ref, "pv"))
    out["export / netting"] = res.hourly["export"] - ref.hourly["export"]
    out["import"] = res.hourly["import"] - ref.hourly["import"]
    return out


def shift_by_class(res: dict[str, Results]) -> pd.DataFrame:
    """Energy shifted by every load class, sum |l - ref| / 2 over its members (kWh/day), and the battery
    discharge (kWh/day), for every setup in ``res``: one row per class L1-L4 and 'battery discharge', one column
    per setup."""
    out = {}
    for s, r in res.items():
        slope = r.member_costs["archetype"].str[:2]          # "L1" .. "L4"
        shifted = (r.members["deviation"].groupby(level="member").sum() / 2).groupby(slope).sum()
        out[s] = {**shifted.to_dict(), "battery discharge": r.members["p_dis"].sum()}
    return pd.DataFrame(out)


def settlement(res: dict[str, Results]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Dual settlement of the community contract: member i pays sum_t lambda_t q_i,t, lambda_t the dual of the
    community balance in C (the one Gurobi returns; it is not unique in the hours at the cap).

    Returns (members, budget), in DKK/day. members, one row per member: archetype, payment (< 0: the member is
    paid), cost_C = its disutility + PV cost + payment, cost_R and cost_I (its total cost in R and I),
    gain_vs_R = cost_R - cost_C and gain_vs_I = cost_I - cost_C (> 0: better off in C), IR_vs_R, IR_vs_I (gain >= 0).
    budget: collected = sum of the payments; energy, tariff, penalty and paid_to_DSO; surplus = collected - paid;
    cap_rent = sum_t dual_cap_t x cap_t, equal to the surplus by complementary slackness.
    """
    C = res["C"]
    pay = C.members["net"].mul(C.hourly["dual_balance"], level="hour").groupby(level="member").sum()
    t = pd.DataFrame(index=C.member_costs.index)
    t["archetype"] = C.member_costs["archetype"]
    t["payment_DKK"] = pay
    t["cost_C_DKK"] = C.member_costs["total"] + pay
    t["cost_R_DKK"] = res["R"].member_costs["total"]
    t["cost_I_DKK"] = res["I"].member_costs["total"]
    t["gain_vs_R_DKK"] = t["cost_R_DKK"] - t["cost_C_DKK"]
    t["gain_vs_I_DKK"] = t["cost_I_DKK"] - t["cost_C_DKK"]
    t["IR_vs_R"] = t["gain_vs_R_DKK"] >= -TOL
    t["IR_vs_I"] = t["gain_vs_I_DKK"] >= -TOL

    terms = C.meta["objective_terms"]
    paid = terms["energy"] + terms["tariff"] + terms["penalty"]
    budget = {"collected_DKK": pay.sum(), "energy_DKK": terms["energy"], "tariff_DKK": terms["tariff"],
              "penalty_DKK": terms["penalty"], "paid_to_DSO_DKK": paid, "surplus_DKK": pay.sum() - paid,
              "cap_rent_DKK": (C.hourly["dual_cap"] * C.hourly["cap"]).sum()}
    return t, budget


def run_q1_base(out: Path | str, day: str = "d1") -> dict[str, Results]:
    """R, I and C on the base community on ``day``: print and save the metrics and the cost terms.

    Writes to ``out``: metrics_<day>.csv (one row per set-up), cost_terms_<day>.csv (DKK per term and set-up),
    import_change_<I, C>_vs_R_<day>.csv (``import_change_by_source``), shift_by_class_<day>.csv
    (``shift_by_class``), settlement_<day>.csv and budget_<day>.csv (``settlement``), the figures
    fig_import_vs_cap, fig_shift, fig_internal_price and fig_member_gains (``plotting``), and the hourly /
    members / member_costs tables of every run (``Results.save``). Returns {setup: Results}.
    """
    com = load_community("base", day=day)
    res = solve_setups(com)
    table = pd.DataFrame({s: metrics(r, res["R"]) for s, r in res.items()}).T
    terms = pd.DataFrame({s: r.meta["objective_terms"] for s, r in res.items()}).T.fillna(0.0)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"metrics_{day}.csv", index_label="setup")
    terms.to_csv(out / f"cost_terms_{day}.csv", index_label="setup")
    for r in res.values():
        r.save(out)
    changes = {s: import_change_by_source(res[s], res["R"]) for s in ("I", "C")}
    for s, change in changes.items():
        change.to_csv(out / f"import_change_{s}_vs_R_{day}.csv", index_label="hour")
    plot_import_vs_cap(res, save_to=out / f"fig_import_vs_cap_{day}.png")
    plot_shift_heatmap(changes, save_to=out / f"fig_shift_{day}.png")
    shifts = shift_by_class(res)
    shifts.to_csv(out / f"shift_by_class_{day}.csv", index_label="source")
    members, budget = settlement(res)
    members.to_csv(out / f"settlement_{day}.csv", index_label="member")
    pd.Series(budget, name="DKK").to_csv(out / f"budget_{day}.csv", index_label="item")
    plot_internal_price(res["C"], com, save_to=out / f"fig_internal_price_{day}.png")
    plot_member_gains(members, save_to=out / f"fig_member_gains_{day}.png")

    print(f"\nQuestion 1, base community, {day} - metrics against R:")
    print(table.round(2).to_string())
    print(f"value of pooling (Cost_I - Cost_C): {res['I'].objective - res['C'].objective:.2f} DKK")
    print("\ncost terms (DKK):")
    print(terms.round(2).to_string())
    print("\nenergy shifted by load class and battery discharge (kWh/day):")
    print(shifts.round(2).to_string())
    print("\nsettlement of C (DKK/day; gain > 0: the member is better off in C):")
    print(members.round(2).to_string())
    print("\nbudget of the community manager (DKK/day):")
    print(pd.Series(budget).round(2).to_string())
    return res


def sweep(com: CommunityData, term: str, values: list[float]) -> pd.DataFrame:
    """Metrics of R, I and C (``metrics``, against R on the same data) for every value of one contract term,
    ``term`` = "beta" (``with_discount``) or "cap_factor" (``with_cap_factor``): one row per (value, setup)."""
    rows = []
    for v in values:
        res = solve_setups(VARIANTS[term](com, v))
        rows += [{term: v, "setup": s, **metrics(r, res["R"])} for s, r in res.items()]
    return pd.DataFrame(rows)


def run_q1_sensitivity(out: Path | str) -> dict[str, pd.DataFrame]:
    """Sweeps of the discount beta and of the cap factor f on the base community (d1), one term at a time with
    the other at its base value (``SWEEPS``). Writes sweep_<term>.csv, fig_sensitivity.png (the cap: value and
    max exceedance) and fig_sensitivity_full.png (both terms, all three metrics) to ``out``; returns
    {term: sweep table}."""
    com = load_community("base")
    c = com.contract
    base = {"beta": c.tariff_discount_ratio, "cap_factor": c.capacity_limit_daily_energy_factor}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sweeps = {term: sweep(com, term, values) for term, values in SWEEPS.items()}
    for term, t in sweeps.items():
        t.to_csv(out / f"sweep_{term}.csv", index=False)
        print(f"\nQuestion 1, sweep of {term} (base {base[term]}), metrics against R:")
        print(t.pivot(index=term, columns="setup",
                      values=["value_DKK", "max_exceedance_kW", "hours_above_cap"]).round(2).to_string())
    plot_sensitivity({"cap_factor": sweeps["cap_factor"]}, base, rows=("value_DKK", "max_exceedance_kW"),
                     save_to=out / "fig_sensitivity.png")
    plot_sensitivity(sweeps, base, save_to=out / "fig_sensitivity_full.png")
    return sweeps
