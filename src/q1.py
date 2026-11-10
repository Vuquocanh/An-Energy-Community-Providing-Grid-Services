"""Question 1 experiments: is the community worth forming? R, I and C on one day.

    python main.py --q1-base     # R, I and C on the base community: metrics table + CSVs in results/q1_base/

Metrics, every one against the reference R (members alone, no contract):
    value_DKK            community   Cost_R - Cost_X, what the contract is worth to the community (> 0: worth signing)
    max_exceedance_kW    DSO         max_t (import_t - cap_t)+ at the connection point: is the feeder still overloaded?
    hours_above_cap      DSO         number of hours with import_t > cap_t: how often it is
The value of pooling is value_C - value_I = Cost_I - Cost_C (same contract, with and without the community).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import CommunityData, load_community
from .model import CommunityModel, Results

TOL = 1e-6      # kW: an import above the cap by less than this is counted as at the cap
SETUPS_Q1 = ("R", "I", "C")


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


def run_q1_base(out: Path | str, day: str = "d1") -> dict[str, Results]:
    """R, I and C on the base community on ``day``: print and save the metrics and the cost terms.

    Writes to ``out``: metrics_<day>.csv (one row per set-up), cost_terms_<day>.csv (DKK per term and set-up)
    and the hourly / members / member_costs tables of every run (``Results.save``). Returns {setup: Results}.
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

    print(f"\nQuestion 1, base community, {day} - metrics against R:")
    print(table.round(2).to_string())
    print(f"value of pooling (Cost_I - Cost_C): {res['I'].objective - res['C'].objective:.2f} DKK")
    print("\ncost terms (DKK):")
    print(terms.round(2).to_string())
    return res
