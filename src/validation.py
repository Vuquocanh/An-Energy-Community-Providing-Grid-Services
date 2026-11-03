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
