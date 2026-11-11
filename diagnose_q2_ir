"""Is BB+IR reachable *without giving up welfare* under uniform prices?

Model C (and the cap-only pricing model) attain the welfare-optimal dispatch.  This script fixes
every primal variable to that dispatch and asks the BB+IR MILP whether *any* price vector and KKT
duals support it while (i) the manager's budget balances and (ii) every member is at least as well
off as in model R.

If the fixed-dispatch problem is INFEASIBLE, then BB+IR necessarily costs welfare: no price
vector can redistribute the welfare-optimal outcome, so the requirements force a different
(the more expensive) dispatch.  This is the "how much would have to be given up" question of
Question 2, answered from the feasible side.

Usage:
    python diagnose_q2_ir.py            # base community, d1
    python diagnose_q2_ir.py --day d2
"""
from __future__ import annotations

import argparse

from gurobipy import GRB

from src.data_loader import load_community
from src.model import CommunityModel
from src.model_q2_pricing import DynamicPricingModel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base")
    ap.add_argument("--day", default="d1")
    ap.add_argument("--big-m", type=float, default=100.0)
    ap.add_argument("--time-limit", type=float, default=300)
    args = ap.parse_args()

    com = load_community(args.community, day=args.day)
    res_C = CommunityModel(com, setup="C").build().solve()
    res_R = CommunityModel(com, setup="R").build().solve()
    bench = res_R.member_costs["total"].to_dict()
    print(f"C = {res_C.objective:.4f} DKK | R = {res_R.objective:.4f} DKK")

    # The BB solution is a welfare-optimal dispatch (its objective equals C on the base case).
    bb = DynamicPricingModel(com, mode="BB", benchmark_costs=bench, big_m=args.big_m,
                             time_limit=args.time_limit).build()
    res_bb = bb.solve()
    print(f"BB = {res_bb.objective:.4f} DKK (welfare gap to C: {res_bb.objective - res_C.objective:+.4f})")

    # Fix the whole dispatch, keep prices and duals free, and impose BB + IR.
    ir = DynamicPricingModel(com, mode="BB_IR", benchmark_costs=bench, big_m=args.big_m,
                             time_limit=args.time_limit).build()
    n_fixed = 0
    for family in ("load", "dev", "pv", "p_ch", "p_dis", "soc"):
        for i, values in bb.mvar.get(family, {}).items():
            for t, var in values.items():
                fixed = ir.mvar[family][i][t]
                fixed.LB = fixed.UB = var.X
                n_fixed += 1
    for name in ("import", "export", "exceedance"):
        for t, var in bb.var[name].items():
            fixed = ir.var[name][t]
            fixed.LB = fixed.UB = var.X
            n_fixed += 1
    print(f"Fixing {n_fixed} primal variables to the BB dispatch; searching only prices and duals ...")

    ir.m.optimize()
    if ir.m.Status == GRB.OPTIMAL:
        print("FEASIBLE: a supporting price vector satisfies BB+IR at the welfare-optimal dispatch.")
        print("  prices:", [round(ir.var['x_price'][ir.ids[0], t].X, 3) for t in ir.T])
    elif ir.m.Status == GRB.INFEASIBLE:
        print("INFEASIBLE: no supporting price vector satisfies BB+IR at the welfare-optimal dispatch.")
        print("  => individual rationality costs welfare under uniform prices.")
    else:
        print(f"UNRESOLVED: Gurobi status {ir.m.Status}")

    bb.m.dispose()
    ir.m.dispose()


if __name__ == "__main__":
    main()
