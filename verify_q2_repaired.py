"""Regression checks for the merged Question 2 model, using independent Q1 LPs.

Run it from the package root:

    python verify_q2_repaired.py

Every case fixes the posted prices, solves the merged MILP and lets ``validate_solution`` check
that (i) the members' dispatch is really their own LP optimum at those prices and (ii) the
dual-derived payments equal ``sum_t x_t q_{i,t}``.  The last block checks the accounting of the
requirements (BB, hard IR, soft IR, transfers).
"""
from dataclasses import replace

import numpy as np
from src.data_loader import load_community
from src.model import CommunityModel
from src.model_q2_pricing import DynamicPricingModel


def main():
    base = load_community("base")
    cases = [
        ("C1_zero_price", "C1", np.zeros(24), "constraint"),
        ("PV_zero_price", "C1_PV", np.zeros(24), "constraint"),
        ("PV_marginal_price", "C1_PV", np.full(24, 1.41), "constraint"),
        ("battery_cyclic", "C1_B", np.linspace(-.5, 4, 24), "constraint"),
        ("battery_free", "C1_B", np.linspace(.1, 4, 24), "free"),
        ("battery_terminal_value", "C1_PVB", np.linspace(.1, 4, 24), "objective"),
    ]
    for name, member, prices, end_mode in cases:
        com = replace(base, members=[base.member(member)])
        model = DynamicPricingModel(com, end_soc_mode=end_mode, time_limit=30).build()
        try:
            for t in model.T:
                model.var["x_price"][member, t].LB = float(prices[t])
                model.var["x_price"][member, t].UB = float(prices[t])
            result = model.solve()  # includes the independent LP and payment checks
            print(f"PASS {name}: payment_error="
                  f"{result.meta['validation']['max_payment_identity_error']:.3g}; "
                  f"response_gap={result.meta['validation']['max_best_response_gap']:.3g}", flush=True)
            if name == "C1_zero_price":
                assert abs(result.member_costs.loc[member, "payment"]) < 1e-5
                assert abs(result.member_costs.loc[member, "disutility"]) < 1e-5
        finally:
            model.m.dispose()

    central = CommunityModel(base, "C").build()
    target = central.solve()
    central.m.dispose()
    model = DynamicPricingModel(base, time_limit=45).build()
    try:
        for t in model.T:
            model.var["x_price"][base.member_ids[0], t].Start = float(target.hourly.loc[t, "dual_balance"])
        result = model.solve()
        assert abs(result.objective - target.objective) < 1e-4
        print(f"PASS full_community_recovery: Q2={result.objective:.6f}; C={target.objective:.6f}")
    finally:
        model.m.dispose()

    # Small accounting fixtures isolate BB, strict IR, soft IR and positive transfers.
    com = replace(base, members=[base.member("C1")])
    for mode in ("BB", "BB_IR", "soft_ir", "extended"):
        benchmark = {"C1": 1000.0} if mode != "soft_ir" else {"C1": 1.0}
        model = DynamicPricingModel(com, mode=mode, benchmark_costs=benchmark, time_limit=30).build()
        try:
            if mode == "extended":
                for t in model.T:
                    model.var["x_price"]["C1", t].LB = 10
                    model.var["x_price"]["C1", t].UB = 10
            result = model.solve()
            costs = result.member_costs.loc["C1"]
            assert abs(costs["total_cost_before_transfer"] - costs["delta_transfer"] - costs["total_cost"]) < 1e-5
            if mode == "extended":
                assert costs["delta_transfer"] > 1
                terms = result.meta["objective_terms"]
                assert abs(result.meta["solver_objective"] - result.objective - terms["transfer_penalty"]) < 1e-4
            if mode == "soft_ir":
                # The benchmark is unreachable at x = 10: the slack must be positive and the soft
                # constraint must hold exactly at the optimum.
                assert costs["ir_slack"] > 1e-3, costs["ir_slack"]
                assert costs["total_cost"] - costs["ir_slack"] <= 1.0 + 1e-6
                assert result.meta["objective_terms"]["total_ir_slack"] > 1e-3
                print(f"PASS accounting_soft_ir: slack={costs['ir_slack']:.4f} DKK, "
                      f"member cost={costs['total_cost']:.4f} DKK, benchmark=1.0 DKK", flush=True)
            else:
                print(f"PASS accounting_{mode}: actual-budget and independent responses checked", flush=True)
        finally:
            model.m.dispose()


if __name__ == "__main__":
    main()
