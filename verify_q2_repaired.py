"""Regression checks for the repaired friend's Q2, using independent Q1 LPs."""
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
                model.var["x_price"][member,t].LB = float(prices[t])
                model.var["x_price"][member,t].UB = float(prices[t])
            result = model.solve()  # Includes independent LP and payment checks.
            print(f"PASS {name}: payment_error={result.meta['validation']['max_payment_identity_error']:.3g}; response_gap={result.meta['validation']['max_best_response_gap']:.3g}", flush=True)
            if name == "C1_zero_price":
                assert abs(result.member_costs.loc[member,"payment"]) < 1e-5
                assert abs(result.member_costs.loc[member,"disutility"]) < 1e-5
        finally:
            model.m.dispose()
    central = CommunityModel(base, "C").build()
    target = central.solve()
    central.m.dispose()
    model = DynamicPricingModel(base, time_limit=45).build()
    try:
        for t in model.T:
            model.var["x_price"][base.member_ids[0],t].Start = float(target.hourly.loc[t,"dual_balance"])
        result = model.solve()
        assert abs(result.objective-target.objective) < 1e-4
        print(f"PASS full_community_recovery: Q2={result.objective:.6f}; C={target.objective:.6f}")
    finally:
        model.m.dispose()

    # Small accounting fixtures isolate BB, strict IR and positive transfers.
    com = replace(base, members=[base.member("C1")])
    for mode in ("BB", "BB_IR", "extended"):
        model = DynamicPricingModel(com, mode=mode, benchmark_costs={"C1":1000.0}, time_limit=30).build()
        try:
            if mode == "extended":
                for t in model.T:
                    model.var["x_price"]["C1",t].LB = 10
                    model.var["x_price"]["C1",t].UB = 10
            result = model.solve()
            costs = result.member_costs.loc["C1"]
            assert abs(costs["total_cost_before_transfer"]-costs["delta_transfer"]-costs["total_cost"]) < 1e-5
            if mode == "extended":
                assert costs["delta_transfer"] > 1
                terms = result.meta["objective_terms"]
                assert abs(result.meta["solver_objective"] - result.objective - terms["transfer_penalty"]) < 1e-4
            print(f"PASS accounting_{mode}: actual-budget and independent responses checked", flush=True)
        finally:
            model.m.dispose()


if __name__ == "__main__":
    main()
