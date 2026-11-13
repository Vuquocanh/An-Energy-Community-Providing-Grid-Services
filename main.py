"""Entry point of the Assignment 2 package: load a community and run the experiments.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --q2                              # Question 2: cap_only -> BB -> BB+IR (-> soft IR) + figures
    python main.py --q2 --q2-time-limit 900          # give the BB+IR / soft-IR MILPs more time
    python main.py --q2-sweep                        # Question 2 contract-cap sensitivity study + figure
    python main.py --q2 --day d2                     # Question 2 experiments on day d2
    python main.py --verify                          # Question 1 validation checks
    python main.py --q1-base                         # Question 1: R, I and C on the base case (results/q1_base/)
    python main.py --plot-data                       # input data of d1-d7: prices, reference load, PV, cap (results/data/)
    python main.py --q1-sensitivity                  # Question 1: sweeps of the discount and of the cap (results/q1_sensitivity/)
    python main.py --demo                            # scenario-generation demonstration (Question 3)

The Q2 flow follows the mandatory decomposition of the assignment:
    step 1  cap only                 (M^P_cap)
    step 2  + budget balance         (M^P_BB)
    step 3  + individual rationality (M^P_BB+IR); if that MILP is infeasible or does not finish,
            the soft-IR relaxation (M^P_soft) reports the minimum compensation and the blocking
            members, and the extended model (M^P_ext) the lump-sum transfers when it is solvable.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
from gurobipy import GRB

from src.metrics import (
    calc_budget_balance_summary,
    calc_efficiency_cost,
    calc_welfare_recovery_ratio,
    identify_blocking_members,
    summarize_q2_experiments,
)
from src.model import CommunityModel
from src.model_q2_pricing import DynamicPricingModel, NoPricingSolution
from src.plotting import (
    plot_day_data,
    plot_q2_3step_decomposition,
    plot_q2_cap_sensitivity,
    plot_q2_indifference_risk,
    plot_q2_member_gains_and_blocking,
    plot_q2_posted_prices,
)
from src.data_loader import DAYS, load_community, list_communities
from src.q1 import run_q1_base, run_q1_sensitivity
from src.validation import analyze_indifference_risk, run_q1_checks, run_q2_validation_pipeline

RESULTS_DIR = Path(__file__).resolve().parent / "results"

#-----------q2----------------


def _print_checks(name: str, res) -> None:
    """Print the independent (model-internal) validation numbers of one pricing run."""
    checks = res.meta.get("validation", {})
    if not checks:
        return
    txt = ", ".join(f"{k}={v:.3g}" for k, v in checks.items())
    print(f"  [validation] {name}: {txt}")


def _build_and_solve(com, mode: str, benchmarks: dict[str, float] | None,
                     time_limit: float, warm_from=None, mip_focus: int = 0,
                     big_m: float = 100.0, mip_gap: float = 1e-3):
    """Build one pricing variant, optionally warm-start it, and solve it.

    Returns ``(model, results)``; ``results`` is None when the solver produced no incumbent.
    """
    model = DynamicPricingModel(com, mode=mode, benchmark_costs=benchmarks, big_m=big_m,
                                mip_gap=mip_gap, time_limit=time_limit, mip_focus=mip_focus).build()
    if warm_from is not None:
        n = model.warm_start_from(warm_from)
        print(f"  [warm start] {n} variables initialised from {warm_from.mode}")
    try:
        return model, model.solve()
    except NoPricingSolution as exc:
        model.mode = mode
        level = "INFEASIBLE (proved)" if exc.status_code == GRB.INFEASIBLE else "UNDECIDED (no incumbent)"
        print(f"  -> {mode}: {level}, GRB status {exc.status_code} ({exc})")
        return model, None


def run_q2_3step_decomposition(community_name: str = "base", day: str = "d1",
                               time_limit: float = 420, big_m: float = 100.0,
                               mip_gap: float = 1e-4) -> None:
    """Question 2 mandatory 3-step pricing decomposition, validation and figures.

    ``time_limit`` is used for the BB+IR and soft-IR MILPs (the two expensive ones).
    """
    print("\n" + "=" * 70)
    print(f" Running Question 2 Pricing Models ({community_name}, day {day})")
    print("=" * 70)

    res_dir = RESULTS_DIR / community_name / "q2"
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load community data
    com = load_community(community_name, day=day)

    # 2. Question 1 benchmarks: C (direct control), R (no contract), I (individual contracts)
    print("\n[Step 0] Question 1 benchmarks (Model C / R / I)...")
    res_C = CommunityModel(com, setup="C").build().solve()
    res_R = CommunityModel(com, setup="R").build().solve()
    res_I = CommunityModel(com, setup="I").build().solve()
    benchmarks_R = res_R.member_costs["total"].to_dict()
    benchmarks_I = res_I.member_costs["total"].to_dict()
    print(f"  C = {res_C.objective:.4f} DKK | R = {res_R.objective:.4f} DKK | I = {res_I.objective:.4f} DKK")
    res_C.save(res_dir, tag=f"benchmark_C_{day}")
    res_R.save(res_dir, tag=f"benchmark_R_{day}")
    res_I.save(res_dir, tag=f"benchmark_I_{day}")

    # 3. Step 1: cap only
    print("\n[Step 1] Model M^P_cap (cap only)...")
    m_cap = DynamicPricingModel(com, mode="cap_only", big_m=big_m, mip_gap=mip_gap,
                                time_limit=time_limit).build()
    res_cap = m_cap.solve()
    res_cap.save(res_dir, tag=f"step1_cap_only_{day}")
    print(f"  -> cost = {res_cap.objective:.4f} DKK | welfare recovery = "
          f"{calc_welfare_recovery_ratio(res_cap.objective, res_C.objective, res_R.objective):.2f}%")
    print(f"  -> payments collected = {res_cap.meta['objective_terms']['total_payments_collected']:.4f} DKK"
          f" | DSO cost = {res_cap.meta['objective_terms']['dso_cost']:.4f} DKK")
    _print_checks("cap_only", res_cap)

    # 4. Step 2: + budget balance
    print("\n[Step 2] Model M^P_BB (budget balance)...")
    m_bb, res_bb = _build_and_solve(com, "BB", benchmarks_R, time_limit, big_m=big_m, mip_gap=mip_gap)
    if res_bb is None:
        raise RuntimeError("Model BB found no feasible solution: check the log.")
    res_bb.save(res_dir, tag=f"step2_BB_{day}")
    print(f"  -> cost = {res_bb.objective:.4f} DKK | welfare recovery = "
          f"{calc_welfare_recovery_ratio(res_bb.objective, res_C.objective, res_R.objective):.2f}%")
    print(f"  -> payments collected = {res_bb.meta['objective_terms']['total_payments_collected']:.4f} DKK"
          f" | DSO cost = {res_bb.meta['objective_terms']['dso_cost']:.4f} DKK"
          f" | budget residual = {calc_budget_balance_summary(res_bb)['net_manager_profit_DKK']:.4f} DKK")
    _print_checks("BB", res_bb)

    # 5. Step 3: + individual rationality (hard), then the soft relaxation
    print("\n[Step 3] Model M^P_BB+IR (budget balance + individual rationality)...")
    m_ir, res_ir = _build_and_solve(com, "BB_IR", benchmarks_R, time_limit,
                                    warm_from=m_bb, mip_focus=1, big_m=big_m, mip_gap=mip_gap)
    if res_ir is not None:
        res_ir.save(res_dir, tag=f"step3_BB_IR_{day}")
        print(f"  -> cost = {res_ir.objective:.4f} DKK (FEASIBLE) | welfare recovery = "
              f"{calc_welfare_recovery_ratio(res_ir.objective, res_C.objective, res_R.objective):.2f}%")
        _print_checks("BB_IR", res_ir)
    else:
        print("  -> BB+IR did not produce a solution under uniform prices.")

    print("\n[Step 3b] Model M^P_soft (BB + soft IR: minimum compensation for the blocking members)...")
    m_soft, res_soft = _build_and_solve(com, "soft_ir", benchmarks_R, time_limit,
                                        warm_from=m_bb, mip_focus=1, big_m=big_m, mip_gap=mip_gap)
    if res_soft is not None:
        res_soft.save(res_dir, tag=f"step3b_soft_ir_{day}")
        terms = res_soft.meta["objective_terms"]
        print(f"  -> community cost = {res_soft.objective:.4f} DKK"
              f" | efficiency cost vs cap-only = {calc_efficiency_cost(res_soft, res_cap):.4f} DKK")
        print(f"  -> total IR compensation sum(s_i) = {terms['total_ir_slack']:.4f} DKK"
              f" | solver objective = {terms['solver_objective']:.4f} DKK")
        slack = res_soft.member_costs["ir_slack"]
        blockers = slack[slack > 1e-4].sort_values(ascending=False)
        print(f"  -> {len(blockers)} member(s) keep a residual IR violation after the compensation:")
        for cid, value in blockers.items():
            print(f"     - {cid:8s} ({com.member(cid).archetype:10s}) needs {value:.4f} DKK")
        _print_checks("soft_ir", res_soft)

    # 6. Summary table
    results_dict: dict[str, object] = {"cap_only": res_cap, "BB": res_bb}
    if res_ir is not None:
        results_dict["BB_IR"] = res_ir
    if res_soft is not None:
        results_dict["soft_ir"] = res_soft
    df_summary = summarize_q2_experiments(results_dict, res_C, res_R)
    print("\n--- Summary comparison table ---")
    print(df_summary.to_string())
    df_summary.to_csv(res_dir / f"q2_3step_summary_{day}.csv")

    # 7. Validation pipeline
    print("\n--- Question 2 validation ---")
    val_results = run_q2_validation_pipeline(com, res_cap, res_C, m_cap)
    print("\nMember indifference analysis:")
    print(val_results["indifference_analysis"].to_string())
    val_results["indifference_analysis"].to_csv(res_dir / f"q2_indifference_analysis_{day}.csv")
    print("\nIR against the individual-contract benchmark (model I):")
    for name in results_dict:
        res = results_dict[name]
        costs = res.member_costs["total_cost"]
        viol = [(i, costs[i] - benchmarks_I[i]) for i in com.member_ids
                if costs[i] - benchmarks_I[i] > 1e-4]
        print(f"  {name:9s}: {len(viol)}/{len(com.member_ids)} members worse off than model I"
              + (f", worst {max(v for _, v in viol):.3f} DKK" if viol else ""))

    # 8. Figures
    print(f"\nGenerating figures under '{res_dir}' ...")
    plot_q2_posted_prices(res_cap, com, results_MC=res_C, save_to=res_dir / f"q2_fig1_posted_prices_{day}.png")
    plot_q2_3step_decomposition(df_summary.reset_index(), save_to=res_dir / f"q2_fig2_3step_decomposition_{day}.png")
    plot_q2_member_gains_and_blocking(
        res_bb.member_costs, title_suffix=f"Model BB ({day})", benchmark_costs=benchmarks_R,
        save_to=res_dir / f"q2_fig3_member_gains_{day}.png")
    if res_soft is not None:
        plot_q2_member_gains_and_blocking(
            res_soft.member_costs, title_suffix=f"Model soft-IR ({day})", benchmark_costs=benchmarks_R,
            save_to=res_dir / f"q2_fig3b_member_gains_soft_ir_{day}.png")
    posted_x = res_cap.hourly["posted_price"].to_numpy()
    df_indiff = analyze_indifference_risk(com, posted_x)
    plot_q2_indifference_risk(df_indiff, save_to=res_dir / f"q2_fig5_indifference_risk_{day}.png")

    m_cap.m.dispose()
    m_bb.m.dispose()
    m_ir.m.dispose()
    m_soft.m.dispose()
    print(f"[SUCCESS] Question 2 results under '{res_dir}'")


#-------------------q2 Experiments/Sensitivity---------------

def run_q2_sensitivity_sweep(community_name: str = "base", day: str = "d1",
                             time_limit: float = 60, big_m: float = 100.0,
                             mip_gap: float = 1e-4) -> None:
    """Contract-cap sensitivity study: cap_t = shipped cap_t * cap_factor.

    ``cap_factor = 1.0`` is the shipped cap (9.39 kW on d1), below 1 is tighter, above 1 looser.
    For every value the three Q2 steps are solved and the feasibility / efficiency cost / blocking
    members of the requirements are recorded.
    """
    print("\n" + "=" * 70)
    print(f" Running Question 2 contract-cap sensitivity sweep ({community_name}, day {day})")
    print("=" * 70)

    res_dir = RESULTS_DIR / community_name / "q2"
    res_dir.mkdir(parents=True, exist_ok=True)

    cap_factors = [0.6, 0.8, 1.0, 1.2, 1.4]
    base_cap = load_community(community_name, day=day).contract.capacity_limit_kW.copy()
    records = []
    bb_limit = max(time_limit, 120.0)   # cap_only and BB need a full solve: their objective is reported

    for factor in cap_factors:
        print(f"\nEvaluating cap factor {factor:.2f} (peak cap {base_cap.max() * factor:.2f} kW)...")

        com_mod = deepcopy(load_community(community_name, day=day))
        com_mod.contract.capacity_limit_kW = base_cap * factor

        res_C = CommunityModel(com_mod, setup="C").build().solve()
        res_R = CommunityModel(com_mod, setup="R").build().solve()
        benchmarks_R = res_R.member_costs["total"].to_dict()

        m_cap, res_cap = _build_and_solve(com_mod, "cap_only", None, bb_limit, big_m=big_m, mip_gap=mip_gap)
        m_bb, res_bb = _build_and_solve(com_mod, "BB", benchmarks_R, bb_limit, big_m=big_m, mip_gap=mip_gap)
        if res_cap is None or res_bb is None:
            print("  -> cap_only / BB found no solution at this cap: skipping the factor.")
            m_cap.m.dispose()
            m_bb.m.dispose()
            continue

        status_ir, cost_ir, slack_ir, n_block = "not solved", float("nan"), float("nan"), 0
        m_ir, res_ir = _build_and_solve(com_mod, "BB_IR", benchmarks_R, time_limit,
                                        warm_from=m_bb, mip_focus=1, big_m=big_m, mip_gap=mip_gap)
        if res_ir is not None:
            status_ir, cost_ir = "feasible", res_ir.objective
        m_soft, res_soft = _build_and_solve(com_mod, "soft_ir", benchmarks_R, time_limit,
                                            warm_from=m_bb, mip_focus=1, big_m=big_m, mip_gap=mip_gap)
        if res_soft is not None:
            slack_ir = float(res_soft.member_costs["ir_slack"].sum())
            n_block = int((res_soft.member_costs["ir_slack"] > 1e-4).sum())
            if status_ir != "feasible":
                status_ir = "soft only"
                cost_ir = res_soft.objective

        records.append(
            {
                "cap_factor": factor,
                "cap_peak_kW": round(float(base_cap.max() * factor), 3),
                "cost_MC_DKK": round(res_C.objective, 2),
                "cost_MR_DKK": round(res_R.objective, 2),
                "cost_cap_only_DKK": round(res_cap.objective, 2),
                "welfare_recovery_cap_pct": round(
                    calc_welfare_recovery_ratio(res_cap.objective, res_C.objective, res_R.objective), 2),
                "cost_BB_DKK": round(res_bb.objective, 2),
                "bb_gap": round(float(res_bb.meta.get("solver_gap", float("nan"))), 5),
                "bb_status": res_bb.status,
                "cost_BB_IR_DKK": round(cost_ir, 2),
                "bb_ir_status": status_ir,
                "is_BB_IR_feasible": status_ir == "feasible",
                "total_ir_slack_DKK": round(slack_ir, 4),
                "num_blocking_members": n_block,
                "efficiency_cost_BB_DKK": round(res_bb.objective - res_cap.objective, 2),
                "efficiency_cost_IR_DKK": round(cost_ir - res_bb.objective, 2) if cost_ir == cost_ir else float("nan"),
            }
        )
        m_cap.m.dispose()
        m_bb.m.dispose()
        m_ir.m.dispose()
        m_soft.m.dispose()

    df_sweep = pd.DataFrame(records)
    print("\n--- Sensitivity sweep results ---")
    print(df_sweep.to_string())
    df_sweep.to_csv(res_dir / f"q2_cap_sensitivity_sweep_{day}.csv", index=False)

    plot_q2_cap_sensitivity(df_sweep, save_to=res_dir / f"q2_fig4_cap_sensitivity_{day}.png")
    print(f"[SUCCESS] Cap sensitivity figure saved under '{res_dir}'")

#---------------q2 END----------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--q1-base", action="store_true", help="Question 1: R, I and C on the base community (day: --day)")
    ap.add_argument("--plot-data", action="store_true", help="plot the input data of every day d1-d7 (base community)")
    ap.add_argument("--q1-sensitivity", action="store_true", help="Question 1: sweeps of the discount beta and of the cap (d1)")
    ap.add_argument("--q2", action="store_true", help="run Question 2 dynamic pricing models & generate figures")
    ap.add_argument("--q2-sweep", action="store_true", help="run Question 2 contract-cap sensitivity sweep & generate figure")
    ap.add_argument("--q2-time-limit", type=float, default=420, help="time limit (s) of each BB+IR / soft-IR MILP")
    ap.add_argument("--sweep-time-limit", type=float, default=60, help="time limit (s) of each MILP in the sweep")
    ap.add_argument("--big-m", type=float, default=100.0,
                    help="Big-M of the KKT complementarities (DKK/kWh); it must exceed every dual variable")
    ap.add_argument("--mip-gap", type=float, default=1e-4, help="relative MIP gap of the pricing models")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--verify", action="store_true", help="run the Question 1 validation checks")
    args = ap.parse_args()

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")

    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)

    if args.q1_base:
        run_q1_base(RESULTS_DIR / "q1_base", day=args.day)

    if args.plot_data:
        for day in DAYS:
            plot_day_data(load_community("base", day=day), save_to=RESULTS_DIR / "data" / f"data_{day}.png")

    if args.q1_sensitivity:
        run_q1_sensitivity(RESULTS_DIR / "q1_sensitivity")

    if args.q2:
        run_q2_3step_decomposition(args.community, args.day, time_limit=args.q2_time_limit,
                                   big_m=args.big_m, mip_gap=args.mip_gap)

    if args.q2_sweep:
        run_q2_sensitivity_sweep(args.community, args.day, time_limit=args.sweep_time_limit,
                                 big_m=args.big_m, mip_gap=args.mip_gap)

    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")

    if args.verify and not run_q1_checks():
        sys.exit(1)


if __name__ == "__main__":
    main()
