"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --q2                              # Run Question 2 3-step dynamic pricing & generate figures
    python main.py --q2-sweep                        # Run Question 2 contract capacity sensitivity sweep & plot
    python main.py --q2 --day d2                     # Run Question 2 experiments on day d2
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)

No model is provided: build your Assignment 1 model class on the ``InputData`` of each member, then the community
models of the assignment (R, I, C, DP, ...) on top of it, one function per experiment, writing to ``results/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from copy import deepcopy
import pandas as pd
from src.metrics import (
    calc_budget_balance_summary,
    calc_welfare_recovery_ratio,
    identify_blocking_members,
    summarize_q2_experiments,
)
from src.model import CommunityModel
from src.model_q2_pricing import DynamicPricingModel
from src.plotting import (
    plot_q2_3step_decomposition,
    plot_q2_cap_sensitivity,
    plot_q2_indifference_risk,
    plot_q2_member_gains_and_blocking,
    plot_q2_posted_prices,
)
from src.data_loader import load_community, list_communities
from src.validation import analyze_indifference_risk, run_q2_validation_pipeline
RESULTS_DIR = Path(__file__).resolve().parent / "results"

#-----------q2----------------

def run_q2_3step_decomposition(community_name: str = "base", day: str = "d1") -> None:
    """Execute Question 2 mandatory 3-step pricing model decomposition, validation, and figure generation."""
    print("\n" + "=" * 70)
    print(f" Running Question 2 Pricing Models ({community_name}, day {day})")
    print("=" * 70)

    res_dir = RESULTS_DIR / community_name / "q2"
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load community data
    com = load_community(community_name, day=day)

    # 2. Solve Question 1 benchmark models (Model C and Model R)
    print("\n[Step 0] Solving Question 1 Benchmarks (Model C & Model R)...")
    res_C = CommunityModel(com, setup="C").build().solve()
    res_R = CommunityModel(com, setup="R").build().solve()
    benchmarks_R = res_R.member_costs["total"].to_dict()

    print(f"  -> Model C Welfare Cost (Min Benchmark): {res_C.objective:.4f} DKK")
    print(f"  -> Model R Total Cost   (No Contract)  : {res_R.objective:.4f} DKK")

    res_C.save(res_dir, tag=f"benchmark_C_{day}")
    res_R.save(res_dir, tag=f"benchmark_R_{day}")

    # 3. Step 1: Model M^P_cap (Cap-only)
    print("\n[Step 1] Solving Model M^P_cap (Cap-only)...")
    m_cap = DynamicPricingModel(com, mode="cap_only").build()
    res_cap = m_cap.solve()
    res_cap.save(res_dir, tag=f"step1_cap_only_{day}")

    rec_cap = calc_welfare_recovery_ratio(res_cap.objective, res_C.objective, res_R.objective)
    print(f"  -> Model M^P_cap Cost : {res_cap.objective:.4f} DKK")
    print(f"  -> Welfare Recovery   : {rec_cap:.2f}%")

    # 4. Step 2: Model M^P_BB (+ Budget Balance)
    print("\n[Step 2] Solving Model M^P_BB (+ Budget Balance)...")
    m_bb = DynamicPricingModel(com, mode="BB", benchmark_costs=benchmarks_R).build()
    res_bb = m_bb.solve()
    res_bb.save(res_dir, tag=f"step2_BB_{day}")

    rec_bb = calc_welfare_recovery_ratio(res_bb.objective, res_C.objective, res_R.objective)
    bb_summary = calc_budget_balance_summary(res_bb)
    print(f"  -> Model M^P_BB Cost  : {res_bb.objective:.4f} DKK")
    print(f"  -> Welfare Recovery   : {rec_bb:.2f}%")
    print(f"  -> Budget Residual    : {bb_summary['net_manager_profit_DKK']:.4f} DKK")

    # 5. Step 3: Model M^P_BB+IR (+ Individual Rationality)
    print("\n[Step 3] Solving Model M^P_BB+IR (+ Individual Rationality)...")
    try:
        m_ir = DynamicPricingModel(com, mode="BB_IR", benchmark_costs=benchmarks_R).build()
        res_ir = m_ir.solve()
        res_ir.save(res_dir, tag=f"step3_BB_IR_{day}")
        rec_ir = calc_welfare_recovery_ratio(res_ir.objective, res_C.objective, res_R.objective)
        print(f"  -> Model M^P_BB+IR Cost: {res_ir.objective:.4f} DKK (FEASIBLE)")
        print(f"  -> Welfare Recovery    : {rec_ir:.2f}%")
        results_dict = {"cap_only": res_cap, "BB": res_bb, "BB_IR": res_ir}
    except Exception as e:
        print(f"  -> Model M^P_BB+IR is INFEASIBLE under uniform pricing!")
        print(f"     Solver Message: {e}")

        # Step 4: Extended Model (Lump-sum transfers delta_i)
        print("\n[Step 4] Solving Extended Model to Identify Blocking Members...")
        m_ext = DynamicPricingModel(com, mode="extended", benchmark_costs=benchmarks_R).build()
        res_ext = m_ext.solve()
        res_ext.save(res_dir, tag=f"step4_extended_{day}")

        df_blocking = identify_blocking_members(res_ext)
        print("\n--- Member Cost Breakdown & Required Transfers (delta_i) ---")
        print(df_blocking.to_string())

        blocking_only = df_blocking[df_blocking["is_blocking"]]
        print(f"\nIdentified {len(blocking_only)} Blocking Members:")
        for cid, row in blocking_only.iterrows():
            print(f"  - Member {cid} ({row['archetype']}): requires {row['required_compensation_DKK']:.4f} DKK compensation")

        results_dict = {"cap_only": res_cap, "BB": res_bb, "extended": res_ext}

    # 6. Export summary table
    df_summary = summarize_q2_experiments(results_dict, res_C, res_R)
    print("\n--- Summary Comparison Table ---")
    print(df_summary.to_string())
    df_summary.to_csv(res_dir / f"q2_3step_summary_{day}.csv")

    # 7. Run full validation pipeline
    print("\n--- Validation Pipeline ---")
    val_results = run_q2_validation_pipeline(com, res_cap, res_C, m_cap)
    print("\nMember Indifference Analysis:")
    print(val_results["indifference_analysis"].to_string())

    # 8. Generate and save Question 2 figures
    print(f"\nGenerating and saving Question 2 figures under '{res_dir}'...")

    # Figure 1: Posted price signals vs day-ahead price & Model C duals
    plot_q2_posted_prices(res_cap, com, results_MC=res_C, save_to=res_dir / f"q2_fig1_posted_prices_{day}.png")

    # Figure 2: Three-step requirements decomposition
    plot_q2_3step_decomposition(df_summary.reset_index(), save_to=res_dir / f"q2_fig2_3step_decomposition_{day}.png")

    # Figure 3: Member-by-member Individual Rationality & blocking members
    plot_q2_member_gains_and_blocking(
        res_bb.member_costs,
        title_suffix=f"Model BB ({day})",
        benchmark_costs=benchmarks_R,
        save_to=res_dir / f"q2_fig3_member_gains_{day}.png",
    )

    # Figure 5: Consumer indifference risk analysis
    posted_x = res_cap.hourly["posted_price"].to_numpy()
    df_indiff = analyze_indifference_risk(com, posted_x)
    plot_q2_indifference_risk(df_indiff, save_to=res_dir / f"q2_fig5_indifference_risk_{day}.png")

    print(f"[SUCCESS] All Question 2 base figures saved under '{res_dir}'")

#-------------------q2 Experiments/Sensitivity---------------

def run_q2_sensitivity_sweep(community_name: str = "base", day: str = "d1") -> None:
    """Perform a contract tightness sensitivity sweep over scaling factor kappa and generate plot."""
    print("\n" + "=" * 70)
    print(f" Running Question 2 Contract Sensitivity Sweep ({community_name}, day {day})")
    print("=" * 70)

    res_dir = RESULTS_DIR / community_name / "q2"
    res_dir.mkdir(parents=True, exist_ok=True)

    cap_factors = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
    records = []

    for factor in cap_factors:
        print(f"\nEvaluating Cap Tightness Factor kappa = {factor:.2f}...")

        com_mod = deepcopy(load_community(community_name, day=day))
        com_mod.contract.capacity_limit_kW = com_mod.contract.capacity_limit_kW * (factor / 1.3)

        # Benchmarks
        res_C = CommunityModel(com_mod, setup="C").build().solve()
        res_R = CommunityModel(com_mod, setup="R").build().solve()
        benchmarks_R = res_R.member_costs["total"].to_dict()

        # Pricing Models
        res_cap = DynamicPricingModel(com_mod, mode="cap_only").build().solve()
        rec_cap = calc_welfare_recovery_ratio(res_cap.objective, res_C.objective, res_R.objective)
        res_bb = DynamicPricingModel(com_mod, mode="BB").build().solve()

        is_ir_feasible = True
        total_transfer = 0.0
        num_blocking = 0

        try:
            res_ir = DynamicPricingModel(com_mod, mode="BB_IR", benchmark_costs=benchmarks_R).build().solve()
            cost_ir = res_ir.objective
        except Exception:
            is_ir_feasible = False
            try:
                res_ext = DynamicPricingModel(com_mod, mode="extended", benchmark_costs=benchmarks_R).build().solve()
                cost_ir = res_ext.objective
                df_blk = identify_blocking_members(res_ext)
                total_transfer = df_blk[df_blk["required_compensation_DKK"] > 0]["required_compensation_DKK"].sum()
                num_blocking = int(df_blk["is_blocking"].sum())
            except Exception:
                cost_ir = res_cap.objective
                total_transfer = float("nan")
                num_blocking = len(com_mod.members)

        records.append(
            {
                "cap_factor": factor,
                "cap_peak_kW": com_mod.contract.capacity_limit_kW.max(),
                "cost_MC_DKK": round(res_C.objective, 2),
                "cost_MR_DKK": round(res_R.objective, 2),
                "cost_cap_only_DKK": round(res_cap.objective, 2),
                "welfare_recovery_cap_pct": round(rec_cap, 2),
                "cost_BB_DKK": round(res_bb.objective, 2),
                "is_BB_IR_feasible": is_ir_feasible,
                "cost_extended_DKK": round(cost_ir, 2),
                "total_transfer_delta_DKK": round(total_transfer, 4),
                "num_blocking_members": num_blocking,
            }
        )

    df_sweep = pd.DataFrame(records)
    print("\n--- Sensitivity Sweep Results ---")
    print(df_sweep.to_string())
    df_sweep.to_csv(res_dir / f"q2_cap_sensitivity_sweep_{day}.csv", index=False)

    # Generate Figure 4: Cap sensitivity plot
    plot_q2_cap_sensitivity(df_sweep, save_to=res_dir / f"q2_fig4_cap_sensitivity_{day}.png")
    print(f"[SUCCESS] Cap sensitivity figure saved under '{res_dir}'")

#---------------q2 END----------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--q2", action="store_true", help="run Question 2 dynamic pricing models & generate figures")
    ap.add_argument("--q2-sweep", action="store_true", help="run Question 2 contract capacity sensitivity sweep & generate figure")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    
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

    if args.q2:
        run_q2_3step_decomposition(args.community, args.day)

    if args.q2_sweep:
        run_q2_sensitivity_sweep(args.community, args.day)

    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")


if __name__ == "__main__":
    main()
