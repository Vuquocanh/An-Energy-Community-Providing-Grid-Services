"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)
    python main.py --verify                          # run the Question 1 validation checks (src/validation.py)
    python main.py --q1-base                         # Question 1: R, I and C on the base case (results/q1_base/)
    python main.py --plot-data                       # input data of d1-d7: prices, reference load, PV, cap (results/data/)
    python main.py --q1-sensitivity                  # Question 1: sweeps of the discount and of the cap (results/q1_sensitivity/)

No model is provided: build your Assignment 1 model class on the ``InputData`` of each member, then the community
models of the assignment (R, I, C, DP, ...) on top of it, one function per experiment, writing to ``results/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data_loader import DAYS, load_community, list_communities
from src.plotting import plot_day_data
from src.q1 import run_q1_base, run_q1_sensitivity
from src.validation import run_q1_checks

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--verify", action="store_true", help="run the Question 1 validation checks")
    ap.add_argument("--q1-base", action="store_true", help="Question 1: R, I and C on the base community (day: --day)")
    ap.add_argument("--plot-data", action="store_true", help="plot the input data of every day d1-d7 (base community)")
    ap.add_argument("--q1-sensitivity", action="store_true", help="Question 1: sweeps of the discount beta and of the cap (d1)")
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
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")
    if args.verify and not run_q1_checks():
        sys.exit(1)
    if args.q1_base:
        run_q1_base(RESULTS_DIR / "q1_base", day=args.day)
    if args.plot_data:
        for day in DAYS:
            plot_day_data(load_community("base", day=day), save_to=RESULTS_DIR / "data" / f"data_{day}.png")
    if args.q1_sensitivity:
        run_q1_sensitivity(RESULTS_DIR / "q1_sensitivity")

if __name__ == "__main__":
    main()
