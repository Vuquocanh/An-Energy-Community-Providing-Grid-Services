"""Metrics for Question 2.

Everything is defined on the ``Results`` objects returned by ``DynamicPricingModel`` and
``CommunityModel``, so that the summary table, the figures and the report use the same numbers.

Conventions
-----------
* Costs are DKK, minimised (welfare = -cost).
* "Member cost" is what member i pays under the posted prices plus its own disutility / PV /
  terminal-battery cost: ``total_cost`` in ``member_costs``.
* The reference for welfare is model C (direct control); the reference for individual rationality
  is model R (no contract), with model I (individual contracts) as a second benchmark.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .model import Results


def calc_welfare_recovery_ratio(cost_MP: float, cost_MC: float, cost_MR: float) -> float:
    """Welfare recovered by a pricing model, in % of what direct control adds over no contract.

    ``eta = (cost_R - cost_MP) / (cost_R - cost_C) * 100``; 100 % means the pricing model reaches
    the welfare of model C, 0 % means it is no better than staying alone (model R).
    """
    denom = cost_MR - cost_MC
    if abs(denom) < 1e-6:
        return 100.0
    return ((cost_MR - cost_MP) / denom) * 100.0


def calc_efficiency_cost(results_variant: Results, results_reference: Results) -> float:
    """Welfare lost by imposing a requirement: cost(variant) - cost(reference), DKK.

    Use it with the previous step as the reference (BB against cap-only, IR against BB).
    """
    return float(results_variant.objective - results_reference.objective)


def calc_budget_balance_summary(results_pricing: Results) -> dict[str, float]:
    """Manager's budget: payments collected vs the DSO cost at the connection point."""
    meta = results_pricing.meta.get("objective_terms", {})
    payments_collected = float(meta.get("total_payments_collected", 0.0))
    dso_cost = float(meta.get("dso_cost", 0.0))
    return {
        "payments_collected_DKK": payments_collected,
        "dso_cost_DKK": dso_cost,
        "net_manager_profit_DKK": payments_collected - dso_cost,
        "budget_balance_ratio": payments_collected / dso_cost if abs(dso_cost) > 1e-6 else 1.0,
        "gross_payments_DKK": float(meta.get("gross_payments_collected", payments_collected)),
        "transfers_DKK": float(meta.get("total_transfers", 0.0)),
        "ir_slack_DKK": float(meta.get("total_ir_slack", 0.0)),
    }


def calc_individual_gains(results_pricing: Results, results_benchmark: Results) -> pd.DataFrame:
    """Member-by-member gain against a benchmark (model R or model I)."""
    if results_pricing.member_costs is None or results_benchmark.member_costs is None:
        raise ValueError("Member costs table is missing from one of the results objects.")

    p_costs = results_pricing.member_costs
    df = pd.DataFrame(index=p_costs.index)
    df["archetype"] = p_costs["archetype"]
    df["cost_pricing"] = p_costs["total_cost"]
    df["cost_benchmark"] = results_benchmark.member_costs["total"]
    df["absolute_gain_DKK"] = df["cost_benchmark"] - df["cost_pricing"]
    df["relative_gain_pct"] = 100.0 * df["absolute_gain_DKK"] / df["cost_benchmark"]
    df["is_individually_rational"] = df["absolute_gain_DKK"] >= -1e-4
    return df


def identify_blocking_members(results: Results, threshold: float = 1e-4) -> pd.DataFrame:
    """Members that prevent hard IR: their required compensation (> threshold), by archetype.

    Works for every relaxation: the soft-IR slack ``ir_slack`` (soft_ir), the lump-sum transfer
    ``delta_transfer`` (extended), or the IR violation of a run without any relaxation.
    """
    if results.member_costs is None:
        raise ValueError("Member costs table is missing from results.")

    df = results.member_costs.copy()
    if "ir_slack" in df.columns and float(df["ir_slack"].abs().sum()) > threshold:
        df["required_compensation_DKK"] = df["ir_slack"]
    elif "delta_transfer" in df.columns and float(df["delta_transfer"].abs().sum()) > threshold:
        df["required_compensation_DKK"] = df["delta_transfer"]
    elif "ir_violation_or_gain" in df.columns:
        df["required_compensation_DKK"] = df["ir_violation_or_gain"].clip(lower=0.0)
    else:
        raise KeyError("No ir_slack / delta_transfer / benchmark column in member_costs.")

    df["is_blocking"] = df["required_compensation_DKK"] > threshold
    cols = ["archetype", "total_cost", "required_compensation_DKK", "is_blocking"]
    if "benchmark_cost" in df.columns:
        cols.insert(2, "benchmark_cost")
    return df[cols]


def analyze_price_signals(results_pricing: Results, com_data: Any) -> pd.DataFrame:
    """Hourly properties of the posted prices against the market benchmarks.

    Flags the hours where the price is pinned to the discounted import price, to the discounted
    export price, or where the community import sits at the cap (the cap sets the price).
    """
    df = pd.DataFrame(index=results_pricing.hourly.index)
    df["day_ahead_price"] = com_data.energy_price
    df["posted_price"] = results_pricing.hourly["posted_price"]
    df["price_spread"] = df["posted_price"] - df["day_ahead_price"]

    beta = com_data.contract.tariff_discount_ratio
    tau_imp = com_data.import_tariff
    tau_exp = com_data.export_tariff

    df["discounted_import_price"] = df["day_ahead_price"] + (1.0 - beta) * tau_imp
    df["discounted_export_price"] = df["day_ahead_price"] - (1.0 - beta) * tau_exp

    tol = 1e-6
    df["pinned_to_import"] = (df["posted_price"] - df["discounted_import_price"]).abs() <= tol
    df["pinned_to_export"] = (df["posted_price"] - df["discounted_export_price"]).abs() <= tol
    df["is_at_cap"] = results_pricing.hourly["exceedance"] > 1e-4
    df["is_cap_binding"] = results_pricing.hourly["import"] >= results_pricing.hourly["cap"] - 1e-3
    return df


def summarize_q2_experiments(
    results_dict: dict[str, Results], results_C: Results, results_R: Results
) -> pd.DataFrame:
    """One row per pricing variant: welfare, manager budget, IR, flexibility and requirement cost."""
    records = []
    reference = next(iter(results_dict.values()), None)

    for name, res in results_dict.items():
        welfare_rec = calc_welfare_recovery_ratio(res.objective, results_C.objective, results_R.objective)
        bb_summary = calc_budget_balance_summary(res)
        gains = calc_individual_gains(res, results_R)
        ir_violations = int((~gains["is_individually_rational"]).sum())
        terms = res.meta.get("objective_terms", {})

        records.append(
            {
                "Variant": name,
                "Status": res.status,
                "Objective_Cost_DKK": round(res.objective, 2),
                "Welfare_Recovery_Pct": round(welfare_rec, 2),
                "Efficiency_Cost_DKK": round(calc_efficiency_cost(res, reference), 4) if reference is not None else 0.0,
                "Manager_Revenues_DKK": round(bb_summary["payments_collected_DKK"], 2),
                "DSO_Costs_DKK": round(bb_summary["dso_cost_DKK"], 2),
                "Budget_Residual_DKK": round(bb_summary["net_manager_profit_DKK"], 2),
                "IR_Violations_Count": ir_violations,
                "IR_Slack_DKK": round(float(terms.get("total_ir_slack", 0.0)), 4),
                "Peak_Import_kW": round(res.hourly["import"].max(), 2),
                "Max_Cap_Exceedance_kW": round(res.hourly["exceedance"].max(), 2),
            }
        )

    return pd.DataFrame(records).set_index("Variant")
