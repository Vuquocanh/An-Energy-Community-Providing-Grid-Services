"""Metrics calculation and evaluation utilities for Question 2.

Provides analytical tools to quantify welfare recovery efficiency, identify blocking
members under uniform pricing, analyze posted price properties, and verify budget balance
and individual rationality.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .model import Results


def calc_welfare_recovery_ratio(
    cost_MP: float, cost_MC: float, cost_MR: float
) -> float:
    """Calculate the welfare recovery ratio (%) of a pricing model against direct control.

    Note: In cost minimization formulation, Welfare = -Cost.
    Formula: eta = (Cost(M^R) - Cost(M^P)) / (Cost(M^R) - Cost(M^C)) * 100%

    Parameters
    ----------
    cost_MP : float
        Objective cost of the dynamic pricing model (M^P).
    cost_MC : float
        Objective cost of the central dispatch benchmark (M^C).
    cost_MR : float
        Objective cost of the no-contract reference baseline (M^R).

    Returns
    -------
    float
        Welfare recovery ratio in percentage (100% means full recovery of M^C welfare).
    """
    denom = cost_MR - cost_MC
    if abs(denom) < 1e-6:
        return 100.0
    return ((cost_MR - cost_MP) / denom) * 100.0


def identify_blocking_members(
    results_extended: Results, threshold: float = 1e-4
) -> pd.DataFrame:
    """Identify members blocking uniform price feasibility under IR and BB constraints.

    Reads the lump-sum transfers delta_i required in the extended model to satisfy IR.

    Parameters
    ----------
    results_extended : Results
        Results object returned by DynamicPricingModel in 'extended' mode.
    threshold : float, default 1e-4
        Numerical threshold to consider a transfer positive (blocking).

    Returns
    -------
    pd.DataFrame
        DataFrame containing archetype, total cost, benchmark cost, required transfer (delta_i),
        and blocking status for each member.
    """
    if results_extended.member_costs is None:
        raise ValueError("Member costs table is missing from results.")

    df = results_extended.member_costs.copy()
    if "delta_transfer" not in df.columns:
        raise KeyError("Column 'delta_transfer' not found. Run model in 'extended' mode.")

    df["is_blocking"] = df["delta_transfer"] > threshold
    df["required_compensation_DKK"] = df["delta_transfer"]

    return df[["archetype", "total_cost", "benchmark_cost", "required_compensation_DKK", "is_blocking"]]


def calc_budget_balance_summary(results_pricing: Results) -> dict[str, float]:
    """Analyze the community manager's financial balance (revenues vs DSO costs).

    Parameters
    ----------
    results_pricing : Results
        Results object from DynamicPricingModel.

    Returns
    -------
    dict[str, float]
        Dictionary containing total payments collected, DSO grid costs, net balance,
        and budget balance ratio.
    """
    meta = results_pricing.meta.get("objective_terms", {})
    payments_collected = meta.get("total_payments_collected", 0.0)
    dso_cost = meta.get("dso_cost", 0.0)
    net_balance = payments_collected - dso_cost

    return {
        "payments_collected_DKK": payments_collected,
        "dso_cost_DKK": dso_cost,
        "net_manager_profit_DKK": net_balance,
        "budget_balance_ratio": payments_collected / dso_cost if abs(dso_cost) > 1e-6 else 1.0,
    }


def analyze_price_signals(results_pricing: Results, com_data: Any) -> pd.DataFrame:
    """Analyze posted internal price properties relative to day-ahead prices and tariffs.

    Parameters
    ----------
    results_pricing : Results
        Results object from DynamicPricingModel.
    com_data : CommunityData
        Community data containing contract parameters and base prices.

    Returns
    -------
    pd.DataFrame
        Hourly summary with posted price, day-ahead price, price spread, and boundary flags.
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

    # Check if price is pinned to boundary or driven by flexibility merit order
    df["is_at_cap"] = results_pricing.hourly["exceedance"] > 1e-4

    return df


def calc_individual_gains(
    results_pricing: Results, results_benchmark: Results
) -> pd.DataFrame:
    """Calculate cost savings and individual rationality gains for each community member.

    Parameters
    ----------
    results_pricing : Results
        Results object from DynamicPricingModel.
    results_benchmark : Results
        Results object from reference benchmark model (e.g., Model R or Model I).

    Returns
    -------
    pd.DataFrame
        DataFrame with archetype, benchmark cost, pricing cost, absolute gain, and relative gain (%).
    """
    if results_pricing.member_costs is None or results_benchmark.member_costs is None:
        raise ValueError("Member costs table is missing from one of the results objects.")

    df = pd.DataFrame(index=results_pricing.member_costs.index)
    df["archetype"] = results_pricing.member_costs["archetype"]

    # Compute total member cost under pricing model: disutility + pv_cost + payment
    p_costs = results_pricing.member_costs
    df["cost_pricing"] = p_costs["disutility"] + p_costs["pv_cost"] + p_costs["payment"]

    # Benchmark total cost (e.g., from Model R)
    b_costs = results_benchmark.member_costs
    df["cost_benchmark"] = b_costs["total"] if "total" in b_costs else b_costs["total_cost"]

    df["absolute_gain_DKK"] = df["cost_benchmark"] - df["cost_pricing"]
    df["relative_gain_pct"] = (df["absolute_gain_DKK"] / df["cost_benchmark"]) * 100.0
    df["is_individually_rational"] = df["absolute_gain_DKK"] >= -1e-4

    return df


def summarize_q2_experiments(
    results_dict: dict[str, Results], results_C: Results, results_R: Results
) -> pd.DataFrame:
    """Generate a comprehensive metrics table comparing all Question 2 model variants.

    Parameters
    ----------
    results_dict : dict[str, Results]
        Mapping of variant name (e.g., 'cap_only', 'BB', 'BB_IR', 'extended') to Results.
    results_C : Results
        Centralized model benchmark.
    results_R : Results
        No-contract reference model benchmark.

    Returns
    -------
    pd.DataFrame
        Comparison summary table formatted for inclusion in report tables.
    """
    records = []

    for name, res in results_dict.items():
        welfare_rec = calc_welfare_recovery_ratio(res.objective, results_C.objective, results_R.objective)
        bb_summary = calc_budget_balance_summary(res)

        # Count members violating IR against Model R
        gains = calc_individual_gains(res, results_R)
        ir_violations = sum(~gains["is_individually_rational"])

        records.append(
            {
                "Variant": name,
                "Status": res.status,
                "Objective_Cost_DKK": round(res.objective, 2),
                "Welfare_Recovery_Pct": round(welfare_rec, 2),
                "Manager_Revenues_DKK": round(bb_summary["payments_collected_DKK"], 2),
                "DSO_Costs_DKK": round(bb_summary["dso_cost_DKK"], 2),
                "Budget_Residual_DKK": round(bb_summary["net_manager_profit_DKK"], 2),
                "IR_Violations_Count": ir_violations,
                "Peak_Import_kW": round(res.hourly["import"].max(), 2),
                "Max_Cap_Exceedance_kW": round(res.hourly["exceedance"].max(), 2),
            }
        )

    return pd.DataFrame(records).set_index("Variant")
