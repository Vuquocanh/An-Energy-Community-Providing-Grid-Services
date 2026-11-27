"""Question 3 -- stochastic dynamic pricing: one price vector, N scenarios of the members' preferences.

``SDPModel`` is ``DynamicPricingModel`` (Question 2) with a scenario index on the second stage:

* first stage: the posted prices x_t, shared by every scenario (price bounds and the lexicographic price rule of DP);
* second stage, one copy per scenario s with probability pi_s: every member's KKT system under its realised
  reference profile / daily minimum / bounds (``q3_scenarios``), the meter P^imp_s, P^exp_s, E_s, the DSO bill, the
  payments (strong duality, as in DP);
* objective: the expected community cost  sum_s pi_s [DSO_s + sum_i J_{i,s}]  (``SDP^E``), optionally mixed with the
  CVaR of the scenario costs, (1 - gamma) E + gamma CVaR_beta  (``SDP^CVaR``);
* requirements in expectation: budget balance  E[payments] = E[DSO],  individual rationality per member
  E[cost_i] <= E[B_i]  (hard, or soft with one slack per member on the expectation), B_{i,s} = min(R, I) under theta_s;
* chance-constrained cap (``SDP^CC``): one binary z_s per scenario (joint over the hours), P^imp_{t,s} - cap_t <= M_t z_s,
  sum_s pi_s z_s <= eps; or one z_{t,s} per hour with the probability bound per hour (``cc_joint=False``).

The member KKT blocks are built by the parent's ``_add_lower_level_member_kkt`` on a copy of each scenario member whose
consumer_id is the composite ``"<id>__s<k>"``; every composite id maps to the same price variables, so the parent's
price-selection stage and big-M bookkeeping work unchanged. Modes: 'cap_only', 'BB', 'BB_IR', 'soft_ir'.

    scenario_benchmarks(scenarios)            -> {composite id: min(cost_R, cost_I)} (2 LPs per scenario)
    SDPModel(com, scenarios, mode, ...).build().solve()   -> SDPResults

Validation: with the single expected scenario (theta = 1) the model is DP and must reproduce its objective and prices.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from .data_loader import CommunityData
from .model import CommunityModel, _status_name
from .model_q2_pricing import DynamicPricingModel, NoPricingSolution
from .q3_evaluate import member_benchmarks
from .q3_scenarios import Scenario

SEP = "__s"


def composite_id(member_id: str, k: int) -> str:
    return f"{member_id}{SEP}{k}"


def split_id(cid: str) -> tuple[str, int]:
    i, k = cid.rsplit(SEP, 1)
    return i, int(k)


def scenario_benchmarks(scenarios: list[Scenario], end_soc_mode: str = "constraint") -> dict[str, float]:
    """Outside option of every member in every scenario, keyed by composite id (for BB_IR / soft_ir)."""
    out = {}
    for k, s in enumerate(scenarios):
        b = member_benchmarks(s, end_soc_mode)
        out.update({composite_id(i, k): v for i, v in b.items()})
    return out


@dataclass
class SDPResults:
    """Prices, expected cost and the per-scenario / per-member detail of one SDP solve."""
    status: str
    objective: float                  # expected community cost, DKK/day (the quantity the report compares)
    prices: np.ndarray                # posted prices x_t
    scenarios: pd.DataFrame           # one row per in-sample scenario: cost, DSO bill, exceedance, payments, budget, z
    members: pd.DataFrame             # one row per (member, scenario): payment, private cost, cost, benchmark
    hourly: pd.DataFrame              # price, expected import / export / exceedance and the cap
    meta: dict[str, Any] = field(default_factory=dict)


class SDPModel(DynamicPricingModel):
    """Stochastic version of the dynamic-pricing MILP: see the module docstring."""

    def __init__(
        self,
        com: CommunityData,
        scenarios: list[Scenario],
        mode: Literal["cap_only", "BB", "BB_IR", "soft_ir"] = "BB",
        benchmark_costs: dict[str, float] | None = None,
        cc_eps: float | None = None,
        cc_joint: bool = True,
        cvar_beta: float | None = None,
        cvar_gamma: float = 0.5,
        **kwargs,
    ):
        if mode not in ("cap_only", "BB", "BB_IR", "soft_ir"):
            raise ValueError(f"SDPModel supports cap_only / BB / BB_IR / soft_ir, not {mode!r}")
        if kwargs.get("allow_price_discrimination"):
            raise ValueError("price discrimination is not implemented in SDPModel")
        if not scenarios:
            raise ValueError("at least one scenario is needed")
        if mode in ("BB_IR", "soft_ir") and benchmark_costs is None:
            benchmark_costs = scenario_benchmarks(scenarios, kwargs.get("end_soc_mode", "constraint"))
        super().__init__(com, mode=mode, benchmark_costs=benchmark_costs or {}, **kwargs)
        self.scenarios = scenarios
        self.N = len(scenarios)
        self.prob = np.array([s.probability for s in scenarios], dtype=float)
        self.prob = self.prob / self.prob.sum()
        self.member_ids = com.member_ids
        self.ids = [composite_id(i, k) for k in range(self.N) for i in self.member_ids]   # used by the parent
        self.cc_eps = cc_eps
        self.cc_joint = bool(cc_joint)
        self.cvar_beta = cvar_beta
        self.cvar_gamma = float(cvar_gamma)
        if cc_eps is not None and not (0.0 <= cc_eps <= 1.0):
            raise ValueError("cc_eps must lie in [0, 1]")
        if cvar_beta is not None and not (0.0 < cvar_beta < 1.0 and 0.0 <= cvar_gamma <= 1.0):
            raise ValueError("cvar_beta in (0, 1) and cvar_gamma in [0, 1]")
        self.m.ModelName = f"sdp_{mode}_N{self.N}"
        self.scen_expr: dict[str, list] = {}   # per-scenario expressions: dso_cost, community_cost, payments

    # ------------------------------------------------------------------ build
    def build(self) -> "SDPModel":
        self._add_upper_level_variables()
        for k, s in enumerate(self.scenarios):
            for d in s.community.members:
                d_k = copy.copy(d)
                d_k.consumer_id = composite_id(d.consumer_id, k)
                self._add_lower_level_member_kkt(d_k)
        self._add_community_meter_and_dso_costs()
        self._add_pricing_requirements()
        self._add_chance_constraint()
        self._set_objective()
        self.m.update()
        return self

    def _add_upper_level_variables(self) -> None:
        m, T = self.m, self.T
        lo, hi = (-GRB.INFINITY, GRB.INFINITY) if self.price_bounds is None else self.price_bounds
        lo = np.broadcast_to(np.asarray(lo, dtype=float), (len(T),))
        hi = np.broadcast_to(np.asarray(hi, dtype=float), (len(T),))
        x_uni = m.addVars(T, lb=-GRB.INFINITY, name="x_price_uni")
        for t in T:
            x_uni[t].LB, x_uni[t].UB = lo[t], hi[t]
        self.var["x_price"] = {(cid, t): x_uni[t] for cid in self.ids for t in T}
        self.var["x_uni"] = x_uni
        if self.mode == "soft_ir":
            self.var["ir_slack"] = m.addVars(self.member_ids, lb=0.0, name="ir_slack")   # one per member, on the expectation

    def _add_community_meter_and_dso_costs(self) -> None:
        m, T, p = self.m, self.T, np.asarray(self.com.energy_price, dtype=float)
        c = self.com.contract
        f = 1.0 - c.tariff_discount_ratio
        tau_imp, tau_exp, alpha = self.com.import_tariff, self.com.export_tariff, c.penalty_DKK_per_kWh
        cap = np.asarray(c.capacity_limit_kW, dtype=float)
        imp = m.addVars(range(self.N), T, lb=0.0, name="imp_grid")
        exp = m.addVars(range(self.N), T, lb=0.0, name="exp_grid")
        exc = m.addVars(range(self.N), T, lb=0.0, name="exc_grid")
        self.var["import"], self.var["export"], self.var["exceedance"] = imp, exp, exc
        dso, comm, pays = [], [], []
        for k in range(self.N):
            ids_k = [composite_id(i, k) for i in self.member_ids]
            for t in T:
                m.addConstr(imp[k, t] - exp[k, t] == gp.quicksum(self.net[cid][t] for cid in ids_k), name=f"comm_bal_{k}_{t}")
                m.addConstr(exc[k, t] >= imp[k, t] - cap[t], name=f"comm_cap_{k}_{t}")
                if self.exact_exceedance:
                    z_cap = m.addVar(vtype=GRB.BINARY, name=f"cap_binary_{k}_{t}")
                    m.addConstr(exc[k, t] <= self.big_m * z_cap)
                    m.addConstr(exc[k, t] <= imp[k, t] - cap[t] + self.big_m * (1 - z_cap))
                    m.addSOS(GRB.SOS_TYPE1, [imp[k, t], exp[k, t]])
            dso_k = gp.quicksum(p[t] * (imp[k, t] - exp[k, t]) + f * tau_imp * (imp[k, t] - exc[k, t]) + tau_imp * exc[k, t]
                                + alpha * exc[k, t] + f * tau_exp * exp[k, t] for t in T)
            private_k = gp.quicksum(self.disutility[cid] + self.pv_cost[cid] + self.end_soc_value[cid] for cid in ids_k)
            dso.append(dso_k)
            comm.append(dso_k + private_k)
            pays.append(gp.quicksum(self.payment[cid] for cid in ids_k))
        self.scen_expr = {"dso_cost": dso, "community_cost": comm, "payments": pays}
        self.expr["dso_cost"] = gp.quicksum(self.prob[k] * dso[k] for k in range(self.N))              # expected
        self.expr["community_cost"] = gp.quicksum(self.prob[k] * comm[k] for k in range(self.N))       # expected
        self.expr["penalty"] = gp.quicksum(self.prob[k] * alpha * exc.sum(k, "*") for k in range(self.N))

    def _expected_member_cost(self, i: str) -> gp.LinExpr:
        return gp.quicksum(self.prob[k] * self._member_cost(composite_id(i, k)) for k in range(self.N))

    def _expected_benchmark(self, i: str) -> float:
        return float(sum(self.prob[k] * self.benchmark_costs[composite_id(i, k)] for k in range(self.N)))

    def _add_pricing_requirements(self) -> None:
        m = self.m
        if self.mode in ("BB", "BB_IR", "soft_ir"):
            total_payment = gp.quicksum(self.prob[k] * self.scen_expr["payments"][k] for k in range(self.N))
            m.addConstr(total_payment == self.expr["dso_cost"], name="req_budget_balance_expected")
        if self.mode == "BB_IR":
            for i in self.member_ids:
                m.addConstr(self._expected_member_cost(i) <= self._expected_benchmark(i), name=f"req_ir_{i}")
        elif self.mode == "soft_ir":
            s = self.var["ir_slack"]
            for i in self.member_ids:
                m.addConstr(self._expected_member_cost(i) - s[i] <= self._expected_benchmark(i), name=f"req_ir_soft_{i}")

    def _add_chance_constraint(self) -> None:
        """P(cap holds in every hour) >= 1 - eps as its scenario approximation (joint), or hour by hour."""
        if self.cc_eps is None:
            return
        m, T = self.m, self.T
        cap = np.asarray(self.com.contract.capacity_limit_kW, dtype=float)
        big = np.array([sum(d.load_max_kWh + (d.battery_max_charge_kW or 0.0) for d in self.com.members) - cap[t] for t in T])
        imp = self.var["import"]
        if self.cc_joint:
            z = m.addVars(range(self.N), vtype=GRB.BINARY, name="cc_violate")
            for k in range(self.N):
                for t in T:
                    m.addConstr(imp[k, t] - cap[t] <= big[t] * z[k], name=f"cc_{k}_{t}")
            m.addConstr(gp.quicksum(self.prob[k] * z[k] for k in range(self.N)) <= self.cc_eps, name="cc_budget")
        else:
            z = m.addVars(range(self.N), T, vtype=GRB.BINARY, name="cc_violate")
            for k in range(self.N):
                for t in T:
                    m.addConstr(imp[k, t] - cap[t] <= big[t] * z[k, t], name=f"cc_{k}_{t}")
            for t in T:
                m.addConstr(gp.quicksum(self.prob[k] * z[k, t] for k in range(self.N)) <= self.cc_eps, name=f"cc_budget_{t}")
        self.var["cc_violate"] = z

    def _set_objective(self) -> None:
        m = self.m
        objective = gp.LinExpr(self.expr["community_cost"])
        if self.cvar_beta is not None:
            eta = m.addVar(lb=-GRB.INFINITY, name="cvar_eta")
            u = m.addVars(range(self.N), lb=0.0, name="cvar_excess")
            for k in range(self.N):
                m.addConstr(u[k] >= self.scen_expr["community_cost"][k] - eta, name=f"cvar_{k}")
            cvar = eta + gp.quicksum(self.prob[k] * u[k] for k in range(self.N)) / (1.0 - self.cvar_beta)
            self.expr["cvar"] = cvar
            self.var["cvar_eta"], self.var["cvar_excess"] = eta, u
            objective = (1.0 - self.cvar_gamma) * self.expr["community_cost"] + self.cvar_gamma * cvar
        if self.mode == "soft_ir":
            objective = objective + self.ir_penalty * gp.quicksum(self.var["ir_slack"].values())
        m.setObjective(objective, GRB.MINIMIZE)


    def warm_start_from_prices(self, prices) -> None:
        """Partial MIP start: the posted prices (e.g. the DP prices at theta = 1); Gurobi completes the start by
        solving the members' responses under each scenario before branching. Call after build(), before solve()."""
        x = np.asarray(prices, dtype=float)
        for t in self.T:
            self.var["x_uni"][t].Start = float(x[t])
        self.m.update()

    # ------------------------------------------------------------------ validation
    def validate_solution(self, tolerance: float | None = None) -> dict[str, float]:
        """Independent checks per scenario: best response (every member re-solved at the posted prices under its
        scenario data), payment identity, meter, complementarity; then the requirements in expectation."""
        if tolerance is None:
            tolerance = max(1e-4, 10.0 * float(self.m.Params.FeasibilityTol) * self.big_m)
        T = self.T
        prices = [self.var["x_uni"][t].X for t in T]
        pay_err, resp_err, meter_err, exc_err, overlap = [], [], [], [], []
        exp_pay, exp_dso, exp_cost = 0.0, 0.0, {i: 0.0 for i in self.member_ids}
        cap = np.asarray(self.com.contract.capacity_limit_kW, dtype=float)
        for k, s in enumerate(self.scenarios):
            follower = CommunityModel(s.community, "C", end_soc_mode=self.end_soc_mode, final_soc_price=self.final_soc_price)
            try:
                for d in s.community.members:
                    follower._add_member(d)
                obj = {d.consumer_id: gp.quicksum(pm[d.consumer_id] for pm in follower.mexpr.values() if d.consumer_id in pm)
                       + gp.quicksum(prices[t] * follower.net[d.consumer_id][t] for t in T) for d in s.community.members}
                follower.m.setObjective(gp.quicksum(obj.values()), GRB.MINIMIZE)
                follower.m.Params.OutputFlag = 0
                follower.m.optimize()
                if follower.m.Status != GRB.OPTIMAL:
                    raise RuntimeError(f"independent follower LP failed in scenario {k}")
                for d in s.community.members:
                    cid = composite_id(d.consumer_id, k)
                    paid = sum(prices[t] * self.net[cid][t].getValue() for t in T)
                    private = self.disutility[cid].getValue() + self.pv_cost[cid].getValue() + self.end_soc_value[cid].getValue()
                    pay_err.append(abs(paid - self.payment[cid].getValue()))
                    resp_err.append(abs(paid + private - obj[d.consumer_id].getValue()))
                    exp_pay += self.prob[k] * paid
                    exp_cost[d.consumer_id] += self.prob[k] * (paid + private)
            finally:
                follower.m.dispose()
            exp_dso += self.prob[k] * self.scen_expr["dso_cost"][k].getValue()
            for t in T:
                imp, exp, exc = (self.var[n][k, t].X for n in ("import", "export", "exceedance"))
                meter_err.append(abs(imp - exp - sum(self.net[composite_id(i, k)][t].getValue() for i in self.member_ids)))
                exc_err.append(abs(exc - max(0.0, imp - cap[t])))
                overlap.append(min(imp, exp))
        checks = {
            "max_payment_identity_error": max(pay_err), "max_best_response_gap": max(resp_err),
            "max_complementarity_product": max(abs(a.X * b.X) for a, b in self.complementarity_pairs),
            "max_dual_over_bigM": max(abs(a.X) for a, _ in self.complementarity_pairs) / self.big_m,
            "max_meter_error": max(meter_err), "max_exceedance_error": max(exc_err), "max_simultaneous_grid_flow": max(overlap),
            "max_linear_constraint_violation": self.m.ConstrVio, "max_bound_violation": self.m.BoundVio, "tolerance": tolerance,
        }
        if self.mode != "cap_only":
            checks["expected_budget_error"] = abs(exp_pay - exp_dso)
        if self.mode in ("BB_IR", "soft_ir"):
            slack = {i: (self.var["ir_slack"][i].X if self.mode == "soft_ir" else 0.0) for i in self.member_ids}
            checks["max_expected_IR_violation"] = max(max(0.0, exp_cost[i] - slack[i] - self._expected_benchmark(i)) for i in self.member_ids)
            if self.mode == "soft_ir":
                checks["total_ir_slack"] = float(sum(slack.values()))
        if self.cc_eps is not None:
            z = self.var["cc_violate"]
            n_viol = sum(1 for k in range(self.N) if any(self.var["exceedance"][k, t].X > 1e-6 for t in self.T))
            checks["cc_in_sample_violations"] = n_viol
            checks["cc_allowed"] = int(np.floor(self.cc_eps * self.N + 1e-9)) if self.cc_joint else -1
            if self.cc_joint and n_viol > checks["cc_allowed"]:
                raise RuntimeError(f"chance constraint violated in-sample: {n_viol} > {checks['cc_allowed']}")
        residuals = {k: v for k, v in checks.items() if k not in ("total_ir_slack", "max_dual_over_bigM", "tolerance",
                                                                   "cc_in_sample_violations", "cc_allowed")}
        if checks["max_dual_over_bigM"] > 0.9:
            raise RuntimeError(f"big_m={self.big_m} is too small: a dual reached {checks['max_dual_over_bigM']:.0%} of it")
        if max(residuals.values()) > tolerance:
            raise RuntimeError(f"SDP solution failed independent checks: {checks}")
        return checks

    # ------------------------------------------------------------------ extraction
    def _extract_results(self, status: str) -> SDPResults:
        T, cap = list(self.T), np.asarray(self.com.contract.capacity_limit_kW, dtype=float)
        x = np.array([self.var["x_uni"][t].X for t in T])
        rows, mrows = [], []
        imp_e, exp_e, exc_e = np.zeros(len(T)), np.zeros(len(T)), np.zeros(len(T))
        for k, s in enumerate(self.scenarios):
            imp = np.array([self.var["import"][k, t].X for t in T]); exp = np.array([self.var["export"][k, t].X for t in T])
            exc = np.array([self.var["exceedance"][k, t].X for t in T])
            imp_e += self.prob[k] * imp; exp_e += self.prob[k] * exp; exc_e += self.prob[k] * exc
            pays = self.scen_expr["payments"][k].getValue(); dso = self.scen_expr["dso_cost"][k].getValue()
            rows.append({"scenario": s.id, "k": k, "probability": self.prob[k], "community_cost": self.scen_expr["community_cost"][k].getValue(),
                         "dso_bill": dso, "payments": pays, "budget": pays - dso, "exceedance_kWh": float(exc.sum()),
                         "max_exceedance_kW": float(exc.max()), "violated": bool(exc.max() > 1e-6), "peak_import_kW": float(imp.max()),
                         "cc_z": (self.var["cc_violate"][k].X if (self.cc_eps is not None and self.cc_joint) else np.nan)})
            for i in self.member_ids:
                cid = composite_id(i, k)
                pay = self.payment[cid].getValue(); priv = self.disutility[cid].getValue() + self.pv_cost[cid].getValue() + self.end_soc_value[cid].getValue()
                mrows.append({"scenario": s.id, "k": k, "member": i, "archetype": self.com.member(i).archetype, "payment": pay,
                              "private_cost": priv, "cost": pay + priv, "benchmark": self.benchmark_costs.get(cid, np.nan),
                              "ir_slack": (self.var["ir_slack"][i].X if self.mode == "soft_ir" else 0.0)})
        scen, mem = pd.DataFrame(rows), pd.DataFrame(mrows)
        hourly = pd.DataFrame({"posted_price": x, "import_expected": imp_e, "export_expected": exp_e, "exceedance_expected": exc_e, "cap": cap})
        meta = {"setup": "SDP", "mode": self.mode, "N": self.N, "cc_eps": self.cc_eps, "cc_joint": self.cc_joint,
                "cvar_beta": self.cvar_beta, "cvar_gamma": self.cvar_gamma if self.cvar_beta is not None else None,
                "expected_community_cost": self.expr["community_cost"].getValue(), "expected_dso_cost": self.expr["dso_cost"].getValue(),
                "expected_penalty": self.expr["penalty"].getValue(), "cvar": self.expr["cvar"].getValue() if "cvar" in self.expr else None,
                "total_ir_slack": float(sum(self.var["ir_slack"][i].X for i in self.member_ids)) if self.mode == "soft_ir" else 0.0,
                "in_sample_violations": int(scen["violated"].sum()),
                "solver_objective": self.stage1.get("objective", self.m.ObjVal), "solver_bound": self.stage1.get("bound", self.m.ObjBound),
                "solver_gap": self.stage1.get("gap", self.m.MIPGap), "runtime": self.stage1.get("runtime", self.m.Runtime),
                "price_rule": self.price_rule, "stage2_runtime": self.stage2.get("runtime", 0.0),
                "max_price_deviation": self.stage2.get("max_deviation", np.nan), "sum_price_deviation": self.stage2.get("sum_deviation", np.nan),
                "n_binaries": self.m.NumBinVars, "n_constraints": self.m.NumConstrs, "n_vars": self.m.NumVars,
                "big_m": self.big_m, "price_bounds": self.price_bounds, "end_soc_mode": self.end_soc_mode}
        return SDPResults(status=status, objective=float(self.expr["community_cost"].getValue()), prices=x,
                          scenarios=scen, members=mem, hourly=hourly, meta=meta)
