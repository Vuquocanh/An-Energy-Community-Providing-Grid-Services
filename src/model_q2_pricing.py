"""Dynamic Pricing Model for Question 2 (bilevel -> MPEC -> MILP).

The community manager posts one internal price x_t per hour; each member i then solves its own
Assignment-1 problem at those prices. The manager anticipates the members' best responses, so the
bilevel problem is rewritten as a single-level MPEC:

1. Lower level: each member's LP (load, deviation, daily minimum, PV, battery, end-of-day SoC).
2. Reformulation: the lower level is replaced by its complete KKT system -- stationarity, dual
   feasibility and complementarity -- with every complementarity linearised by a Big-M
   (``_complementarity``).  The meter flows carry an SOS1 so that import and export never overlap.
3. Dual trick: the payment ``sum_t x_t q_{i,t}`` is bilinear (price x dispatch); by lower-level
   strong duality it equals the member's dual objective, which is linear in the dual variables.
4. Requirements (the mandatory Q2 decomposition and the infeasibility extensions):
   - ``cap_only``  : no requirement, community welfare only;
   - ``BB``        : budget balance, payments collected == the manager's DSO cost;
   - ``BB_IR``     : BB + individual rationality (no member worse off than model R);
   - ``soft_ir``   : BB + IR as a soft constraint, member i pays at most
                     ``benchmark_i + s_i`` with a penalty ``w * sum_i s_i``; always feasible,
                     reports the minimum compensation and the blocking members;
   - ``extended``  : BB + IR restored by lump-sum transfers ``delta_i >= 0`` (compensation).
5. ``allow_price_discrimination`` replaces the uniform price by x_{i,t} (counterfactual for the
   cost of uniformity).

Everything the model reports is checked a posteriori by :meth:`DynamicPricingModel.validate_solution`,
which re-solves every member's LP from scratch and verifies that the reported dispatch is really a
best response and that the dual-derived payments equal the actual payments ``sum_t x_t q_{i,t}``.
"""
from __future__ import annotations

from typing import Any, Literal

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from .data_loader import CommunityData, InputData
from .model import Results, _status_name


class NoPricingSolution(RuntimeError):
    """No incumbent; only status INFEASIBLE establishes infeasibility."""

    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"No solution available: {_status_name(status_code)}")


MODES = ("cap_only", "BB", "BB_IR", "soft_ir", "extended")


class DynamicPricingModel:
    """Question 2 dynamic pricing model (MPEC -> MILP via KKT and strong duality).

    Parameters
    ----------
    com : CommunityData
        The community (members, day-ahead prices, contract).
    mode : {'cap_only', 'BB', 'BB_IR', 'soft_ir', 'extended'}
        Which requirements are imposed (see the module docstring).
    benchmark_costs : dict[str, float] | None
        Member cost under the outside option (model R, or model I), DKK.  Required by every mode
        that imposes (hard or soft) individual rationality.
    allow_price_discrimination : bool
        If True the manager posts one price per member and hour (x_{i,t}) instead of x_t.
    end_soc_mode : {'constraint', 'objective', 'free'}
        Battery end-of-day convention, as in ``CommunityModel``.
    final_soc_price : float | None
        DKK/kWh of the energy left in the battery at t = 24 when ``end_soc_mode='objective'``;
        default the member's ``energy_price[-1] + import_tariff``, as in ``CommunityModel``.
    big_m : float
        Big-M of the complementarity constraints, DKK/kWh.  It must exceed every dual variable.
    price_bounds : (float, float) | None
        Optional bounds on every posted price (a modelling choice of Table 11; None = free price).
    ir_penalty : float
        Weight of ``sum_i s_i`` in the objective of ``soft_ir`` (DKK per DKK of compensation).
    mip_gap, time_limit : float
        Solver settings.
    mip_focus : int
        1 emphasises finding feasible solutions (useful for BB_IR / soft_ir).
    """

    def __init__(
        self,
        com: CommunityData,
        mode: Literal["cap_only", "BB", "BB_IR", "soft_ir", "extended"] = "cap_only",
        benchmark_costs: dict[str, float] | None = None,
        allow_price_discrimination: bool = False,
        end_soc_mode: Literal["constraint", "objective", "free"] = "constraint",
        final_soc_price: float | None = None,
        big_m: float = 500.0,
        price_bounds: tuple[float, float] | None = None,
        ir_penalty: float = 1e3,
        mip_gap: float = 1e-3,
        time_limit: float = 300,
        mip_focus: int = 0,
        verbose: bool = False,
    ):
        if mode not in MODES:
            raise ValueError(f"Invalid mode: {mode!r}")
        if mode in ("BB_IR", "soft_ir", "extended") and benchmark_costs is None:
            raise ValueError(f"Benchmark costs dictionary required for mode {mode!r}")

        self.com = com
        self.mode = mode
        self.benchmark_costs = benchmark_costs or {}
        self.allow_price_discrimination = allow_price_discrimination
        self.end_soc_mode = end_soc_mode
        self.final_soc_price = final_soc_price
        self.big_m = float(big_m)
        self.price_bounds = price_bounds
        self.ir_penalty = float(ir_penalty)

        if not np.isfinite(self.big_m) or self.big_m <= 0:
            raise ValueError("big_m must be finite and positive")
        if end_soc_mode not in ("constraint", "objective", "free"):
            raise ValueError("Unknown end_soc_mode")
        if time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if price_bounds is not None and price_bounds[0] > price_bounds[1]:
            raise ValueError("price_bounds must be (lower, upper)")
        self.T = range(com.members[0].n_hours)
        self.ids = com.member_ids

        self.m = gp.Model(f"pricing_{mode}")
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.DualReductions = 0

        # Solver performance tuning
        self.m.Params.MIPGap = mip_gap
        self.m.Params.Presolve = 2
        self.m.Params.TimeLimit = time_limit
        self.m.Params.Threads = 4
        if mip_focus:
            self.m.Params.MIPFocus = mip_focus
        self.complementarity_pairs = []
        self.end_soc_value = {}

        # Model containers
        self.var: dict[str, Any] = {}
        self.con: dict[str, Any] = {}
        self.expr: dict[str, gp.LinExpr] = {}

        # Lower-level decision variables & expressions per member
        self.mvar: dict[str, dict[str, Any]] = {}
        self.mcon: dict[str, dict[str, Any]] = {}
        self.net: dict[str, dict[int, gp.LinExpr]] = {}
        self.payment: dict[str, gp.LinExpr] = {}
        self.disutility: dict[str, gp.LinExpr] = {}
        self.pv_cost: dict[str, gp.LinExpr] = {}

    # ------------------------------------------------------------------ build
    def build(self) -> "DynamicPricingModel":
        """Construct the complete single-level MILP formulation."""
        self._add_upper_level_variables()
        for d in self.com.members:
            self._add_lower_level_member_kkt(d)
        self._add_community_meter_and_dso_costs()
        self._add_pricing_requirements()
        self._set_objective()
        self.m.update()
        return self

    def _complementarity(self, multiplier, slack_expression, name):
        """Enforce ``multiplier >= 0, slack >= 0, multiplier * slack = 0`` with a binary and Big-M.

        The slack expression is written as an auxiliary variable, so the multiplier of the
        corresponding bound/constraint is implicitly defined by ``slack_expression``; the Big-M
        pair then enforces non-negativity and complementarity.
        """
        m = self.m
        M = self.big_m

        slack = m.addVar(lb=0.0, name=f"slack_{name}")
        z = m.addVar(vtype=GRB.BINARY, name=f"binary_{name}")

        m.addConstr(slack == slack_expression, name=f"slack_definition_{name}")
        m.addConstr(multiplier <= M * z, name=f"bigM_first_{name}")
        m.addConstr(slack <= M * (1 - z), name=f"bigM_second_{name}")
        self.complementarity_pairs.append((multiplier, slack))

    def _add_upper_level_variables(self) -> None:
        """Add posted prices x_t (or x_{i,t}) and optional transfer/slack variables."""
        m, T, ids = self.m, self.T, self.ids
        lo, hi = (-GRB.INFINITY, GRB.INFINITY) if self.price_bounds is None else self.price_bounds
        if self.allow_price_discrimination:
            x = m.addVars(ids, T, lb=lo, ub=hi, name="x_price")
        else:
            x_uni = m.addVars(T, lb=lo, ub=hi, name="x_price_uni")
            x = {(i, t): x_uni[t] for i in ids for t in T}
        self.var["x_price"] = x

        if self.mode == "extended":
            self.var["delta_transfer"] = m.addVars(ids, lb=0.0, name="delta_transfer")
        if self.mode == "soft_ir":
            self.var["ir_slack"] = m.addVars(ids, lb=0.0, name="ir_slack")

    def _add_lower_level_member_kkt(self, d: InputData) -> None:
        """Formulate complete member KKT: primal variables, duals, stationarity, complementarity."""
        m, T, i = self.m, self.T, d.consumer_id
        ref = d.reference_load
        if ref is None or d.linear_disutility is None or d.min_daily_energy_kWh is None:
            raise ValueError(f"Member {i!r} is missing reference profile or disutility slope.")

        # --- Primal variables ---
        load = m.addVars(T, lb=-GRB.INFINITY, name=f"load_{i}")
        dev = m.addVars(T, lb=0.0, name=f"dev_{i}")
        self._put(self.mvar, "load", i, load)
        self._put(self.mvar, "dev", i, dev)

        net = {t: gp.LinExpr(load[t]) for t in T}

        if d.has_pv:
            pv = m.addVars(T, lb=0.0, name=f"pv_{i}")
            self._put(self.mvar, "pv", i, pv)
            for t in T:
                net[t].add(pv[t], -1.0)

        if d.has_battery:
            p_ch = m.addVars(T, lb=0.0, ub=d.battery_max_charge_kW, name=f"p_ch_{i}")
            p_dis = m.addVars(T, lb=0.0, ub=d.battery_max_discharge_kW, name=f"p_dis_{i}")
            soc = m.addVars(range(len(T) + 1), lb=0.0, ub=d.battery_capacity_kWh, name=f"soc_{i}")
            self._put(self.mvar, "p_ch", i, p_ch)
            self._put(self.mvar, "p_dis", i, p_dis)
            self._put(self.mvar, "soc", i, soc)
            for t in T:
                net[t].add(p_ch[t] - p_dis[t], 1.0)

        self.net[i] = net

        # --- Dual variables ---
        alpha_up = m.addVars(T, lb=0.0, name=f"alpha_up_{i}")
        alpha_dn = m.addVars(T, lb=0.0, name=f"alpha_dn_{i}")
        nu_lower = m.addVars(T, lb=0.0, name=f"nu_lower_{i}")
        nu_upper = m.addVars(T, lb=0.0, name=f"nu_upper_{i}")
        mu_energy = m.addVar(lb=0.0, name=f"mu_energy_{i}")

        self._put(self.mvar, "dual_dev_up", i, alpha_up)
        self._put(self.mvar, "dual_dev_dn", i, alpha_dn)
        self._put(self.mvar, "dual_load_lower", i, nu_lower)
        self._put(self.mvar, "dual_load_upper", i, nu_upper)
        self.mvar.setdefault("dual_min_daily_energy", {})[i] = mu_energy

        if d.has_pv:
            phi_pv = m.addVars(T, lb=0.0, name=f"phi_pv_{i}")
            self._put(self.mvar, "dual_pv_max", i, phi_pv)

        if d.has_battery:
            gamma_ch = m.addVars(T, lb=0.0, name=f"gamma_ch_{i}")
            gamma_dis = m.addVars(T, lb=0.0, name=f"gamma_dis_{i}")
            omega_soc = m.addVars(range(1, len(T) + 1), lb=0.0, name=f"omega_soc_{i}")
            theta_soc = m.addVars(range(len(T) + 1), lb=-GRB.INFINITY, name=f"theta_soc_{i}")
            self._put(self.mvar, "dual_p_ch_max", i, gamma_ch)
            self._put(self.mvar, "dual_p_dis_max", i, gamma_dis)
            self._put(self.mvar, "dual_soc_max", i, omega_soc)
            self._put(self.mvar, "dual_theta_soc", i, theta_soc)
            if self.end_soc_mode == "constraint":
                pi_end = m.addVar(lb=0.0, name=f"pi_end_{i}")
                self.mvar.setdefault("dual_soc_end", {})[i] = pi_end

        # --- Lower-level primal constraints & complementarity (Big-M) ---
        for t in T:
            m.addConstr(dev[t] - load[t] >= -ref[t], name=f"c_dev_up_{i}_{t}")
            self._complementarity(alpha_up[t], dev[t] - load[t] + ref[t], f"dev_up_{i}_{t}")

            m.addConstr(dev[t] + load[t] >= ref[t], name=f"c_dev_dn_{i}_{t}")
            self._complementarity(alpha_dn[t], dev[t] + load[t] - ref[t], f"dev_dn_{i}_{t}")

            m.addConstr(load[t] >= d.load_lower_kWh[t], name=f"c_load_lower_{i}_{t}")
            self._complementarity(nu_lower[t], load[t] - d.load_lower_kWh[t], f"load_lower_{i}_{t}")

            m.addConstr(load[t] <= d.load_upper_kWh[t], name=f"c_load_upper_{i}_{t}")
            self._complementarity(nu_upper[t], d.load_upper_kWh[t] - load[t], f"load_upper_{i}_{t}")

        m.addConstr(load.sum() >= d.min_daily_energy_kWh, name=f"c_min_energy_{i}")
        self._complementarity(mu_energy, load.sum() - d.min_daily_energy_kWh, f"energy_{i}")

        if d.has_pv:
            for t in T:
                m.addConstr(pv[t] <= d.pv_available[t], name=f"c_pv_max_{i}_{t}")
                self._complementarity(phi_pv[t], d.pv_available[t] - pv[t], f"pv_upper_{i}_{t}")

        if d.has_battery:
            m.addConstr(soc[0] == d.battery_initial_soc_kWh, name=f"c_soc_init_{i}")

            for t in T:
                m.addConstr(
                    soc[t + 1] == soc[t] + d.battery_charging_efficiency * p_ch[t] - p_dis[t] / d.battery_discharging_efficiency,
                    name=f"c_soc_dyn_{i}_{t}",
                )
                m.addConstr(p_ch[t] <= d.battery_max_charge_kW)
                self._complementarity(gamma_ch[t], d.battery_max_charge_kW - p_ch[t], f"charge_upper_{i}_{t}")

                m.addConstr(p_dis[t] <= d.battery_max_discharge_kW)
                self._complementarity(gamma_dis[t], d.battery_max_discharge_kW - p_dis[t], f"discharge_upper_{i}_{t}")

                m.addConstr(soc[t + 1] <= d.battery_capacity_kWh)
                self._complementarity(omega_soc[t + 1], d.battery_capacity_kWh - soc[t + 1], f"soc_upper_{i}_{t+1}")

            if self.end_soc_mode == "constraint":
                m.addConstr(soc[len(T)] >= d.battery_initial_soc_kWh)
                self._complementarity(self.mvar["dual_soc_end"][i], soc[len(T)] - d.battery_initial_soc_kWh, f"soc_end_{i}")

        # --- Stationarity ---
        x = self.var["x_price"]
        c_L = d.linear_disutility
        c_PV = d.pv_marginal_cost

        for t in T:
            m.addConstr(x[i, t] + alpha_up[t] - alpha_dn[t] - nu_lower[t] + nu_upper[t] - mu_energy == 0, name=f"stat_load_{i}_{t}")
            self._complementarity(dev[t], c_L - alpha_up[t] - alpha_dn[t], f"stat_dev_{i}_{t}")

            if d.has_pv:
                self._complementarity(pv[t], -x[i, t] + c_PV + phi_pv[t], f"stat_pv_{i}_{t}")

            if d.has_battery:
                eta_ch, eta_dis = d.battery_charging_efficiency, d.battery_discharging_efficiency
                self._complementarity(p_ch[t], x[i, t] - eta_ch * theta_soc[t + 1] + gamma_ch[t], f"stat_pch_{i}_{t}")
                self._complementarity(p_dis[t], -x[i, t] + theta_soc[t + 1] / eta_dis + gamma_dis[t], f"stat_pdis_{i}_{t}")

        if d.has_battery:
            # theta[0] is the multiplier of (S_initial - soc[0]) = 0.
            # Dynamics use theta[t+1]*(soc[t+1]-soc[t]-eta*ch+dis/eta).
            # soc[0] is fixed and below capacity in the base case, so its bound multiplier is zero.
            m.addConstr(theta_soc[0] + theta_soc[1] == 0, name=f"stat_soc_initial_{i}")
            for t in range(1, len(T)):
                self._complementarity(soc[t], theta_soc[t] - theta_soc[t + 1] + omega_soc[t], f"stat_soc_{i}_{t}")
            if self.end_soc_mode == "constraint":
                self._complementarity(soc[len(T)], theta_soc[len(T)] - self.mvar["dual_soc_end"][i] + omega_soc[len(T)], f"stat_soc_end_{i}")
            elif self.end_soc_mode == "objective":
                p_end = self._final_soc_price(d)
                self._complementarity(soc[len(T)], theta_soc[len(T)] - p_end + omega_soc[len(T)], f"stat_soc_end_{i}")
            else:
                self._complementarity(soc[len(T)], theta_soc[len(T)] + omega_soc[len(T)], f"stat_soc_end_{i}")

        # --- Strong duality: the member's dual objective, the payment identity ---
        dual_obj = gp.LinExpr()
        dual_obj.add(mu_energy * d.min_daily_energy_kWh)
        for t in T:
            dual_obj.add(alpha_dn[t] * ref[t] - alpha_up[t] * ref[t])
            dual_obj.add(nu_lower[t] * d.load_lower_kWh[t] - nu_upper[t] * d.load_upper_kWh[t])
            if d.has_pv:
                dual_obj.add(-phi_pv[t] * d.pv_available[t])
            if d.has_battery:
                dual_obj.add(-gamma_ch[t] * d.battery_max_charge_kW)
                dual_obj.add(-gamma_dis[t] * d.battery_max_discharge_kW)

        if d.has_battery:
            dual_obj.add(theta_soc[0] * d.battery_initial_soc_kWh)
            for t in range(1, len(T) + 1):
                dual_obj.add(-omega_soc[t] * d.battery_capacity_kWh)
            if self.end_soc_mode == "constraint":
                dual_obj.add(self.mvar["dual_soc_end"][i] * d.battery_initial_soc_kWh)

        disut = c_L * dev.sum()
        pv_c = (c_PV * pv.sum()) if d.has_pv else gp.LinExpr(0.0)

        self.disutility[i] = disut
        self.pv_cost[i] = pv_c
        terminal = -self._final_soc_price(d) * soc[len(T)] if d.has_battery and self.end_soc_mode == "objective" else gp.LinExpr(0)
        self.end_soc_value[i] = terminal
        self.payment[i] = dual_obj - disut - pv_c - terminal

    def _final_soc_price(self, d: InputData) -> float:
        """DKK/kWh of the energy left at t = 24 in ``end_soc_mode='objective'``."""
        if self.final_soc_price is not None:
            return float(self.final_soc_price)
        p_last = d.energy_price[-1] if getattr(d, "energy_price", None) is not None else self.com.energy_price[-1]
        return float(p_last + d.import_tariff)

    def _add_community_meter_and_dso_costs(self) -> None:
        """Formulate community grid import/export meter and DSO cost calculations."""
        m, T, p = self.m, self.T, self.com.energy_price
        c = self.com.contract

        imp = m.addVars(T, lb=0.0, name="imp_grid")
        exp = m.addVars(T, lb=0.0, name="exp_grid")
        exc = m.addVars(T, lb=0.0, name="exc_grid")

        self.var["import"] = imp
        self.var["export"] = exp
        self.var["exceedance"] = exc

        for t in T:
            m.addConstr(imp[t] - exp[t] == gp.quicksum(self.net[i][t] for i in self.ids), name=f"comm_bal_{t}")
            # e_t = max(0, import - cap) *exactly*: with budget balance the DSO cost is an equality
            # constraint, so leaving only e_t >= import - cap lets the model inflate the manager's
            # cost (and hence balance the budget) by declaring a fictitious exceedance.
            m.addConstr(exc[t] >= imp[t] - c.capacity_limit_kW[t], name=f"comm_cap_{t}")
            z_cap = m.addVar(vtype=GRB.BINARY, name=f"cap_binary_{t}")
            m.addConstr(exc[t] <= self.big_m * z_cap, name=f"exc_zero_{t}")
            m.addConstr(exc[t] <= imp[t] - c.capacity_limit_kW[t] + self.big_m * (1 - z_cap), name=f"exc_exact_{t}")
            # One meter: no simultaneous import and export.
            m.addSOS(GRB.SOS_TYPE1, [imp[t], exp[t]])

        f = 1.0 - c.tariff_discount_ratio
        tau_imp, tau_exp = self.com.import_tariff, self.com.export_tariff

        cost_energy = gp.quicksum(p[t] * (imp[t] - exp[t]) for t in T)
        cost_tariff = gp.quicksum(f * tau_imp * (imp[t] - exc[t]) + tau_imp * exc[t] + f * tau_exp * exp[t] for t in T)
        cost_penalty = c.penalty_DKK_per_kWh * exc.sum()

        self.expr["energy"] = cost_energy
        self.expr["tariff"] = cost_tariff
        self.expr["penalty"] = cost_penalty
        self.expr["dso_cost"] = cost_energy + cost_tariff + cost_penalty

    def _member_cost(self, i: str) -> gp.LinExpr:
        """What member i pays under the posted prices plus its own disutility/PV/terminal cost."""
        return self.payment[i] + self.disutility[i] + self.pv_cost[i] + self.end_soc_value[i]

    def _add_pricing_requirements(self) -> None:
        """Budget balance, individual rationality (hard, soft or via transfers)."""
        m = self.m
        total_payment = gp.quicksum(self.payment.values())

        if self.mode in ("BB", "BB_IR", "soft_ir"):
            m.addConstr(total_payment == self.expr["dso_cost"], name="req_budget_balance")
        elif self.mode == "extended":
            delta = self.var["delta_transfer"]
            m.addConstr(total_payment - gp.quicksum(delta.values()) == self.expr["dso_cost"], name="req_budget_balance_ext")

        if self.mode == "BB_IR":
            for i in self.ids:
                m.addConstr(self._member_cost(i) <= self.benchmark_costs[i], name=f"req_ir_{i}")
        elif self.mode == "soft_ir":
            s = self.var["ir_slack"]
            for i in self.ids:
                m.addConstr(self._member_cost(i) - s[i] <= self.benchmark_costs[i], name=f"req_ir_soft_{i}")
        elif self.mode == "extended":
            delta = self.var["delta_transfer"]
            for i in self.ids:
                m.addConstr(self._member_cost(i) - delta[i] <= self.benchmark_costs[i], name=f"req_ir_ext_{i}")

    def _set_objective(self) -> None:
        """Minimise the community cost, plus the penalty of the relaxation in the extended modes."""
        objective = self.expr["dso_cost"] + gp.quicksum(self.disutility.values()) \
            + gp.quicksum(self.pv_cost.values()) + gp.quicksum(self.end_soc_value.values())
        # The community cost is what the report compares; the penalty only selects among solutions.
        self.expr["community_cost"] = objective.copy()

        if self.mode == "soft_ir":
            s = self.var["ir_slack"]
            objective.add(self.ir_penalty * gp.quicksum(s[i] for i in self.ids))
        elif self.mode == "extended":
            delta = self.var["delta_transfer"]
            objective.add(1e3 * gp.quicksum(delta[i] for i in self.ids))

        self.m.setObjective(objective, GRB.MINIMIZE)

    # ------------------------------------------------------------------ warm start
    def warm_start_from(self, other: "DynamicPricingModel") -> int:
        """Copy ``Start`` values of every variable whose name exists in both models (MIP start)."""
        if other.m.SolCount == 0:
            return 0
        by_name = {v.VarName: v for v in other.m.getVars()}
        n = 0
        for v in self.m.getVars():
            src = by_name.get(v.VarName)
            if src is None:
                continue
            try:
                v.Start = src.X
                n += 1
            except (gp.GurobiError, AttributeError):
                pass
        return n

    # ------------------------------------------------------------------ solve
    def solve(self) -> Results:
        """Solve the MILP and extract results into the standard ``Results`` dataclass."""
        self.m.update()
        self.m.optimize()
        status = _status_name(self.m.Status)

        if self.m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or (
            self.m.Status == GRB.TIME_LIMIT and self.m.SolCount > 0
        ):
            if self.m.Status == GRB.TIME_LIMIT:
                print("  [WARNING] Optimization hit TIME_LIMIT, using the best feasible solution found.")
            results = self._extract_results(status)
            results.meta["validation"] = self.validate_solution()
            return results

        raise NoPricingSolution(self.m.Status)

    # ------------------------------------------------------------------ validation
    def validate_solution(self, tolerance: float | None = None) -> dict[str, float]:
        """Check the solution independently of the reformulation.

        Re-solves every member's LP at the posted prices (from scratch, with ``CommunityModel``),
        compares objective values (not schedules: optimal responses may be non-unique), checks the
        payment identity ``payment_i = sum_t x_t q_{i,t}``, the meter, the complementarity products
        and the requirements.

        The default tolerance scales with the Big-M: the solver's feasibility tolerance (1e-6) times
        a multiplier of 500 already produces complementarity residuals of order 5e-4, which are
        numerical noise, not a modelling error.
        """
        if tolerance is None:
            tolerance = max(5e-4, 2.0 * float(self.m.Params.FeasibilityTol) * self.big_m)
        from .model import CommunityModel

        payment_errors, response_errors, ir_errors = [], [], []
        actual_payments, transfers, slacks = 0.0, 0.0, 0.0
        for d in self.com.members:
            i = d.consumer_id
            prices = [self.var["x_price"][i, t].X for t in self.T]
            paid = sum(prices[t] * self.net[i][t].getValue() for t in self.T)
            private = self.disutility[i].getValue() + self.pv_cost[i].getValue() + self.end_soc_value[i].getValue()
            payment_errors.append(abs(paid - self.payment[i].getValue()))
            follower = CommunityModel(self.com, "C", end_soc_mode=self.end_soc_mode,
                                      final_soc_price=self.final_soc_price)
            try:
                follower._add_member(d)
                f = gp.quicksum(v[i] for v in follower.mexpr.values())
                follower.m.setObjective(f + gp.quicksum(prices[t] * follower.net[i][t] for t in self.T))
                follower.m.Params.OutputFlag = 0
                follower.m.optimize()
                if follower.m.Status != GRB.OPTIMAL:
                    raise RuntimeError(f"Independent follower LP failed for {i}")
                response_errors.append(abs(paid + private - follower.m.ObjVal))
            finally:
                follower.m.dispose()
            delta = self.var["delta_transfer"][i].X if self.mode == "extended" else 0.0
            slack = self.var["ir_slack"][i].X if self.mode == "soft_ir" else 0.0
            actual_payments += paid
            transfers += delta
            slacks += slack
            if self.mode in ("BB_IR", "soft_ir", "extended"):
                ir_errors.append(max(0.0, paid + private - delta - slack - self.benchmark_costs[i]))

        meter_errors, exc_errors, overlap = [], [], []
        for t in self.T:
            imp, exp, exc = (self.var[k][t].X for k in ("import", "export", "exceedance"))
            meter_errors.append(abs(imp - exp - sum(self.net[i][t].getValue() for i in self.ids)))
            exc_errors.append(abs(exc - max(0.0, imp - self.com.contract.capacity_limit_kW[t])))
            overlap.append(min(imp, exp))

        checks = {
            "max_payment_identity_error": max(payment_errors),
            "max_best_response_gap": max(response_errors),
            "max_complementarity_product": max(abs(a.X * b.X) for a, b in self.complementarity_pairs),
            "max_meter_error": max(meter_errors),
            "max_exceedance_error": max(exc_errors),
            "max_simultaneous_grid_flow": max(overlap),
            "max_linear_constraint_violation": self.m.ConstrVio,
            "max_bound_violation": self.m.BoundVio,
        }
        if self.mode != "cap_only":
            checks["budget_error_after_transfers"] = abs(actual_payments - transfers - self.expr["dso_cost"].getValue())
        if ir_errors:
            checks["max_IR_violation_after_transfers"] = max(ir_errors)
        if self.mode == "soft_ir":
            checks["total_ir_slack"] = slacks

        # total_ir_slack is a reported quantity, not a residual: do not compare it with the tolerance
        residuals = {k: v for k, v in checks.items() if k != "total_ir_slack"}
        if max(residuals.values()) > tolerance:
            raise RuntimeError(f"Pricing solution failed independent checks: {checks}")
        return checks

    # ------------------------------------------------------------------ extraction
    def _extract_results(self, status: str) -> Results:
        T, com = list(self.T), self.com
        x_sol = self.var["x_price"]

        frames = {}
        for i in self.ids:
            d = com.member(i)
            df = pd.DataFrame(index=pd.Index(T, name="hour"))
            df["reference_load"] = d.reference_load
            df["pv_available"] = d.pv_available
            df["posted_price"] = [x_sol[i, t].X for t in T]

            for name, per_member in self.mvar.items():
                if i in per_member and isinstance(per_member[i], (gp.tupledict, dict)):
                    v = per_member[i]
                    col_vals = []
                    for t in T:
                        if t in v:
                            item = v[t]
                            col_vals.append(item.X if hasattr(item, "X") else item)
                        elif (t + 1) in v:
                            item = v[t + 1]
                            col_vals.append(item.X if hasattr(item, "X") else item)
                    if len(col_vals) == len(T):
                        df[name] = col_vals

            df["net"] = [self.net[i][t].getValue() for t in T]
            frames[i] = df
        members = pd.concat(frames, names=["member"])

        costs_dict = {
            "archetype": [com.member(i).archetype for i in self.ids],
            "disutility": [self.disutility[i].getValue() for i in self.ids],
            "pv_cost": [self.pv_cost[i].getValue() for i in self.ids],
            "payment": [self.payment[i].getValue() for i in self.ids],
            "end_soc_value": [self.end_soc_value[i].getValue() for i in self.ids],
        }
        if self.mode == "extended":
            costs_dict["delta_transfer"] = [self.var["delta_transfer"][i].X for i in self.ids]
        if self.mode == "soft_ir":
            costs_dict["ir_slack"] = [self.var["ir_slack"][i].X for i in self.ids]

        costs = pd.DataFrame(costs_dict, index=self.ids)
        costs["total_cost_before_transfer"] = (costs["disutility"] + costs["pv_cost"]
                                               + costs["end_soc_value"] + costs["payment"])
        if "delta_transfer" not in costs:
            costs["delta_transfer"] = 0.0
        if "ir_slack" not in costs:
            costs["ir_slack"] = 0.0
        costs["total_cost"] = costs["total_cost_before_transfer"] - costs["delta_transfer"]
        if self.benchmark_costs:
            costs["benchmark_cost"] = [self.benchmark_costs.get(i, np.nan) for i in self.ids]
            costs["ir_violation_or_gain"] = costs["total_cost"] - costs["benchmark_cost"]

        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = com.energy_price
        hourly["cap"] = com.contract.capacity_limit_kW
        hourly["posted_price"] = [x_sol[self.ids[0], t].X for t in T]
        if self.allow_price_discrimination:
            hourly["posted_price_min"] = [min(x_sol[i, t].X for i in self.ids) for t in T]
            hourly["posted_price_max"] = [max(x_sol[i, t].X for i in self.ids) for t in T]

        for col in ("reference_load", "load", "pv", "p_ch", "p_dis"):
            if col in members:
                hourly[col] = members[col].groupby(level="hour").sum()

        hourly["import"] = [self.var["import"][t].X for t in T]
        hourly["export"] = [self.var["export"][t].X for t in T]
        hourly["exceedance_billed"] = [self.var["exceedance"][t].X for t in T]
        hourly["exceedance"] = (hourly["import"] - hourly["cap"]).clip(lower=0.0)

        meta_obj = {
            "dso_cost": self.expr["dso_cost"].getValue(),
            "energy_cost": self.expr["energy"].getValue(),
            "tariff_cost": self.expr["tariff"].getValue(),
            "penalty_cost": self.expr["penalty"].getValue(),
            "total_disutility": float(costs["disutility"].sum()),
            "total_pv_cost": float(costs["pv_cost"].sum()),
            "gross_payments_collected": float(costs["payment"].sum()),
            "total_transfers": float(costs["delta_transfer"].sum()),
            "total_payments_collected": float((costs["payment"] - costs["delta_transfer"]).sum()),
            "total_ir_slack": float(costs["ir_slack"].sum()),
            "community_cost": self.expr["community_cost"].getValue(),
            "solver_objective": self.m.ObjVal,
            "transfer_penalty": 1000 * float(costs["delta_transfer"].sum()) if self.mode == "extended" else 0.0,
            "ir_penalty": self.ir_penalty * float(costs["ir_slack"].sum()) if self.mode == "soft_ir" else 0.0,
        }

        return Results(
            question=f"{com.community}_{com.day}_DP_{self.mode}",
            status=status,
            objective=self.expr["community_cost"].getValue(),
            hourly=hourly,
            duals={},
            meta={"setup": "DP", "mode": self.mode, "objective_terms": meta_obj,
                  "solver_objective": self.m.ObjVal,
                  "solver_bound": self.m.ObjBound,
                  "solver_gap": self.m.MIPGap, "runtime": self.m.Runtime,
                  "end_soc_mode": self.end_soc_mode,
                  "price_bounds": self.price_bounds,
                  "allow_price_discrimination": self.allow_price_discrimination},
            members=members,
            member_costs=costs,
        )

    @staticmethod
    def _put(store: dict, name: str, i: str, item) -> None:
        store.setdefault(name, {})[i] = item
