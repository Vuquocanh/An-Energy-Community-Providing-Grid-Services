"""Repaired Question 2 pricing model: complete LP KKT with SOS1.

Formulates the community manager's internal pricing problem:
1. Lower level: Each prosumer i responds independently to posted hourly prices x_t (or x_{i,t}).
2. Reformulation: complete KKT, including lower bounds, using SOS1 complementarity.
3. Dual trick: Payment_i = sum_t x_t * q_{i,t} linearized via lower-level Strong Duality.
4. Requirements (3-step variants):
   - 'cap_only': Maximize community welfare under grid cap.
   - 'BB': Budget balance (manager revenues = manager DSO costs).
   - 'BB_IR': Budget balance + Individual Rationality (no member worse off than benchmark).
   - 'extended': Relaxation via lump-sum transfers delta_i or price discrimination.
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


class DynamicPricingModel:
    """Question 2 Dynamic Pricing Model (MPEC -> MILP via KKT and Strong Duality)."""

    def __init__(
        self,
        com: CommunityData,
        mode: Literal["cap_only", "BB", "BB_IR", "extended"] = "cap_only",
        benchmark_costs: dict[str, float] | None = None,
        allow_price_discrimination: bool = False,
        end_soc_mode: Literal["constraint", "objective", "free"] = "constraint",
        big_m: float | None = None,
        verbose: bool = False,
        time_limit: float = 120,
    ):
        if mode not in ("cap_only", "BB", "BB_IR", "extended"):
            raise ValueError(f"Invalid mode: {mode!r}")
        if mode in ("BB_IR", "extended") and benchmark_costs is None:
            raise ValueError(f"Benchmark costs dictionary required for mode {mode!r}")

        self.com = com
        self.mode = mode
        self.benchmark_costs = benchmark_costs or {}
        self.allow_price_discrimination = allow_price_discrimination
        self.end_soc_mode = end_soc_mode
        self.big_m = big_m
        if big_m is not None:
            import warnings
            warnings.warn("big_m is ignored: complementarity now uses SOS1.", stacklevel=2)
        if end_soc_mode not in ("constraint", "objective", "free"):
            raise ValueError("Unknown end_soc_mode")
        if time_limit <= 0:
            raise ValueError("time_limit must be positive")
        self.T = range(com.members[0].n_hours)
        self.ids = com.member_ids

        self.m = gp.Model(f"pricing_{mode}")
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.DualReductions = 0

        # Solver performance tuning
        self.m.Params.MIPGap = 1e-3
        self.m.Params.Presolve = 2
        self.m.Params.TimeLimit = time_limit
        self.m.Params.Threads = 4
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

    def build(self) -> DynamicPricingModel:
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
        """Nonnegative multiplier and slack; at most one may be nonzero."""
        slack = self.m.addVar(lb=0, name=f"slack_{name}")
        self.m.addConstr(slack == slack_expression, name=f"slack_definition_{name}")
        self.m.addSOS(GRB.SOS_TYPE1, [multiplier, slack])
        self.complementarity_pairs.append((multiplier, slack))

    def _add_upper_level_variables(self) -> None:
        """Add posted prices x_t (or x_{i,t}) and optional transfer variables."""
        m, T, ids = self.m, self.T, self.ids
        if self.allow_price_discrimination:
            x = m.addVars(ids, T, lb=-GRB.INFINITY, name="x_price")
        else:
            x_uni = m.addVars(T, lb=-GRB.INFINITY, name="x_price_uni")
            x = {(i, t): x_uni[t] for i in ids for t in T}
        self.var["x_price"] = x

        if self.mode == "extended":
            self.var["delta_transfer"] = m.addVars(ids, lb=0.0, name="delta_transfer")

    def _add_lower_level_member_kkt(self, d: InputData) -> None:
        """Formulate complete member KKT, with SOS1 for every inequality/bound."""
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

        # --- Lower-Level Primal Constraints & Complementarity (SOS1) ---
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

        # --- Stationarity Conditions ---
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
            # soc[0] is fixed, so its bound multipliers can be taken as zero.
            m.addConstr(theta_soc[0] + theta_soc[1] == 0, name=f"stat_soc_initial_{i}")
            for t in range(1, len(T)):
                self._complementarity(soc[t], theta_soc[t] - theta_soc[t + 1] + omega_soc[t], f"stat_soc_{i}_{t}")
            if self.end_soc_mode == "constraint":
                self._complementarity(soc[len(T)], theta_soc[len(T)] - self.mvar["dual_soc_end"][i] + omega_soc[len(T)], f"stat_soc_end_{i}")
            elif self.end_soc_mode == "objective":
                p_end = self.com.energy_price[-1] + self.com.import_tariff
                self._complementarity(soc[len(T)], theta_soc[len(T)] - p_end + omega_soc[len(T)], f"stat_soc_end_{i}")
            else:
                self._complementarity(soc[len(T)], theta_soc[len(T)] + omega_soc[len(T)], f"stat_soc_end_{i}")

        # --- Strong Duality ---
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
        terminal = -p_end * soc[len(T)] if d.has_battery and self.end_soc_mode == "objective" else gp.LinExpr(0)
        self.end_soc_value[i] = terminal
        self.payment[i] = dual_obj - disut - pv_c - terminal

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
            m.addConstr(exc[t] >= imp[t] - c.capacity_limit_kW[t], name=f"comm_cap_{t}")
            m.addSOS(GRB.SOS_TYPE1, [imp[t], exp[t]])
            difference = m.addVar(lb=-GRB.INFINITY, name=f"cap_difference_{t}")
            m.addConstr(difference == imp[t] - c.capacity_limit_kW[t])
            m.addGenConstrMax(exc[t], [difference], constant=0)

        f = 1.0 - c.tariff_discount_ratio
        tau_imp, tau_exp = self.com.import_tariff, self.com.export_tariff

        cost_energy = gp.quicksum(p[t] * (imp[t] - exp[t]) for t in T)
        cost_tariff = gp.quicksum(f * tau_imp * (imp[t] - exc[t]) + tau_imp * exc[t] + f * tau_exp * exp[t] for t in T)
        cost_penalty = c.penalty_DKK_per_kWh * exc.sum()

        self.expr["energy"] = cost_energy
        self.expr["tariff"] = cost_tariff
        self.expr["penalty"] = cost_penalty
        self.expr["dso_cost"] = cost_energy + cost_tariff + cost_penalty

    def _add_pricing_requirements(self) -> None:
        """Add Budget Balance (BB) and Individual Rationality (IR) constraints."""
        m = self.m
        total_payment = gp.quicksum(self.payment.values())

        if self.mode in ("BB", "BB_IR"):
            m.addConstr(total_payment == self.expr["dso_cost"], name="req_budget_balance")
        elif self.mode == "extended":
            delta = self.var["delta_transfer"]
            m.addConstr(total_payment - gp.quicksum(delta.values()) == self.expr["dso_cost"], name="req_budget_balance_ext")

        if self.mode == "BB_IR":
            for i in self.ids:
                total_member_cost = self.payment[i] + self.disutility[i] + self.pv_cost[i] + self.end_soc_value[i]
                m.addConstr(total_member_cost <= self.benchmark_costs[i], name=f"req_ir_{i}")
        elif self.mode == "extended":
            delta = self.var["delta_transfer"]
            for i in self.ids:
                total_member_cost = self.payment[i] + self.disutility[i] + self.pv_cost[i] + self.end_soc_value[i]
                m.addConstr(total_member_cost - delta[i] <= self.benchmark_costs[i], name=f"req_ir_ext_{i}")

    def _set_objective(self) -> None:
        """Objective: Minimize community cost (welfare maximization) + optional penalty on transfer payments."""
        total_disutility = gp.quicksum(self.disutility.values())
        total_pv_cost = gp.quicksum(self.pv_cost.values())

        objective = self.expr["dso_cost"] + total_disutility + total_pv_cost + gp.quicksum(self.end_soc_value.values())
        self.expr["community_cost"] = objective.copy()

        if self.mode == "extended":
            delta = self.var["delta_transfer"]
            objective.add(1e3 * gp.quicksum(delta.values()))

        self.m.setObjective(objective, GRB.MINIMIZE)

    def solve(self) -> Results:
        """Solve MILP model and extract results into standard Results dataclass."""
        self.m.update()
        self.m.optimize()
        status = _status_name(self.m.Status)
        
        if self.m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or (
            self.m.Status == GRB.TIME_LIMIT and self.m.SolCount > 0
        ):
            if self.m.Status == GRB.TIME_LIMIT:
                print(f"  [WARNING] Optimization hit TIME_LIMIT, using best feasible solution found.")
            results = self._extract_results(status)
            results.meta["validation"] = self.validate_solution()
            return results

        raise NoPricingSolution(self.m.Status)

    def validate_solution(self, tolerance=1e-4):
        """Check actual payments and independently solve the Q1 member LPs.

        Compare objective values, not schedules: optimal responses may be nonunique.
        Transfers are lump sums independent of dispatch, so they do not change the LP.
        """
        from .model import CommunityModel
        payment_errors, response_errors, ir_errors = [], [], []
        actual_payments, transfers = 0.0, 0.0
        for d in self.com.members:
            i = d.consumer_id
            prices = [self.var["x_price"][i,t].X for t in self.T]
            paid = sum(prices[t] * self.net[i][t].getValue() for t in self.T)
            private = self.disutility[i].getValue() + self.pv_cost[i].getValue() + self.end_soc_value[i].getValue()
            payment_errors.append(abs(paid - self.payment[i].getValue()))
            follower = CommunityModel(self.com, "C", end_soc_mode=self.end_soc_mode)
            try:
                follower._add_member(d)
                f = gp.quicksum(v[i] for v in follower.mexpr.values())
                follower.m.setObjective(f + gp.quicksum(prices[t]*follower.net[i][t] for t in self.T))
                follower.m.optimize()
                if follower.m.Status != GRB.OPTIMAL:
                    raise RuntimeError(f"Independent follower LP failed for {i}")
                response_errors.append(abs(paid + private - follower.m.ObjVal))
            finally:
                follower.m.dispose()
            delta = self.var["delta_transfer"][i].X if self.mode == "extended" else 0.0
            actual_payments += paid
            transfers += delta
            if self.mode in ("BB_IR", "extended"):
                ir_errors.append(max(0.0, paid + private - delta - self.benchmark_costs[i]))
        meter_errors, exc_errors, overlap = [], [], []
        for t in self.T:
            imp, exp, exc = (self.var[k][t].X for k in ("import", "export", "exceedance"))
            meter_errors.append(abs(imp-exp-sum(self.net[i][t].getValue() for i in self.ids)))
            exc_errors.append(abs(exc-max(0,imp-self.com.contract.capacity_limit_kW[t])))
            overlap.append(min(imp,exp))
        checks = {
            "max_payment_identity_error": max(payment_errors),
            "max_best_response_gap": max(response_errors),
            "max_complementarity_product": max(abs(a.X*b.X) for a,b in self.complementarity_pairs),
            "max_meter_error": max(meter_errors),
            "max_exceedance_error": max(exc_errors),
            "max_simultaneous_grid_flow": max(overlap),
            "max_linear_constraint_violation": self.m.ConstrVio,
            "max_bound_violation": self.m.BoundVio,
        }
        if self.mode != "cap_only":
            checks["budget_error_after_transfers"] = abs(actual_payments-transfers-self.expr["dso_cost"].getValue())
        if ir_errors:
            checks["max_IR_violation_after_transfers"] = max(ir_errors)
        if max(checks.values()) > tolerance:
            raise RuntimeError(f"Pricing solution failed independent checks: {checks}")
        return checks

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

        costs = pd.DataFrame(costs_dict, index=self.ids)
        costs["total_cost_before_transfer"] = costs["disutility"] + costs["pv_cost"] + costs["end_soc_value"] + costs["payment"]
        if "delta_transfer" not in costs:
            costs["delta_transfer"] = 0.0
        costs["total_cost"] = costs["total_cost_before_transfer"] - costs["delta_transfer"]
        if self.benchmark_costs:
            costs["benchmark_cost"] = [self.benchmark_costs.get(i, np.nan) for i in self.ids]
            costs["ir_violation_or_gain"] = costs["total_cost"] - costs["benchmark_cost"]

        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = com.energy_price
        hourly["cap"] = com.contract.capacity_limit_kW
        hourly["posted_price"] = [x_sol[self.ids[0], t].X for t in T]

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
            "total_disutility": sum(costs["disutility"]),
            "total_pv_cost": sum(costs["pv_cost"]),
            "gross_payments_collected": float(costs["payment"].sum()),
            "total_transfers": float(costs["delta_transfer"].sum()),
            "total_payments_collected": float((costs["payment"] - costs["delta_transfer"]).sum()),
            "community_cost": self.expr["community_cost"].getValue(),
            "solver_objective": self.m.ObjVal,
            "transfer_penalty": 1000 * float(costs["delta_transfer"].sum()) if self.mode == "extended" else 0.0,
        }

        return Results(
            question=f"{com.community}_{com.day}_DP_{self.mode}",
            status=status,
            objective=self.expr["community_cost"].getValue(),
            hourly=hourly,
            duals={},
            meta={"setup": "DP", "mode": self.mode, "objective_terms": meta_obj,
                  "solver_objective": self.m.ObjVal, "solver_bound": self.m.ObjBound,
                  "solver_gap": self.m.MIPGap, "runtime": self.m.Runtime,
                  "end_soc_mode": self.end_soc_mode},
            members=members,
            member_costs=costs,
        )

    @staticmethod
    def _put(store: dict, name: str, i: str, item) -> None:
        store.setdefault(name, {})[i] = item
