"""Question 3 model extending FlexibleConsumerModel."""
from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB
from typing import Literal

from .model import FlexibleConsumerModel


class ModelQ3(FlexibleConsumerModel):
    """Question 3: Quadratic disutility with a daily minimum energy requirement constraint.

    Direction: MINIMIZE total cost (procurement cost + disutility penalty)
    """

    def build(self) -> "ModelQ3":
        d, m, T = self.data, self.m, self.T

        self._add_common_variables()

        ref_load = getattr(d, "reference_load", getattr(d, "l_ref", None))
        if ref_load is None:
            ref_load = [0.0] * len(T)

        c_Q = getattr(d, "quadratic_disutility", getattr(d, "c_Q", getattr(d, "c_q", None)))
        if c_Q is None:
            c_Q = 1.0

        E_min = getattr(d, "min_daily_energy_kWh", getattr(d, "E_min", getattr(d, "min_daily_energy", None)))
        if E_min is None:
            E_min = 0.0

        self.expr["disutility"] = gp.quicksum(
            c_Q * (self.var["load"][t] - ref_load[t]) * (self.var["load"][t] - ref_load[t]) for t in T
        )
        self.expr["procurement_cost"] = gp.quicksum(
            self._money_terms(t) for t in T
        )
        m.setObjective(
            self.expr["disutility"] + self.expr["procurement_cost"], GRB.MINIMIZE
        )

        self._add_common_constraints()

        self.con["min_daily_energy"] = m.addConstr(
            E_min - gp.quicksum(self.var["load"][t] for t in T) <= 0,
            name="min_daily_energy"
        )

        m.update()
        return self
    
class ModelQ3Battery(FlexibleConsumerModel):
    """Question 3.(g): Flexible consumer with 
        quadratic disutility, minimum daily energy, and battery storage.

    Parameters
    ----------
    data : InputData
        Input data container with load, PV, grid tariff, and battery specifications.
    end_soc_mode : {'constraint', 'objective', 'free'}, default='constraint'
        Strategy for treating the battery state of charge (SoC) at the end of the horizon (t=24):
        - 'constraint' (Option A): Enforces SoC_24 >= SoC_0 (cyclic operation requirement).
        - 'objective' (Option B): Values the remaining energy SoC_24 in the objective function
          as a revenue term (- final_soc_price * SoC_24).
        - 'free': No end-of-horizon condition or valuation (battery will fully discharge at end of day).
    final_soc_price : float, optional
        Valuation price [DKK/kWh] for remaining stored energy at t=24 when end_soc_mode='objective'.
        If None, defaults to the effective import price at the final hour t=23 (p_23 + tau_imp).
    """

    def __init__(
        self,
        data,
        end_soc_mode: Literal["constraint", "objective", "free"] = "constraint",
        final_soc_price: float | None = None,
    ):
        super().__init__(data)
        if end_soc_mode not in ("constraint", "objective", "free"):
            raise ValueError(
                f"Invalid end_soc_mode: {end_soc_mode!r}. Must be 'constraint', 'objective', or 'free'."
            )
        self.end_soc_mode = end_soc_mode
        self.final_soc_price = final_soc_price

    def build(self) -> "ModelQ3Battery":
        d, m, T = self.data, self.m, self.T

        # 1. Base decision variables (load, import, export, pv)
        self._add_common_variables()

        # 2. Retrieve preferences and requirements
        ref_load = getattr(d, "reference_load", getattr(d, "l_ref", None))
        if ref_load is None:
            ref_load = [0.0] * len(T)

        c_Q = getattr(d, "quadratic_disutility", getattr(d, "c_Q", getattr(d, "c_q", None)))
        if c_Q is None:
            c_Q = 1.0

        E_min = getattr(d, "min_daily_energy_kWh", getattr(d, "E_min", getattr(d, "min_daily_energy", None)))
        if E_min is None:
            E_min = 0.0

        # 3. Retrieve battery parameters
        cap = getattr(d, "battery_capacity_kWh", 0.0) or 0.0
        p_ch_max = getattr(d, "battery_max_charge_kW", 0.0) or 0.0
        p_dis_max = getattr(d, "battery_max_discharge_kW", 0.0) or 0.0
        eta_ch = getattr(d, "battery_charging_efficiency", 1.0) or 1.0
        eta_dis = getattr(d, "battery_discharging_efficiency", 1.0) or 1.0
        soc_0 = getattr(d, "battery_initial_soc_kWh", 0.0) or 0.0
        soc_0 = min(soc_0, cap)

        # 4. Add battery decision variables
        p_ch = m.addVars(T, lb=0.0, ub=p_ch_max, name="p_ch")
        p_dis = m.addVars(T, lb=0.0, ub=p_dis_max, name="p_dis")
        soc = m.addVars(range(len(T) + 1), lb=0.0, ub=cap, name="soc")

        self.var["p_ch"] = p_ch
        self.var["p_dis"] = p_dis
        self.var["soc"] = soc

        # 5. Common bounds and constraints (PV limits, load_min, load_max)
        self.con["pv_max"] = m.addConstrs(
            (self.var["pv"][t] <= d.pv_available[t] for t in T), name="pv_max"
        )
        self.con["load_min"] = m.addConstrs(
            (d.load_min_kWh - self.var["load"][t] <= 0 for t in T), name="load_min"
        )
        self.con["load_max"] = m.addConstrs(
            (self.var["load"][t] <= d.load_max_kWh for t in T), name="load_max"
        )

        # 6. Modified nodal power balance equation with battery charging/discharging
        # Import - Export + PV + Discharge == Load + Charge
        self.con["power_balance"] = m.addConstrs(
            (
                self.var["import"][t] - self.var["export"][t] + self.var["pv"][t] + p_dis[t]
                == self.var["load"][t] + p_ch[t]
                for t in T
            ),
            name="power_balance",
        )

        # 7. Battery SoC dynamics
        self.con["soc_init"] = m.addConstr(soc[0] == soc_0, name="soc_init")
        self.con["soc_dynamic"] = m.addConstrs(
            (
                soc[t + 1] == soc[t] + eta_ch * p_ch[t] - (1.0 / eta_dis) * p_dis[t]
                for t in T
            ),
            name="soc_dynamic",
        )

        # 8. End-of-horizon SoC strategy implementation
        if self.end_soc_mode == "constraint":
            # Option A: Hard constraint forcing end SoC to be at least initial SoC
            self.con["soc_end"] = m.addConstr(soc[len(T)] >= soc_0, name="soc_end_min")

        # 9. Minimum daily energy consumption constraint
        self.con["min_daily_energy"] = m.addConstr(
            E_min - gp.quicksum(self.var["load"][t] for t in T) <= 0,
            name="min_daily_energy",
        )

        # 10. Objective function formulation
        self.expr["disutility"] = gp.quicksum(
            c_Q * (self.var["load"][t] - ref_load[t]) * (self.var["load"][t] - ref_load[t]) for t in T
        )
        self.expr["procurement_cost"] = gp.quicksum(
            self._money_terms(t) for t in T
        )

        # Option B: Add economic valuation of remaining SoC at t=24 to objective function
        if self.end_soc_mode == "objective":
            p_end = (
                self.final_soc_price
                if self.final_soc_price is not None
                else (d.energy_price[-1] + d.import_tariff)
            )
            self.expr["end_soc_valuation"] = -p_end * soc[len(T)]
            obj_expr = (
                self.expr["disutility"]
                + self.expr["procurement_cost"]
                + self.expr["end_soc_valuation"]
            )
        else:
            obj_expr = self.expr["disutility"] + self.expr["procurement_cost"]

        m.setObjective(obj_expr, GRB.MINIMIZE)

        m.update()
        return self
