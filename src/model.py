"""Community models of Assignment 2 (Question 1), implemented with gurobipy.

One class, ``CommunityModel``, holds every member of the community in ONE gurobipy model and
covers the three set-ups of Question 1 with two switches (``SETUPS``):

    setup   level         contract   meaning
    "R"     "member"      False      no contract: own meter, full tariffs, no cap
    "I"     "member"      True       individual contracts: own meter, (1 - beta) tariffs, own cap kappa_i * cap, penalty
    "C"     "community"   True       community contract: one meter, (1 - beta) tariffs, one cap, penalty
    "C0"    "community"   False      benchmark, pooling alone: one meter, full tariffs, no cap

Under a contract, a kWh imported below the cap pays (1 - beta) * tau_imp, a kWh above the cap loses
the discount and pays the full tau_imp plus the penalty alpha; a kWh exported pays (1 - beta) * tau_exp.

Direction: MINIMIZE the total cost of the community (= minus its welfare), in DKK
    sum_i [ c_L,i * sum_t dev_i,t + c_PV * sum_t pv_i,t ]     members' disutility and PV cost
  + sum_t p_t * sum_i q_i,t                                   energy at the day-ahead price
  + tariffs (+ penalty)                                       at the level of the meter(s)
with q_i,t = load + charge - pv - discharge the net consumption of member i (kWh/h).
With no constraint coupling the members (level "member"), the joint model decomposes into one
problem per member, so R and I give each member's own optimum.

Usage:

    from src.data_loader import load_community
    model = CommunityModel(load_community("base"), setup="R")   # 1. hand over the data
    model.build()                                               # 2. members, meter(s), objective
    results = model.solve()                                     # 3. primal AND dual values

Conventions (carried over from Assignment 1):

* Variables, constraints and cost terms are stored by name, so that ``_extract_results()``
  reads them all without change when a family is added: member-level families in
  ``self.mvar`` / ``self.mcon`` / ``self.mexpr`` as ``{name: {consumer_id: ...}}``,
  community-level families in ``self.var`` / ``self.con`` / ``self.expr`` as ``{name: ...}``.
* Bounds that we want a dual for are explicit constraints, not variable bounds (Gurobi reports
  the multiplier of a variable bound in ``RC``, not in ``Pi``); ``load`` is therefore free.
* Sign of the duals (``Pi``, MINIMIZE): >= 0 for a ``>=`` constraint, <= 0 for a ``<=``
  constraint; the balances are arranged so that their dual is the (positive) marginal price, DKK/kWh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from .data_loader import CommunityData, InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    # one row per hour, community level: prices, import/export at the meter(s), cap, hourly duals
    hourly: pd.DataFrame
    # duals of non-hourly constraints
    duals: dict[str, float] = field(default_factory=dict)
    # anything else worth keeping (scenario name, ...)
    meta: dict = field(default_factory=dict)
    # one row per (member, hour): the members' variables and hourly duals
    members: pd.DataFrame | None = None
    # one row per member: cost terms (DKK) and duals of the member's daily constraints
    member_costs: pd.DataFrame | None = None

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write the tables to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        if self.members is not None:
            self.members.to_csv(folder / f"{stem}_members.csv")
        if self.member_costs is not None:
            self.member_costs.to_csv(folder / f"{stem}_member_costs.csv", index_label="member")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(
                f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")
            for k, v in self.meta.get("objective_terms", {}).items():
                f.write(f"{k:<16}: {v:.4f} DKK\n")

    def __str__(self) -> str:
        cols = [c for c in self.hourly.columns if not c.startswith("dual_")]
        terms = self.meta.get("objective_terms", {})
        return (
            f"status: {self.status} | objective: {self.objective:.2f} DKK"
            + (" | " + ", ".join(f"{k}={v:.2f}" for k,
               v in terms.items()) if terms else "")
            + "\ndaily totals (kWh): "
            + ", ".join(f"{c}={self.hourly[c].sum():.1f}" for c in cols if c in (
                "import", "export", "load", "pv"))
            + (f"\nduals: {self.duals}" if self.duals else "")
        )


def _dual(c) -> float:
    """Dual value of a linear (``Pi``) or quadratic (``QCPi``, requires QCPDual=1) constraint."""
    return c.QCPi if isinstance(c, gp.QConstr) else c.Pi


#: setup -> (level of the meter(s) the DSO charges, whether the contract applies there)
SETUPS: dict[str, tuple[str, bool]] = {
    "R": ("member", False),
    "I": ("member", True),
    "C": ("community", True),
    "C0": ("community", False),
}


class CommunityModel:
    """R, I or C on one day, every member in one gurobipy model (see the module docstring).

    Parameters
    ----------
    com : CommunityData
        Members, prices and contract (``load_community``); build variations on a copy of it.
    setup : "R", "I", "C" or "C0"
    end_soc_mode : {'constraint', 'objective', 'free'}, default 'constraint'
        End-of-day state of charge of every battery, as in Assignment 1 (Question 3.(g)):
        'constraint' SoC_24 >= SoC_0 (cyclic); 'objective' the energy left at t = 24 is valued at
        ``final_soc_price``; 'free' no end-of-day condition.
    final_soc_price : float, optional
        DKK/kWh for 'objective'; default the full import price of the last hour, p_23 + tau_imp.
    """

    def __init__(
        self,
        com: CommunityData,
        setup: str = "R",
        end_soc_mode: Literal["constraint", "objective", "free"] = "constraint",
        final_soc_price: float | None = None,
        verbose: bool = False,
    ):
        if setup not in SETUPS:
            raise ValueError(f"Unknown setup {setup!r}: one of {list(SETUPS)}.")
        if end_soc_mode not in ("constraint", "objective", "free"):
            raise ValueError(
                f"Invalid end_soc_mode: {end_soc_mode!r}. Must be 'constraint', 'objective', or 'free'.")
        self.com = com
        self.setup = setup
        self.level, self.contract = SETUPS[setup]
        self.end_soc_mode = end_soc_mode
        self.final_soc_price = final_soc_price
        self.T = range(com.members[0].n_hours)
        self.ids = com.member_ids
        self.m = gp.Model(f"community_{setup}")
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.DualReductions = 0
        # member level: {name: {consumer_id: tupledict over hours | Constr | LinExpr}}
        self.mvar: dict[str, dict] = {}
        self.mcon: dict[str, dict] = {}
        self.mexpr: dict[str, dict] = {}    # cost terms of every member, DKK
        # net consumption q_i,t of every member: {consumer_id: {hour: LinExpr}}, kWh/h
        self.net: dict[str, dict[int, gp.LinExpr]] = {}
        # community level: {name: tupledict over hours | Constr | LinExpr}
        self.var: dict[str, gp.tupledict] = {}
        self.con: dict[str, gp.tupledict] = {}
        self.expr: dict[str, gp.LinExpr] = {}   # every cost term summed over the members, DKK

    # ------------------------------------------------------------------ shared helpers
    @staticmethod
    def _put(store: dict, name: str, i: str, item) -> None:
        """Store a member-level item: ``store[name][consumer_id] = item``."""
        store.setdefault(name, {})[i] = item

    def _tariffs(self) -> tuple[float, float, float]:
        """Grid tariffs in DKK/kWh: (import below the cap, import above the cap, export).

        No contract: the full tariffs. Under a contract: (1 - beta) times the tariffs, except the import
        above the cap, which loses the discount (full tau_imp; the penalty alpha comes on top).
        """
        f = 1.0 - self.com.contract.tariff_discount_ratio if self.contract else 1.0
        tau_imp, tau_exp = self.com.import_tariff, self.com.export_tariff
        return f * tau_imp, tau_imp, f * tau_exp

    # ------------------------------------------------------------------ 2. build
    def build(self) -> "CommunityModel":
        """Every member's block, the meter(s) of the set-up, and the objective (MINIMIZE total cost)."""
        for d in self.com.members:
            self._add_member(d)
        if self.level == "member":
            self._add_member_meters()
        else:
            self._add_community_meter()
        for name, per_member in self.mexpr.items():
            self.expr[name] = gp.quicksum(per_member.values())
        self.m.setObjective(gp.quicksum(self.expr.values()), GRB.MINIMIZE)
        self.m.update()
        return self

    def _add_member(self, d: InputData) -> None:
        """Variables, constraints and cost terms of one member, identical in R, I and C (kWh/h, DKK).

        load l_t (free; hourly bounds load_lower/upper as constraints), deviation dev_t >= |l_t - ref_t|
        (linear disutility c_L * dev_t), daily minimum sum_t l_t >= E_min, PV 0 <= pv_t <= PV_avail_t
        (members with PV; cost c_PV * pv_t), battery charge / discharge / state of charge (members with
        a battery). Sets ``self.net[i][t]`` = l_t + charge_t - pv_t - discharge_t.
        """
        m, T, i = self.m, self.T, d.consumer_id
        if d.reference_load is None or d.linear_disutility is None or d.min_daily_energy_kWh is None:
            raise ValueError(f"Member {i!r} has no reference profile / linear disutility / daily minimum.")
        ref = d.reference_load

        # --- flexible load: free variable, its hourly bounds as explicit constraints (duals in Pi)
        load = m.addVars(T, lb=-GRB.INFINITY, name=f"load_{i}")
        self._put(self.mvar, "load", i, load)
        self._put(self.mcon, "load_lower", i, m.addConstrs(
            (load[t] >= d.load_lower_kWh[t] for t in T), name=f"load_lower_{i}"))
        self._put(self.mcon, "load_upper", i, m.addConstrs(
            (load[t] <= d.load_upper_kWh[t] for t in T), name=f"load_upper_{i}"))
        # dev_t >= |load_t - ref_t|, pushed down to equality by its positive cost (LP reformulation)
        dev = m.addVars(T, name=f"deviation_{i}")
        self._put(self.mvar, "deviation", i, dev)
        self._put(self.mcon, "dev_up", i, m.addConstrs(
            (dev[t] >= load[t] - ref[t] for t in T), name=f"dev_up_{i}"))
        self._put(self.mcon, "dev_dn", i, m.addConstrs(
            (dev[t] >= ref[t] - load[t] for t in T), name=f"dev_dn_{i}"))
        # daily minimum (= the daily reference energy); dual mu_i >= 0: value of a kWh of daily need
        self._put(self.mcon, "min_daily_energy", i, m.addConstr(
            load.sum() >= d.min_daily_energy_kWh, name=f"min_daily_energy_{i}"))
        self._put(self.mexpr, "disutility", i, d.linear_disutility * dev.sum())
        net = {t: gp.LinExpr(load[t]) for t in T}

        # --- PV (members with PV only)
        if d.has_pv:
            pv = m.addVars(T, name=f"pv_{i}")
            self._put(self.mvar, "pv", i, pv)
            self._put(self.mcon, "pv_max", i, m.addConstrs(
                (pv[t] <= d.pv_available[t] for t in T), name=f"pv_max_{i}"))
            self._put(self.mexpr, "pv_cost", i, d.pv_marginal_cost * pv.sum())
            for t in T:
                net[t].add(pv[t], -1.0)

        # --- battery (members with a battery only), as in Assignment 1 Question 3.(g)
        if d.has_battery:
            soc_0 = d.battery_initial_soc_kWh
            p_ch = m.addVars(T, ub=d.battery_max_charge_kW, name=f"p_ch_{i}")
            p_dis = m.addVars(T, ub=d.battery_max_discharge_kW, name=f"p_dis_{i}")
            # state of charge at the start of hour t, t = 0..24 (24 = end of the day)
            soc = m.addVars(range(len(T) + 1), ub=d.battery_capacity_kWh, name=f"soc_{i}")
            for name, v in (("p_ch", p_ch), ("p_dis", p_dis), ("soc", soc)):
                self._put(self.mvar, name, i, v)
            self._put(self.mcon, "soc_init", i, m.addConstr(soc[0] == soc_0, name=f"soc_init_{i}"))
            self._put(self.mcon, "soc_dynamic", i, m.addConstrs(
                (soc[t + 1] == soc[t] + d.battery_charging_efficiency * p_ch[t]
                 - p_dis[t] / d.battery_discharging_efficiency for t in T), name=f"soc_dynamic_{i}"))
            if self.end_soc_mode == "constraint":     # cyclic: end the day with at least the initial charge
                self._put(self.mcon, "soc_end", i, m.addConstr(soc[len(T)] >= soc_0, name=f"soc_end_{i}"))
            elif self.end_soc_mode == "objective":    # the energy left at t = 24 is worth final_soc_price
                p_end = (self.final_soc_price if self.final_soc_price is not None
                         else d.energy_price[-1] + d.import_tariff)
                self._put(self.mexpr, "end_soc_value", i, -p_end * soc[len(T)])
            for t in T:
                net[t].add(p_ch[t] - p_dis[t])

        self.net[i] = net

    def _add_member_meters(self) -> None:
        """Own meter of every member (R, I): import and export at the member's connection, kWh/h.

        balance_i,t: import - export - q_i,t == 0, arranged so that its dual is the member's own
        marginal price of energy in hour t (DKK/kWh). Energy at p_t and tariffs on the member's own flows.
        Under a contract (I): exceedance e_i,t >= import - kappa_i * cap_t, e >= 0 (``cap``; its dual >= 0 is
        the value of one more kW of the member's cap), the tariffs of ``_tariffs`` and the penalty alpha * e.
        """
        m, T, p = self.m, self.T, self.com.energy_price
        tau_below, tau_above, tau_exp = self._tariffs()
        for i in self.ids:
            imp = m.addVars(T, name=f"import_{i}")
            exp = m.addVars(T, name=f"export_{i}")
            self._put(self.mvar, "import", i, imp)
            self._put(self.mvar, "export", i, exp)
            self._put(self.mcon, "balance", i, m.addConstrs(
                (imp[t] - exp[t] - self.net[i][t] == 0 for t in T), name=f"balance_{i}"))
            self._put(self.mexpr, "energy", i, gp.quicksum(p[t] * (imp[t] - exp[t]) for t in T))
            if self.contract:
                # e_i,t = max(0, import - kappa_i * cap_t) at the optimum: a kWh of e costs
                # beta * tau_imp + alpha >= 0 more than a kWh below the cap
                cap_i = self.com.contract.individual_capacity_limit_kW[i]
                e = m.addVars(T, name=f"exceedance_{i}")
                self._put(self.mvar, "exceedance", i, e)
                self._put(self.mcon, "cap", i, m.addConstrs(
                    (e[t] >= imp[t] - cap_i[t] for t in T), name=f"cap_{i}"))
                self._put(self.mexpr, "tariff", i, gp.quicksum(
                    tau_below * (imp[t] - e[t]) + tau_above * e[t] + tau_exp * exp[t] for t in T))
                self._put(self.mexpr, "penalty", i, self.com.contract.penalty_DKK_per_kWh * e.sum())
            else:
                self._put(self.mexpr, "tariff", i, gp.quicksum(tau_below * imp[t] + tau_exp * exp[t] for t in T))

    def _add_community_meter(self) -> None:
        """One meter for the community (C, C0): import and export at the connection point, kWh/h.

        balance_t: import - export - sum_i q_i,t == 0, arranged so that its dual lambda_t is the community's
        marginal price of energy in hour t, the internal price of the dual settlement (DKK/kWh). Energy at
        p_t and tariffs on the metered exchange; the members exchange freely behind the meter. Under a
        contract (C): exceedance E_t >= import - cap_t, E >= 0 (``cap``; its dual >= 0 is the value of one
        more kW of the community's cap), the tariffs of ``_tariffs`` and the penalty alpha * E.
        """
        m, T, p = self.m, self.T, self.com.energy_price
        tau_below, tau_above, tau_exp = self._tariffs()
        imp = self.var["import"] = m.addVars(T, name="import")
        exp = self.var["export"] = m.addVars(T, name="export")
        self.con["balance"] = m.addConstrs(
            (imp[t] - exp[t] - gp.quicksum(self.net[i][t] for i in self.ids) == 0 for t in T), name="balance")
        self.expr["energy"] = gp.quicksum(p[t] * (imp[t] - exp[t]) for t in T)
        if self.contract:
            # E_t = max(0, import - cap_t) at the optimum: a kWh of E costs beta * tau_imp + alpha >= 0 more
            cap = self.com.contract.capacity_limit_kW
            e = self.var["exceedance"] = m.addVars(T, name="exceedance")
            self.con["cap"] = m.addConstrs((e[t] >= imp[t] - cap[t] for t in T), name="cap")
            self.expr["tariff"] = gp.quicksum(
                tau_below * (imp[t] - e[t]) + tau_above * e[t] + tau_exp * exp[t] for t in T)
            self.expr["penalty"] = self.com.contract.penalty_DKK_per_kWh * e.sum()
        else:
            self.expr["tariff"] = gp.quicksum(tau_below * imp[t] + tau_exp * exp[t] for t in T)

    # -------------------------------------------------- objective decomposition
    def objective_terms(self) -> dict[str, float]:
        """Optimal value of every cost term summed over the members (DKK)."""
        return {name: e.getValue() for name, e in self.expr.items()}

    # ------------------------------------------------------------------ 3. solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        m = self.m
        m.update()
        if m.NumConstrs == 0:
            raise RuntimeError("The model is empty: call build() before solve().")
        m.optimize()
        status = _status_name(m.Status)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Optimisation ended with status {status}. Check the model (m.computeIIS() helps for infeasibility).")
        return self._extract_results(status)

    # --------------------------------------------------------------- extraction
    def _extract_results(self, status: str) -> Results:
        T, com = list(self.T), self.com

        # --- one table per member: every hourly family of self.mvar / self.mcon becomes a column
        frames = {}
        for i in self.ids:
            d = com.member(i)
            df = pd.DataFrame(index=pd.Index(T, name="hour"))
            df["reference_load"] = d.reference_load
            df["pv_available"] = d.pv_available
            for name, per_member in self.mvar.items():
                if i in per_member:
                    v = per_member[i]
                    shift = len(v) - len(T)     # families indexed 0..24 (soc): value at the end of the hour
                    df[name] = [v[t + shift].X for t in T]
            df["net"] = [self.net[i][t].getValue() for t in T]
            for name, per_member in self.mcon.items():
                if isinstance(per_member.get(i), gp.tupledict):
                    df[f"dual_{name}"] = [_dual(per_member[i][t]) for t in T]
            frames[i] = df
        members = pd.concat(frames, names=["member"])

        # --- one row per member: cost terms, then the duals of its daily (scalar) constraints
        costs = pd.DataFrame(
            {name: {i: e.getValue() for i, e in per_member.items()} for name, per_member in self.mexpr.items()},
            index=self.ids).fillna(0.0)
        costs["total"] = costs.sum(axis=1)
        costs.insert(0, "archetype", [com.member(i).archetype for i in self.ids])
        for name, per_member in self.mcon.items():
            scalars = {i: _dual(c) for i, c in per_member.items() if isinstance(c, gp.Constr)}
            if scalars:
                costs[f"dual_{name}"] = pd.Series(scalars)

        # --- community level, hour by hour: what the DSO meters against the cap
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = com.energy_price
        hourly["cap"] = com.contract.capacity_limit_kW
        for col in ("reference_load", "load", "pv", "p_ch", "p_dis"):
            if col in members:
                hourly[col] = members[col].groupby(level="hour").sum()
        if self.level == "member":      # the DSO meters every member: the sum of their imports and exports
            hourly["import"] = members["import"].groupby(level="hour").sum()
            hourly["export"] = members["export"].groupby(level="hour").sum()
        # what the DSO bills above the cap(s): the members' own caps (I) or the community's cap (C)
        if "exceedance" in members:
            hourly["exceedance_billed"] = members["exceedance"].groupby(level="hour").sum()
        for name, v in self.var.items():    # level "community": import, export (and exceedance) at the meter
            hourly["exceedance_billed" if name == "exceedance" else name] = [v[t].X for t in T]
        # congestion of the feeder in every setup: import at the connection point above the cap
        hourly["exceedance"] = (hourly["import"] - hourly["cap"]).clip(lower=0.0)
        duals: dict[str, float] = {}
        for name, c in self.con.items():
            if isinstance(c, gp.tupledict):
                hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
            else:
                duals[name] = _dual(c)

        return Results(
            question=f"{com.community}_{com.day}_{self.setup}",
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            meta={"setup": self.setup, "level": self.level, "contract": self.contract,
                  "end_soc_mode": self.end_soc_mode, "objective_terms": self.objective_terms()},
            members=members,
            member_costs=costs,
        )


_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")
