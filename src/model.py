"""Optimisation model of a single flexible consumer, implemented with gurobipy.

Here, we will use one class for each question, sharing a common skeleton.

    FlexibleConsumerModel   -> for Q1, price-elastic consumption, MAXIMIZES daily surplus
    Q2LinearModel           -> for Q2b, linear disutility |l - ref|, MINIMIZES cost + disutility
    Q2QuadraticModel        -> for Q2c, quadratic disutility, MINIMIZES cost + disutility

Each model has its own optimisation direction, stated in its class docstring: it decides the
sign of every dual value reported below.

Usage (unchanged from the course template):

    model = FlexibleConsumerModel(data)   # 1. hand over the input data
    model.build()                         # 2. declare variables, objective, constraints
    results = model.solve()               # 3. optimise and collect primal AND dual values

Only ``build()`` differs between the questions: ``solve()`` and ``_extract_results()`` work
through the ``self.var`` / ``self.con`` / ``self.expr`` dictionaries and need no change when a
new model is added.

Two conventions make the dual variables easy to read out afterwards:

* Every constraint family is stored in ``self.con`` under a descriptive name, e.g.
  ``self.con["balance"] = self.m.addConstrs(...)``. ``solve()`` then returns the dual value
  (shadow price, Gurobi attribute ``Pi``) of every constraint in ``self.con`` automatically.
* Bounds that you want a dual for must be written as explicit constraints (``addConstr``),
  not as variable bounds (``lb=``/``ub=``). Gurobi reports the sensitivity of a variable
  bound in the reduced cost (``RC``), not in ``Pi``.
* Duals of quadratic constraints (``m.addQConstr``) are read from ``QCPi`` and require ``QCPDual = 1`` (set below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    # one row per hour: variables, prices, hourly duals
    hourly: pd.DataFrame
    # duals of non-hourly constraints
    duals: dict[str, float] = field(default_factory=dict)
    # anything else worth keeping (scenario name, ...)
    meta: dict = field(default_factory=dict)

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write ``hourly`` to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
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


class FlexibleConsumerModel:
    """Consumption problem of one consumer over a 24-hour horizon (Question 1); extend it for Questions 2 and 3.

        Direction: MAXIMIZE the daily surplus
        sum_t [ u_L * load_t - c_PV * pv_t - pi_imp_t * import_t + pi_exp_t * export_t ]
        with effective prices pi_imp_t = p_t + tau_imp and pi_exp_t = p_t - tau_exp
        Equations (1a)-(1f) of the report.

        Subclass and override :meth:`build` (reusing the ``_add_common_*`` helpers) for the
        other questions.
    """

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        # only relevant if you add a quadratic constraint (none is needed in Assignment 1)
        self.m.Params.QCPDual = 1
        self.m.Params.DualReductions = 0
        # decision variables by name
        self.var: dict[str, gp.tupledict | gp.Var] = {}
        # constraints by name (duals read from here)
        self.con: dict[str, gp.tupledict | gp.Constr] = {}
        # objective components by name (values read from here after solve)
        self.expr: dict[str, gp.LinExpr | gp.QuadExpr] = {}

    # ------------------------------------------------------------------ shared helpers
    @property
    def price_import(self) -> np.ndarray:
        """Effective import price pi_imp_t = p_t + tau_imp (DKK/kWh)."""
        return self.data.energy_price + self.data.import_tariff

    @property
    def price_export(self) -> np.ndarray:
        """Effective export price pi_exp_t = p_t - tau_exp (DKK/kWh)."""
        return self.data.energy_price - self.data.export_tariff

    def _add_common_variables(self) -> None:
        """Hourly variables shared by every question: load, pv, import, export (kWh/h).

        ``load`` is declared free (lb=-GRB.INFINITY): its lower bound is the explicit
        ``load_min`` constraint, and it must not exist twice. With the gurobipy default
        ``lb=0`` the same bound would be both a variable bound and a constraint, the
        multiplier could split arbitrarily between RC and Pi (degenerate), and the
        reported dual of ``load_min`` was 0. pv/import/export keep the default lb=0:
        their non-negativity is not duplicated by any explicit constraint.
        """
        m, T = self.m, self.T
        # consumption (bounds as explicit constraints to get duals)
        self.var['load'] = m.addVars(T, lb=-GRB.INFINITY, name='load')
        # PV production
        self.var['pv'] = m.addVars(T, name='pv')
        # grid import
        self.var['import'] = m.addVars(T, name='import')    
        # grid export            
        self.var['export'] = m.addVars(T, name='export')                

    def _add_common_constraints(self) -> None:
        """Constraints shared by every question: balance (1b), PV limit (1c), load bounds (1d)-(1e).

        All bounds are explicit constraints so that their duals are reported.
        """
        d, m, T = self.data, self.m, self.T
        load, pv = self.var['load'], self.var['pv']
        imp, exp = self.var['import'], self.var['export']

        # 1b: hourly energy balance (sources = sinks), arranged sinks - sources == 0; dual = lambda_t
        self.con['balance'] = m.addConstrs(
            (load[t] + exp[t] - pv[t] - imp[t] == 0 for t in T), name='balance')
        # 1c: PV availability; dual = mu_pv_max_t
        self.con['pv_max'] = m.addConstrs(
            (pv[t] <= d.pv_available[t] for t in T), name='pv_max')
        # 1d - 1e hourly load bounds. duals = mu_load_min_t, mu_load_max_t
        # load_min arranged as L_min - load <= 0 so its dual comes out non-negative
        self.con['load_min'] = m.addConstrs(
            (d.load_min_kWh - load[t] <= 0 for t in T), name='load_min')
        self.con['load_max'] = m.addConstrs(
            (load[t] <= d.load_max_kWh for t in T), name='load_max')

    def _money_terms(self, t: int) -> gp.LinExpr:
        """PV cost + import cost - export revenue in hour t (the grid/PV part of every objective)."""
        return (self.data.pv_marginal_cost * self.var['pv'][t]
                + self.price_import[t] * self.var['import'][t]
                - self.price_export[t] * self.var['export'][t])

    # ------------------------------------------------------------------ 2. build

    def build(self) -> "FlexibleConsumerModel":
        """Question 1 model: maximize daily surplus subject to (1b)-(1f)."""
        d, m, T = self.data, self.m, self.T
        if d.consumption_utility is None:
            raise ValueError(f"Case {d.question!r} has no consumption utility: "
                             "use the model class of its own question, not the Question 1 model.")
        self._add_common_variables()
        # (1a): surplus = utility of consumption - (PV cost + import cost - export revenue)
        self.expr["utility"] = gp.quicksum(
            d.consumption_utility * self.var["load"][t] for t in T)
        self.expr["procurement_cost"] = gp.quicksum(
            self._money_terms(t) for t in T)
        m.setObjective(
            self.expr["utility"] - self.expr["procurement_cost"], GRB.MAXIMIZE)
        self._add_common_constraints()
        m.update()
        return self

    # -------------------------------------------------- objective decomposition
    def objective_terms(self) -> dict[str, float]:
        """Optimal value of every named objective component in ``self.expr``
        (requirement of Question 1.(e): procurement cost and total utility)."""
        return {name: e.getValue() for name, e in self.expr.items()}

    # ------------------------------------------------------------------ 3. solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        m = self.m
        m.update()
        if m.NumConstrs == 0 and m.NumQConstrs == 0:
            raise NotImplementedError(
                "The model has no constraints: complete FlexibleConsumerModel.build() in src/model.py first."
            )
        m.optimize()
        status = _status_name(m.Status)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Optimisation ended with status {status}. Check the model (m.computeIIS() helps for infeasibility).")
        return self._extract_results(status)

    # --------------------------------------------------------------- extraction
    def _extract_results(self, status: str) -> Results:
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        # effective prices the consumer actually faces (DKK/kWh), used by the hour-by-hour
        # price-ladder check of Question 1.(f)
        hourly["pi_imp"] = self.price_import
        hourly["pi_exp"] = self.price_export
        hourly["pv_available"] = d.pv_available
        if d.reference_load is not None:
            hourly["reference_load"] = d.reference_load

        # Primal values: every hourly variable family in self.var becomes a column
        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name,
                   v in self.var.items() if isinstance(v, gp.Var)}

        # Dual values: every constraint family in self.con becomes a 'dual_<name>' column or scalar
        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):
                    hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
                else:
                    duals[name] = _dual(c)
            except (AttributeError, gp.GurobiError):
                # No duals available (e.g. model with integer variables)
                pass

        meta: dict = {"scalar_variables": scalars}
        try:
            meta["objective_terms"] = self.objective_terms()
        except Exception:
            pass

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            meta=meta,
        )


class Q2LinearModel(FlexibleConsumerModel):
    """Question 2.(b): linear disutility of the absolute deviation from the reference profile.

    Direction: MINIMIZE  sum_t [ c_L * s_t + c_PV * pv_t + pi_imp_t * import_t - pi_exp_t * export_t ]
    where the auxiliary variable s_t >= |load_t - ref_t| is enforced by the two linear
    constraint families ``dev_up``/``dev_dn`` and pushed down to equality at the optimum
    by its positive objective coefficient (LP reformulation, eqs. (5a)-(5d) of the report).
    """

    def build(self) -> "Q2LinearModel":
        d, m, T = self.data, self.m, self.T
        if d.reference_load is None or d.linear_disutility is None:
            raise ValueError(f"Case {d.question!r} has no reference profile / linear disutility "
                             "coefficient: use the model class of its own question.")
        ref, c_L = d.reference_load, d.linear_disutility

        self._add_common_variables()
        # s_t >= 0: deviation magnitude |load_t - ref_t| after reformulation, kWh/h
        # (default lb=0 kept: this non-negativity is part of the reformulation, no dual needed)
        self.var["deviation"] = m.addVars(T, name="deviation")

        # (5a): linear disutility + procurement cost, minimized
        self.expr["disutility"] = gp.quicksum(
            c_L * self.var["deviation"][t] for t in T)
        self.expr["procurement_cost"] = gp.quicksum(
            self._money_terms(t) for t in T)
        m.setObjective(
            self.expr["disutility"] + self.expr["procurement_cost"], GRB.MINIMIZE)

        self._add_common_constraints()
        # (5b)-(5c): the auxiliary bounds the deviation from above in both signs,
        # arranged <lhs> - s_t <= 0
        self.con["dev_up"] = m.addConstrs(
            (self.var["load"][t] - ref[t] - self.var["deviation"][t] <= 0 for t in T), name="dev_up")
        self.con["dev_dn"] = m.addConstrs(
            (ref[t] - self.var["load"][t] - self.var["deviation"][t] <= 0 for t in T), name="dev_dn")
        m.update()
        return self


class Q2QuadraticModel(FlexibleConsumerModel):
    """Question 2.(c): strictly convex quadratic disutility of the deviation.

    Direction: MINIMIZE  sum_t [ c_Q * (load_t - ref_t)^2 + c_PV * pv_t
                                 + pi_imp_t * import_t - pi_exp_t * export_t ]
    The quadratic term sits in the objective only (a QP: quadratic objective, linear
    constraints), so the duals of the linear constraints are reported as usual.
    """

    def build(self) -> "Q2QuadraticModel":
        d, m, T = self.data, self.m, self.T
        if d.reference_load is None or d.quadratic_disutility is None:
            raise ValueError(f"Case {d.question!r} has no reference profile / quadratic disutility "
                             "coefficient: use the model class of its own question.")
        ref, c_Q = d.reference_load, d.quadratic_disutility

        self._add_common_variables()
        self.expr["disutility"] = gp.quicksum(
            c_Q * (self.var["load"][t] - ref[t]) * (self.var["load"][t] - ref[t]) for t in T)
        self.expr["procurement_cost"] = gp.quicksum(
            self._money_terms(t) for t in T)
        m.setObjective(
            self.expr["disutility"] + self.expr["procurement_cost"], GRB.MINIMIZE)
        self._add_common_constraints()
        m.update()
        return self


#: Which model class solves which data case (used by main.py). Q3 classes to be added.
MODEL_BY_CASE: dict[str, type[FlexibleConsumerModel]] = {
    "Q1_caseA": FlexibleConsumerModel,
    "Q1_caseB": FlexibleConsumerModel,
    "Q2_linear": Q2LinearModel,
    "Q2_quadratic": Q2QuadraticModel,
}


def model_for_case(question: str) -> type[FlexibleConsumerModel]:
    """Model class registered for a data case, with a clear error for future cases."""
    if question in MODEL_BY_CASE:
        return MODEL_BY_CASE[question]
    
    if "battery" in question.lower():
        from .q3_model import ModelQ3Battery
        return ModelQ3Battery

    if question.startswith("Q3"):
        from .q3_model import ModelQ3
        return ModelQ3

    raise NotImplementedError(
        f"No model implemented yet for case {question!r}. "
        f"Implemented: {sorted(MODEL_BY_CASE)} or any 'Q3_*' case."
    )


_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")
