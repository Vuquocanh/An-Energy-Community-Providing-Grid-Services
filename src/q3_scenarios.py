"""Question 3 -- scenarios of the members' preferences, as communities our model classes can solve.

The starter module ``scenario_generation`` draws the scaling parameters theta (one per member and uncertain
quantity, scaled Beta on [lo, hi]) and writes the perturbed parameters into appliance dicts:
    profile : rho_t -> rho_t ** theta           (the base study, U2)
    cL      : c^L   -> c^L * theta              (a variation, U1)
    daily minimum = max_load * sum_t rho_t ** theta  (follows the profile, not a separate quantity)
This module turns every joint scenario into a fresh :class:`CommunityData` (members with the scenario's reference
profile, hourly bounds and daily minimum; the contract unchanged, because the manager signs it before the
preferences are revealed) and checks the result.

    draw_scenarios(com, n, seed, quantities=("profile",))  -> list[Scenario]   n equiprobable joint scenarios
    expected_scenario(com)                                  -> Scenario         theta = 1 everywhere (the baseline, DP)
    scenario_sets(com, n_in=(5, 10), n_oos=100, ...)        -> dict             in-sample sets + the out-of-sample set
    validate_scenarios(com, scenarios)                      -> dict             the checks (raises on failure)
    theta_table(scenarios)                                  -> DataFrame        one row per (scenario, member)

Seeds: every experiment states them. By default the in-sample sets use seed 46750 + n and the out-of-sample set
seed 46750 + 1000, so the same out-of-sample scenarios are used for every model (common random numbers) and the
in-sample sets of different sizes are distinct draws (not nested).

Run ``python -m src.q3_scenarios`` from the repository root for the checks on the base community, day d1.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import scenario_generation as sg
from .data_loader import CommunityData, InputData

SEED_BASE = 46750


@dataclass
class Scenario:
    """One joint scenario of the community: its probability, the theta of every member and the community itself."""
    id: int
    probability: float
    theta_profile: dict[str, float]
    theta_cL: dict[str, float]
    community: CommunityData
    seed: int | None = None
    label: str = ""
    meta: dict = field(default_factory=dict, repr=False)

    @property
    def members(self) -> list[InputData]:
        return self.community.members


# ----------------------------------------------------------------------------------------------------------------------
# building one community per scenario
# ----------------------------------------------------------------------------------------------------------------------
def _day_catalogue(com: CommunityData) -> dict:
    """The appliance catalogue with the profiles of ``com.day`` written in (what the loader used for ``com``)."""
    raw = com.raw
    bus0 = next(b for b in raw["bus_params"] if b["bus_id"] == com.connection_bus)
    app_day, _ = sg.day_data(raw["appliance_params"], bus0, raw["weekly_profiles"], com.day)
    return app_day


def _rules(com: CommunityData) -> dict:
    """The hyper-parameters of the uncertainty (``preference_uncertainty_params.json``; not read by the loader)."""
    return com.raw.get("preference_uncertainty_params") or sg.load_json("preference_uncertainty_params.json")


def _load_dict(member_appliances: dict) -> dict:
    """The flexible-load appliance dict of one member (the only appliance the scenarios perturb)."""
    loads = [a for a in member_appliances.values() if "load_id" in a]
    if len(loads) != 1:
        raise ValueError(f"expected exactly one flexible load per member, found {len(loads)}")
    return loads[0]


def _member_in_scenario(d: InputData, load: dict) -> InputData:
    """A copy of member ``d`` with the scenario's reference profile, daily minimum, disutility slope and hourly bounds.
    Everything else (prices, tariffs, PV, battery) is the member's own and does not change."""
    m = copy.deepcopy(d)
    rho = np.asarray(load["reference_load_capacity_factor"], dtype=float)
    if rho.shape != (m.n_hours,) or (rho < 0).any() or (rho > 1).any():
        raise ValueError(f"{d.consumer_id}: reference_load_capacity_factor must be 24 values in [0, 1]")
    m.reference_load = m.load_max_kWh * rho
    m.min_daily_energy_kWh = float(load["min_total_energy_per_day_kWh"])
    m.linear_disutility = float(load["linear_disutility_DKK_per_kWh"])
    dmax = m.max_hourly_deviation_kWh
    if dmax is not None:   # the hourly bounds follow the perturbed reference (assignment, Table 10)
        m.load_upper_kWh = np.minimum(m.load_max_kWh, m.reference_load + dmax)
        m.load_lower_kWh = np.maximum(m.load_min_kWh, m.reference_load - dmax)
    m.raw = dict(m.raw, load=load)
    return m


def _community_in_scenario(com: CommunityData, members_dicts: dict, tag: str) -> CommunityData:
    """A shallow copy of ``com`` whose members are rebuilt from the scenario's appliance dicts. The contract is shared
    (not copied): it is signed before the preferences are revealed and is the same in every scenario."""
    members = [_member_in_scenario(d, _load_dict(members_dicts[d.consumer_id])) for d in com.members]
    c = copy.copy(com)
    c.members = members
    c.community = f"{com.community}_{tag}"
    return c


def expected_scenario(com: CommunityData) -> Scenario:
    """The deterministic baseline: theta = 1 for every member and quantity (the catalogue parameters of the day)."""
    app_day = _day_catalogue(com)
    members_dicts = sg.expected_preference_scenario(app_day, com.raw["params"])
    ids = com.member_ids
    return Scenario(id=-1, probability=1.0, theta_profile={i: 1.0 for i in ids}, theta_cL={i: 1.0 for i in ids},
                    community=_community_in_scenario(com, members_dicts, "expected"), label="expected")


def draw_scenarios(com: CommunityData, n: int, seed: int, quantities: tuple[str, ...] = ("profile",),
                   spec: dict | None = None, mode: str = "aligned", label: str = "") -> list[Scenario]:
    """``n`` equiprobable joint scenarios of ``com`` in which ``quantities`` are uncertain, drawn with the rules of
    ``preference_uncertainty_params.json`` (or ``spec`` from :func:`scenario_generation.rules_to_spec`, tuned with
    :func:`scenario_generation.set_rule`). Every scenario is a community ready for ``CommunityModel`` /
    ``DynamicPricingModel``."""
    raw = com.raw
    joint = sg.draw_preference_scenarios(_day_catalogue(com), raw["params"], _rules(com),
                                         n=n, seed=seed, quantities=quantities, spec=spec, mode=mode)
    out = []
    for s in joint:
        out.append(Scenario(
            id=int(s["id"]), probability=float(s["probability"]),
            theta_profile=dict(s["theta"]["profile"]), theta_cL=dict(s["theta"]["cL"]),
            community=_community_in_scenario(com, s["members"], f"{label or 'scen'}{s['id']}"),
            seed=seed, label=label, meta={"index": s["index"], "quantities": tuple(quantities), "mode": mode},
        ))
    return out


def scenario_sets(com: CommunityData, n_in: tuple[int, ...] = (5, 10), n_oos: int = 100,
                  quantities: tuple[str, ...] = ("profile",), seed_in: int | None = None,
                  seed_oos: int | None = None, spec: dict | None = None) -> dict[str, list[Scenario]]:
    """The scenario sets of one experiment: one in-sample set per size in ``n_in`` and one out-of-sample set shared
    by every model (common random numbers). Keys ``"in_5"``, ``"in_10"``, ``"oos"``; the seed of each set is stored
    in its scenarios (``Scenario.seed``) and must be reported."""
    sets = {}
    for n in n_in:
        seed = (SEED_BASE + n) if seed_in is None else (seed_in + n)
        sets[f"in_{n}"] = draw_scenarios(com, n, seed, quantities, spec=copy.deepcopy(spec) if spec else None, label=f"in{n}_")
    seed = (SEED_BASE + 1000) if seed_oos is None else seed_oos
    sets["oos"] = draw_scenarios(com, n_oos, seed, quantities, spec=copy.deepcopy(spec) if spec else None, label="oos_")
    return sets


# ----------------------------------------------------------------------------------------------------------------------
# checks and tables
# ----------------------------------------------------------------------------------------------------------------------
def theta_table(scenarios: list[Scenario]) -> pd.DataFrame:
    """One row per (scenario, member): theta of each quantity, the member's daily reference energy and E_min."""
    rows = []
    for s in scenarios:
        for d in s.members:
            rows.append({"scenario": s.id, "probability": s.probability, "member": d.consumer_id,
                         "theta_profile": s.theta_profile[d.consumer_id], "theta_cL": s.theta_cL[d.consumer_id],
                         "ref_energy_kWh": float(d.reference_load.sum()), "E_min_kWh": d.min_daily_energy_kWh,
                         "ref_peak_kW": float(d.reference_load.max()), "c_L": d.linear_disutility})
    return pd.DataFrame(rows)


def validate_scenarios(com: CommunityData, scenarios: list[Scenario], tol: float = 1e-9) -> dict[str, float]:
    """Checks (raise on failure):
    1. theta = 1 reproduces the base community exactly (reference, bounds, E_min, c^L of every member);
    2. in every scenario E_min = sum_t reference_t and the bounds are max(l_min, ref - D) / min(l_max, ref + D);
    3. every theta lies within the hard bounds [lo, hi] of the rules file and the probabilities sum to one;
    4. the scenario's reference is rho ** theta of the base reference (the rule of the file)."""
    rules = _rules(com)["quantities"]
    base = {d.consumer_id: d for d in com.members}

    # 1. the expected scenario is the base case
    exp = expected_scenario(com)
    err_expected = 0.0
    for d in exp.members:
        b = base[d.consumer_id]
        for a, bb in ((d.reference_load, b.reference_load), (d.load_upper_kWh, b.load_upper_kWh),
                      (d.load_lower_kWh, b.load_lower_kWh)):
            err_expected = max(err_expected, float(np.abs(a - bb).max()))
        err_expected = max(err_expected, abs(d.min_daily_energy_kWh - b.min_daily_energy_kWh),
                           abs(d.linear_disutility - b.linear_disutility))
    if err_expected > tol:
        raise AssertionError(f"theta = 1 does not reproduce the base community (max error {err_expected:.2e})")

    # 2-4. every drawn scenario
    err_emin = err_bounds = err_rule = 0.0
    theta_out = 0
    for s in scenarios:
        for d in s.members:
            b = base[d.consumer_id]
            err_emin = max(err_emin, abs(d.min_daily_energy_kWh - float(d.reference_load.sum())))
            up = np.minimum(d.load_max_kWh, d.reference_load + d.max_hourly_deviation_kWh)
            lo = np.maximum(d.load_min_kWh, d.reference_load - d.max_hourly_deviation_kWh)
            err_bounds = max(err_bounds, float(np.abs(up - d.load_upper_kWh).max()), float(np.abs(lo - d.load_lower_kWh).max()))
            th_p, th_c = s.theta_profile[d.consumer_id], s.theta_cL[d.consumer_id]
            rho_base = b.reference_load / b.load_max_kWh
            err_rule = max(err_rule, float(np.abs(b.load_max_kWh * rho_base ** th_p - d.reference_load).max()),
                           abs(b.linear_disutility * th_c - d.linear_disutility))
            for q, th in (("profile", th_p), ("cL", th_c)):
                r = rules[q]
                if not (r["lo"] - tol <= th <= r["hi"] + tol) and th != 1.0:
                    theta_out += 1
    prob = sum(s.probability for s in scenarios)
    checks = {"max_error_expected_vs_base": err_expected, "max_error_Emin_vs_ref_energy": err_emin,
              "max_error_bounds": err_bounds, "max_error_scaling_rule": err_rule, "n_theta_outside_bounds": theta_out,
              "sum_probability": prob, "n_scenarios": len(scenarios)}
    if max(err_emin, err_bounds, err_rule) > 1e-7 or theta_out or abs(prob - 1.0) > 1e-9:
        raise AssertionError(f"scenario checks failed: {checks}")
    return checks


if __name__ == "__main__":
    from .data_loader import load_community
    com = load_community("base", day="d1")
    sets = scenario_sets(com, n_in=(5, 10), n_oos=100)
    for name, scen in sets.items():
        checks = validate_scenarios(com, scen)
        tab = theta_table(scen)
        print(f"{name:6s}: {len(scen):3d} scenarios, seed {scen[0].seed} | theta_profile mean {tab.theta_profile.mean():.3f} "
              f"[{tab.theta_profile.min():.3f}, {tab.theta_profile.max():.3f}] | E_min mean {tab.E_min_kWh.mean():.2f} kWh "
              f"[{tab.E_min_kWh.min():.2f}, {tab.E_min_kWh.max():.2f}] | checks OK ({checks['max_error_scaling_rule']:.1e})")
    base_E = sum(d.min_daily_energy_kWh for d in com.members)
    tot = [sum(d.min_daily_energy_kWh for d in s.members) for s in sets["oos"]]
    print(f"base community daily reference energy {base_E:.1f} kWh; out-of-sample {np.mean(tot):.1f} "
          f"[{np.min(tot):.1f}, {np.max(tot):.1f}] kWh")
    print(theta_table(sets["in_5"]).query("member in ['C1', 'C4_PVB']").to_string(index=False))
