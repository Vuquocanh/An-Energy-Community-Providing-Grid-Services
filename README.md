# 46750 - Assignment 2: An Energy Community as a Flexibility Provider

Starter package for **Group Assignment 2** of *46750 - Optimization in Modern Power Systems* (DTU).
It contains the input data of every question (in `data/`), the Assignment 1 data loader extended to a
community and the scenario-generation module of Question 3 (in `src/`), and the instructions below.
The assignment text (*Assignment 2 - Instructions* in DTU Learn) is the reference for the questions; this
README documents the data and the code conventions.

**Getting your own copy.** Download the package from DTU Learn (or use the template repository, if one is
announced, with **Use this template -> Create a new repository**). Using the Python code is recommended but
not required; the input data in `data/` must be used as provided, and the scenarios of Question 3
must be drawn with the rules of the uncertainty file (the module in `src/` implements them exactly).

**Submitting your code.** As in Assignment 1: attach your complete project code as a single `.zip` to your
submission in DTU Learn, or give the link to your group's repository on the front page of your report
(accessible to the graders, not modified after the deadline). The README must let the graders reproduce
every number and figure of the report with one command per experiment; the scenarios you generate and any
data you add are part of the code deliverable and must be documented here.

## 1. Setup

### 1.1 Python environment

Use one of the two options (both install the same packages).

**Option A - pip and venv**
```bash
python -m venv venv
source venv/bin/activate            # macOS / Linux
# venv\Scripts\activate.bat         # Windows cmd
# venv\Scripts\Activate.ps1         # Windows PowerShell
pip install -r requirements.txt
```

**Option B - conda**
```bash
conda env create -f environment.yaml
conda activate 46750-a2
```

### 1.2 Gurobi licence

The code uses [Gurobi](https://www.gurobi.com) through the `gurobipy` package. **The restricted licence
shipped with the package (2000 variables/constraints) is not large enough for Questions 2 and 3** - sixteen
members' KKT systems with binaries, replicated per scenario. Request the
free [academic licence](https://www.gurobi.com/academia/academic-program-and-licenses/) with your DTU
e-mail and activate it with `grbgetkey <your-key>` (on the DTU network or VPN) before you start. Every member's
problem is a linear program, so the single-level reformulations of Questions 2-3 are mixed-integer linear programs (KKT
conditions, big-M or SOS1 complementarity, strong-duality linearization of the payments); use a time limit and a MIP gap.

### 1.3 Check that everything works
```bash
python main.py                  # loads the base case (16 members, day d1) and prints a summary
python main.py --members        # ... plus the Assignment 1 summary of every member
python main.py --demo           # ... plus the scenario-generation demonstration (Question 3)
python main.py --verify         # ... plus the validation checks of the Question 1 models (Section 6.4): all must PASS
```

## 2. Repository structure

```
main.py                        Entry point: summary of the community, one flag per experiment (Section 7)
src/
  data_loader.py               load_community("base", day="d1") -> CommunityData:
                                 .members     one Assignment 1 InputData per member (same attribute names as in Assignment 1)
                                 .contract    ContractData: cap, penalty, discount, individual caps and shares of model I
  model.py                     CommunityModel: the models R, I and C of Question 1 as one class, and Results (Section 6)
  q1.py                        Question 1 experiments: metrics against R, energy shifted, dual settlement of C, contract variations and sweeps (run_q1_base, run_q1_sensitivity)
  plotting.py                  Figures: import against the cap, shift heat map, internal price, members' gains, sensitivity, input data of one day
  validation.py                Validation checks of the Question 1 models (python main.py --verify)
  scenario_generation.py       Question 3: independent preference scenarios per member, merged into joint scenarios (numpy only)
data/
  appliance_params.json        SHARED catalogue (Assignment 1 format): the loads FL_L1-FL_L4 (linear disutility 0.25-2.0 DKK/kWh), PV_01_S, BAT_01_S
  bus_params.json              SHARED connection point: day-1 prices and the grid tariffs (unchanged from Assignment 1)
  weekly_profiles.json         Seven days d1-d7 of prices, PV availability ratios and reference load factors
  params_community_base.json   The base-case community (16 members: 4 slopes x 4 asset classes) - the Assignment 1 case-file format (the only community shipped)
  dso_contract_params.json     The capacity limitation contract and the individual contracts of the benchmark model I
  preference_uncertainty_params.json   Question 3: uncertainty about the members' preference parameters
  community_base_case.png/.svg, setup_R.png/.svg, setup_I.png/.svg   Figures 1-3 of the assignment (reference, individual contracts, community contract; schematic)
  fig_q3_scenarios.png/.svg    Figure 4 of the assignment (how the scenarios of Question 3 are generated)
results/                       Written by the code (git-ignored): q1_base/, data/ (Section 7)
requirements.txt, environment.yaml, .gitignore
```

The models are documented in Section 6, the commands that reproduce the results in Section 7.

## 3. How to use the code

**From a notebook or your own script** (run from the repository root):
```python
from src.data_loader import load_community
from src import scenario_generation as sg

com = load_community("base")                       # day d1
print(com.summary())
for m in com.members:                              # Assignment 1 InputData objects
    print(m.consumer_id, m.archetype, m.linear_disutility, m.min_daily_energy_kWh, m.pv_max_kW, m.battery_capacity_kWh)

cap   = com.contract.capacity_limit_kW             # 24 values, kW, at the connection point
beta  = com.contract.tariff_discount_ratio         # 0.75 in the base case
alpha = com.contract.penalty_DKK_per_kWh           # 10 in the base case
cap_i = com.contract.individual_capacity_limit_kW  # {consumer_id: 24 values}  (benchmark model I)

com_d4 = load_community("base", day="d4")          # another day of the week (prices, PV, reference profiles and cap)
com_nb = load_community("base")                    # a member-mix variation, built in code (Section 4.3): no batteries
for m in com_nb.members:
    for k in [k for k in vars(m) if k.startswith("battery_")]:
        setattr(m, k, None)

# Question 3: five joint scenarios with the members' reference profiles uncertain (the daily minimum follows); cL at its base value
app, par = com.raw["appliance_params"], com.raw["params"]
rules3 = sg.load_json("preference_uncertainty_params.json")
scenarios = sg.draw_preference_scenarios(app, par, rules3, n=5, seed=1)                  # quantities=("profile",) by default
# ... or tune the hyper-parameters per quantity and per member, draw, and merge yourself:
spec = sg.rules_to_spec(rules3, [c["consumer_id"] for c in par["consumers"]])
sg.set_rule(spec, "profile", None, n_scenarios=10, lo=0.9, hi=1.1, mean=1.0, kappa=4.0)  # all members
sg.set_rule(spec, "cL", "C4", n_scenarios=10, lo=0.6, hi=1.4, mean=1.0, kappa=2.0)       # one member; cL uncertain as a variation
scaling = sg.draw_scaling_scenarios(spec, seed=1)                 # {quantity: {consumer_id: array of theta}}
joint = sg.merge_scenarios(app, par, scaling, mode="aligned")     # joint scenarios with .index, .theta, .members

```

**Conventions.** The hourly resolution makes kW and kWh/h numerically interchangeable, as in Assignment 1.
`InputData` carries the *full* tariffs of `bus_params.json`; the discount of the contract is in
`ContractData.tariff_discount_ratio` and applies to the metered flows of the community (model C) or of the
member (model I) as the assignment defines them. The notation of the assignment follows Assignment 1
(price p_t, tariffs tau_imp / tau_exp, member variables q_imp, q_exp, l, g, battery p_ch, p_dis, s) with a
member index; the community's import and export at the connection point are P_imp_t and P_exp_t, the cap
is P_DSO_t (`capacity_limit_kW`).

## 4. Input data

All input data lives in `data/`, in the conventions of Assignment 1: every field name carries its unit
(`_kW`, `_kWh`, `_KWh_per_hour`, `_DKK_per_kWh`, `_DKK_per_kWh2`), ratios and capacity factors are
dimensionless fractions of the corresponding maximum, every time series is a list of 24 hourly values (hour 0
to 23), `null` means "not applicable", and a community file composes its members from the shared catalogue by
appliance ID. Section 4.1 describes how the members are built; Section 4.2 gives the structure of every file; Section 4.3 says how to build variations of the base case.

### 4.1 The members and the base case

**The members.** Every member is one Assignment 1 consumer with **one load model and an asset mix**:

* *Load* - the structure of the Assignment 1 load (Question 2), with the same parameters for every member except the slope of
  the disutility: a preferred hourly profile `reference_load[t]` = `max_load_KWh_per_hour` x `reference_load_capacity_factor[t]`
  (`max_load_KWh_per_hour` = 3 kWh/h is the connection capacity; 16.2 kWh/day on `d1`); a **maximum hourly deviation**
  `max_hourly_deviation_kWh` = 1 kWh from the reference, so that the hourly bounds are `load_lower_kWh[t]` = max(0, reference - 1)
  and `load_upper_kWh[t]` = min(3, reference + 1) (the loader computes them); a **linear** disutility c_L (DKK/kWh) on the
  absolute hourly deviation from the reference; and a minimum daily
  energy **equal to the daily reference energy**, E_min = sum_t reference_load[t] (16.2 kWh on `d1`; `min_total_energy_rule`
  in the catalogue, and the loader sets the value from the profile of the day loaded). Four slopes exist in the catalogue
  and every member has one of them:

  | Load | c_L (DKK/kWh) | Behaviour against the day's price spread (0.85-2.50 DKK/kWh) |
  |---|---|---|
  | `FL_L1` | 0.25 | shifts on the day-ahead prices alone |
  | `FL_L2` | 0.5 | shifts on the larger price differences of the day |
  | `FL_L3` | 1.0 | shifts only when the cap rent adds to the price difference |
  | `FL_L4` | 2.0 | keeps its profile and pays |

  The daily minimum fixes how much a member uses over the day; the slope fixes when; the deviation limit fixes how much can
  move in one hour, so that the stiffer classes are called only in the most congested hours. With a linear disutility a member keeps
  its reference in an hour as long as the price differs from its internal value of a kWh of daily need (the multiplier of
  the daily minimum) by less than c_L, and otherwise moves the hour's load to a bound; at a price exactly c_L away it is
  indifferent (its problem is a linear program). `InputData.slope_class` returns 1-4 (1 = most flexible).
* *Assets* - with or without a PV system (`PV_01_S`: 3 kW, the day's availability profile, marginal cost 1.41 DKK/kWh;
  `has_pv`) and with or without a battery (`BAT_01_S`: 2 kWh, 0.75 kW charge/discharge, efficiencies 0.95, initial state of
  charge 50 %, end-of-day treatment your choice; `has_battery`). `InputData.archetype` combines slope class and assets:
  `"L1+PV+BAT"`, `"L3+PV"`, `"L4"` (no assets), ...

**Base case** (`params_community_base.json`, day `d1`): 16 small members, one per combination of a slope and an asset class;
the ID carries the slope index and the assets (`C3_PV` = `FL_L3` with a PV system and no battery):

| Load (slope) | no asset | PV | battery | PV and battery |
|---|---|---|---|---|
| `FL_L1` (0.25) | `C1` | `C1_PV` | `C1_B` | `C1_PVB` |
| `FL_L2` (0.5) | `C2` | `C2_PV` | `C2_B` | `C2_PVB` |
| `FL_L3` (1.0) | `C3` | `C3_PV` | `C3_B` | `C3_PVB` |
| `FL_L4` (2.0) | `C4` | `C4_PV` | `C4_B` | `C4_PVB` |

48 kW of connected load, 24 kW of PV (8 systems), 8 batteries. The eight members without PV import their whole daily energy
(16.2 kWh on `d1`) and are what makes the exchange inside the community non-trivial; the low-slope members and the batteries
are where the community's flexibility comes from. `load_community("base")` returns the members in the order of the table, row by row.

Only this community is shipped. Variations of the member mix are built in code, or as your own community file (Section 4.3).

### 4.2 File structures

#### `appliance_params.json` - the shared catalogue (Assignment 1 format)

| Key | Type | Content |
|---|---|---|
| `description` | str | what the catalogue holds |
| `DERs[]` | list | `DER_id` (`PV_01_S`), `DER_type` (`"PV"`), `max_power_kW` (3.0), `marginal_cost_DKK_per_kWh` (1.41), `hourly_profile_ratio[24]` (day `d1`), `hourly_profile_ratio_next_day[24]` (day `d2`) |
| `loads[]` | list | `load_id` (`FL_L1`-`FL_L4`), `load_type`, `consumption_utility_DKK_per_kWh` (`null`), `linear_disutility_DKK_per_kWh` (0.25 / 0.5 / 1.0 / 2.0), `quadratic_disutility_DKK_per_kWh2` (`null`), `min_total_energy_per_day_kWh` (16.17 = the daily reference energy of `d1`), `min_total_energy_rule` (the daily minimum equals the daily reference energy; the loader sets it per day), `max_load_KWh_per_hour` (3.0), `min_load_KWh_per_hour` (0.0), `max_hourly_deviation_kWh` (1.0), `hourly_bounds_rule` (text), `reference_load_capacity_factor[24]` (day `d1`), `reference_hourly_profile_ratio_next_day[24]` (day `d2`) |
| `storages[]` | list | `storage_id` (`BAT_01_S`), `storage_type`, `storage_capacity_kWh` (2.0), `max_charging_power_kW` / `max_discharging_power_kW` (0.75), `charging_efficiency` / `discharging_efficiency` (0.95), `initial_state_of_charge_ratio` (0.5) |

The profiles in the catalogue are those of day `d1`; the loader replaces them by the profiles of the day you ask for.

#### `bus_params.json` - the connection point (unchanged from Assignment 1)

A list with one bus: `bus_id` (`"Bus1"`), `import_tariff_DKK_per_kWh` (0.5), `export_tariff_DKK_per_kWh` (0.4),
`max_import_kW` / `max_export_kW` (`null`: the cap of the contract, not a physical limit, bounds the import),
`energy_price_DKK_per_kWh[24]` (day `d1`), `energy_price_next_day_DKK_per_kWh[24]` (day `d2`).

#### `weekly_profiles.json` - one week of daily profiles

| Key | Type | Content |
|---|---|---|
| `description`, `base_day` | str | `base_day` is `"d1"` |
| `days.<d>` for `d1` ... `d7` | dict | `name` (the day's story), `energy_price_DKK_per_kWh[24]`, `pv_hourly_profile_ratio[24]` (common to every PV system), `reference_load_capacity_factor[24]` (common to the four loads; the reference load is this factor x `max_load_KWh_per_hour`) |

`d1` and `d2` are the two days of Assignment 1; `d3`-`d7` are constructed from them. `load_community(..., day="d3")`
writes a day's profiles into the Assignment 1 fields; `scenario_generation.day_data` does the same on the raw
dictionaries, for your own loader. The "next day" of `d7` is `d1`.

#### `params_community_<name>.json` - a community (Assignment 1 case-file format)

| Key | Type | Content |
|---|---|---|
| `community_id`, `description` | str | |
| `consumers[]` | list of 16 | `consumer_id` (`C<k>`, `C<k>_PV`, `C<k>_B`, `C<k>_PVB`, k = 1..4), `connection_bus` (`"Bus1"` for every member: they share the meter), `list_appliances` (IDs from the catalogue: optionally `PV_01_S`, exactly one load `FL_L<k>`, optionally `BAT_01_S`) |
| `hyperparameters` | `null` | |

One file is shipped (`base`, Section 4.1). You may add your own `params_community_<name>.json` in the same format and load it with `load_community("<name>")`.

#### `dso_contract_params.json` - the capacity limitation contract

| Key | Type | Content |
|---|---|---|
| `contract_id`, `description`, `connection_point_bus` | str | |
| `tariff_discount_ratio` | float | beta = 0.75: the contract holder pays (1 - beta) x the tariffs of `bus_params.json` on its metered import and export |
| `penalty_DKK_per_kWh` | float | alpha = 10: per kWh imported above the cap in an hour |
| `capacity_limit_kW[24]` | list | the base cap: day `d1`, constant over the hours (9.4 kW) |
| `capacity_limit_rule` | str | cap = f x E / 24 in every hour (a feeder capacity), E = daily import of the reference model R on the day, f the energy factor; scale the constant to vary the tightness |
| `capacity_limit_daily_energy_factor` | float | f = 1.3 |
| `capacity_limit_daily_energy_kWh` | float | f x E on `d1` |
| `export_capacity_limit_kW` | `null` | no export cap |
| `capacity_limits_by_day_kW.<d>` | list | the cap of every day `d1`-`d7` (24 equal values) |
| `reference_model_R_by_day.<d>` | dict | `daily_import_kWh`, `peak_import_kW`, `cost_DKK`, `exceedance_kWh` of the reference model R (members alone, no contract) - to check your model R |
| `individual_contracts.rule` | str | model I: cap_{i,t} = share_i x cap_t, share_i proportional to the member's net import in R on the day |
| `individual_contracts.net_import_in_R_kWh_d1` | dict | consumer_id -> kWh |
| `individual_contracts.share`, `.share_by_day.<d>` | dict | consumer_id -> share (day `d1`; every day) |
| `individual_contracts.individual_capacity_limit_kW_d1` | dict | consumer_id -> 24 values |
| `individual_contracts.alternative_split_rules` | dict | `capacity` (rule text, `max_import_capacity_kW` and `share` per member), `peak`, `energy_gross`, `hourly` (rule texts) |

The shares of the file are those of the base community; for a member-mix variant R changes, so the loader falls back on
the capacity rule (`ContractData.share_rule == "capacity"`) - recompute the net-import shares with your own model R if
you keep the base rule.

#### `preference_uncertainty_params.json` - Question 3

The reference profile of every load is uncertain (the base study), and its disutility slope can be made uncertain as a variation. Each
uncertain quantity is driven by **one scaling parameter theta** per member, drawn from a scaled Beta distribution:

    theta ~ lo + (hi - lo) x Beta(a, b),    a = kappa (mean - lo) / (hi - lo),    b = kappa - a

- `lo`, `hi`: hard bounds, never violated (no clipping);
- `mean`: where the scenarios are centred (1 = the base case; another value shifts them);
- `kappa`: concentration, i.e. how uncertain: Var theta = (hi - lo)^2 mu (1 - mu) / (kappa + 1) with mu = (mean - lo)/(hi - lo);
  kappa -> infinity is deterministic, kappa = 2 with the mean at the midpoint is Uniform(lo, hi), kappa < 2 is U-shaped.
  The code computes a and b (`BetaRule.ab`); check them when you tune - a or b below 1 piles density at a bound;
- `n_scenarios`: 1 means the quantity is not uncertain in the experiment (its single scenario is the base value, theta = 1).

| Key | Quantity | Rule | Defaults (base = Uniform) |
|---|---|---|---|
| `quantities.profile` | `reference_load_capacity_factor` rho_t | rho_t ^ theta, one theta per member for all hours (keeps 0 and 1 fixed, stays in [0, 1]); the daily minimum follows: `min_total_energy_per_day_kWh` = max_load x sum_t rho_t^theta in every scenario | 5 scenarios, [0.8, 1.2], mean 1, kappa 2 |
| `quantities.cL` | `linear_disutility_DKK_per_kWh` | value x theta (a variation: not uncertain in the base study) | 1 scenario; when uncertain [0.6, 1.4], mean 1, kappa 2 |
| `daily_minimum_rule` | the rule above, in words | | |
| `per_consumer` | overrides `{quantity: {consumer_id: {...}}}` of any hyper-parameter | | empty |
| `merging` | how independent draws become joint scenarios: joint scenario s takes scenario s of every uncertain (quantity, member) pair (`aligned`; the code also offers `random` and `product`) | | `aligned` |
| `scenarios` | in-sample sizes to study, out-of-sample size, seed policy | | 5 and 10, 100 |

Code (`src/scenario_generation.py`): `rules_to_spec(rules, consumer_ids)` -> `{quantity: {consumer_id: BetaRule}}`;
`set_rule(spec, quantity, consumer_id or None, **hyper-parameters)`; `draw_scaling_scenarios(spec, seed)` -> independent
scenarios of theta per quantity and per member (each pair has its own random stream, so changing one pair's `n_scenarios`
does not change the others' draws); `merge_scenarios(app, community, scaling, mode="aligned")` -> joint scenarios,
each with `index[(quantity, consumer_id)]` (which independent scenario it took), `theta[quantity][consumer_id]` and
`members[consumer_id] -> {appliance_id: scaled appliance dict}`; `draw_preference_scenarios(app, community, rules, n, seed,
quantities)` does the three steps in one call with the file's hyper-parameters; `expected_preference_scenario` is the
baseline (theta = 1 everywhere).

#### Figures

`community_base_case.png/.svg`, `setup_R.png/.svg`, `setup_I.png/.svg`: Figures 1-3 of the assignment (schematic); `fig_q3_scenarios` (.png/.svg): Figure 4, the scenario generation of Question 3.

### 4.3 Variations of the base case

Every sensitivity analysis is a variation of the base case, built *in code* from the loaded objects. The data files are
never edited, so that base-case results are on the data as provided and comparable across groups. Table 11 of the
assignment gives the same list with the rules that keep a variation consistent.

| Dimension | What to change | Rules |
|---|---|---|
| a) Member mix | After `load_community("base")`, edit `com.members`: drop or duplicate `InputData` objects (new `consumer_id` for a copy); change the load slope `linear_disutility` (0.25 to 2.0); remove a PV system (`pv_max_kW = 0`, `pv_available` zero) or a battery (`battery_*` = `None`), or add them with the catalogue values. Or write your own `params_community_<name>.json` (Section 4.2) and load it by name. | One connection point. Recompute the reference import `E^R`, the cap (`f x E^R`, same profile shape) and the shares `kappa_i` from your own model R for the new mix: the loader's shares for a non-base community (`share_rule = "capacity"`) are a placeholder. Name the mix and state what changed. Not required by any question. |
| b) Daily conditions | `load_community("base", day="d3")`, `d1`-`d7`: prices, PV availability, reference profiles and daily minimum, and the cap of that day, all together. | Never combine the prices of one day with the PV or the cap of another. `d1` and `d2` are the Assignment 1 days. |
| c) Contract | `com.contract.tariff_discount_ratio` (beta), `.penalty_DKK_per_kWh` (alpha), `.export_capacity_limit_kW`. Cap tightness: multiply `.capacity_limit_kW` by `f'/1.3`; a time-varying cap profile is a variation of your own. Split rule of the individual contracts: recompute `.share` from your model R with another rule of `dso_contract_params.json / individual_contracts / alternative_split_rules` (peak, gross energy, hourly, capacity) and `.individual_capacity_limit_kW = kappa_i x cap`. | Shares sum to one. Compare I and C under the same contract. Name the benchmark by its split rule. |
| d) Members' parameters | On each `InputData`: `linear_disutility` (one factor for all members keeps the ordering of the slopes); `max_hourly_deviation_kWh` (recompute `load_lower_kWh` / `load_upper_kWh`); `min_daily_energy_kWh` as a ratio of `reference_load.sum()`; `pv_max_kW` (rescale `pv_available`); `battery_capacity_kWh` (scale `battery_max_charge_kW` / `battery_max_discharge_kW` with it). | Keep `reference_load <= load_max_kWh`. A daily minimum above the reference energy forces consumption beyond the preferred profile: say so if intended. Change the same parameter for all members unless heterogeneity is the point. |
| e) Model variations | Switches of your model class (battery end-of-day; cap as hard constraint or penalty; Question 2 objective, requirements, hard or soft, price bound and regularization; Question 3 cap treatment, requirements per scenario or in expectation, risk measure). | One switch at a time, on the same data. Name the variant with a superscript. |
| f) Uncertainty, Question 3 | `sg.set_rule(spec, quantity, consumer_id, n_scenarios=..., lo=..., hi=..., mean=..., kappa=...)` for `"profile"` (the base study) or `"cL"` (the disutility slope uncertain instead, or as well); in-sample `N`, out-of-sample size, seeds. | Same out-of-sample set and common random numbers across the models compared. Seeds reported. |

Write each variation as a small function that returns a modified copy (`with_discount(com, 0.25)`, `without_batteries(com)`),
so that every variant is reproducible from a name and a value, and run your model class over the variants as named
configurations with outputs saved under `results/`. If you add appliances, communities or scenarios, follow the same
file structures and document them here (file, fields, units, seeds).

## 5. What is expected of your code

As in Assignment 1: keep data loading, model building, solving and plotting separate; document every function you add
(a short docstring with inputs, outputs and units); make each experiment of the report runnable with a single command
and save its outputs under `results/`; fix and report the seeds of every scenario set; and commit regularly with
meaningful messages - the git history is also a record of everyone's contribution. The graders should be able to open
the code, follow this README, and reproduce every number and figure of the report.

## 6. Models (`src/model.py`)

`CommunityModel` holds every member of the community in one gurobipy model. The three setups of Question 1 differ only in two switches (`SETUPS`):

| setup | `level`: where the DSO meters | `contract` | meaning |
|---|---|---|---|
| `"R"` | `"member"` | no | every member alone behind its own meter, full tariffs, no cap |
| `"I"` | `"member"` | yes | own meter and own contract: cap kappa_i x cap, discounted tariffs, penalty on its own exceedance |
| `"C"` | `"community"` | yes | one meter and one contract for the community; the members exchange freely behind the meter |

```python
from src.data_loader import load_community
from src.model import CommunityModel

res = CommunityModel(load_community("base"), setup="C").build().solve()     # -> Results
```

The three are linear programs. The objective MINIMIZES the total cost of the community (= minus its welfare, DKK/day): the members' disutility and PV cost, plus the energy at p_t, the tariffs and the penalty at the meter(s). At level `"member"` no constraint couples the members, so R and I decompose into one problem per member.

### 6.1 The member (identical in R, I and C: `CommunityModel._add_member`)

Per member i and hour t, in kWh/h (= kW at the hourly resolution):

| name in the code | variable / constraint | written as | members |
|---|---|---|---|
| `load` | l_t free | variable | all |
| `load_lower`, `load_upper` | l_t >= `load_lower_kWh`_t, l_t <= `load_upper_kWh`_t | constraints | all |
| `deviation` | dev_t >= 0 | variable | all |
| `dev_up`, `dev_dn` | dev_t >= l_t - ref_t, dev_t >= ref_t - l_t | constraints | all |
| `min_daily_energy` | sum_t l_t >= E_min (one per member) | constraint | all |
| `pv` | pv_t >= 0 | variable | PV owners |
| `pv_max` | pv_t <= `pv_available`_t | constraint | PV owners |
| `p_ch`, `p_dis` | 0 <= p_ch_t <= max charge, 0 <= p_dis_t <= max discharge | variables, **bounds (`ub=`)** | battery owners |
| `soc` | 0 <= s_t <= capacity, t = 0..24 (s_t at the start of hour t) | variable, **bounds (`ub=`)** | battery owners |
| `soc_init`, `soc_dynamic` | s_0 = initial SoC; s_t+1 = s_t + eta_ch p_ch_t - p_dis_t / eta_dis | constraints | battery owners |
| `soc_end` | s_24 >= s_0 (`end_soc_mode="constraint"`, the default) | constraint | battery owners |

Cost of the member: c_L sum_t dev_t + c_PV sum_t pv_t. Net consumption q_t = l_t + p_ch_t - pv_t - p_dis_t.

### 6.2 The meters

| setup | variables | constraints | cost |
|---|---|---|---|
| R | `import`, `export` >= 0 per member | `balance`: import - export - q_i,t = 0 | p_t (import - export) + tau_imp import + tau_exp export |
| I | as R, plus `exceedance` e_i,t >= 0 | as R, plus `cap`: e_i,t >= import_i,t - kappa_i cap_t | p_t (import - export) + (1 - beta) tau_imp (import - e) + tau_imp e + (1 - beta) tau_exp export + alpha e |
| C | `import`, `export`, `exceedance` E_t >= 0 for the community | `balance`: import - export - sum_i q_i,t = 0; `cap`: E_t >= import_t - cap_t | as I, on the community's flows |

**Contract rule** (our reading of the assignment): a kWh imported below the cap pays (1 - beta) tau_imp; a kWh above the cap loses the discount and pays the full tau_imp plus the penalty alpha; a kWh exported pays (1 - beta) tau_exp.
At the optimum e and E equal the exceedance max(0, import - cap), for any alpha >= 0.

**Duals** (Gurobi `Pi`, MINIMIZE): >= 0 for a `>=` constraint, <= 0 for a `<=` constraint. `balance` is arranged so that its dual is the marginal price of energy: the member's own in R and I; in C it is lambda_t, the internal price of the dual settlement. The dual of `cap` is the value of one more kW of cap in that hour.

**Not enforced**, because with positive prices it only loses money: simultaneous import and export, simultaneous charge and discharge (both zero in R, I and C on d1-d7). A model in which prices can be negative must exclude them.

### 6.3 Results

`solve()` returns a `Results`:

* `objective` (DKK) and `meta["objective_terms"]`: `disutility`, `pv_cost`, `energy`, `tariff`, `penalty` (DKK);
* `hourly`, per hour at the community level: `price`, `cap`, sums of `reference_load` / `load` / `pv` / `p_ch` /
  `p_dis`, `import` / `export` at the meter(s) (R, I: the sum of the members' own), `exceedance` = max(0, import -   cap) (the congestion of the feeder), `exceedance_billed` (I: sum of the members' exceedance of their own caps;
  C: the community's), `dual_balance` (lambda_t in C) and `dual_cap`;
* `members`, per (member, hour): every variable of the member (`soc` at the end of the hour), `net` (q) and the   hourly duals `dual_<constraint>`;
* `member_costs`, per member: `archetype`, the member's cost terms and `total`, the duals of its daily constraints (`dual_min_daily_energy` = mu_i). In C the grid costs are the community's (`objective_terms`): `total` covers only the member's disutility and PV cost, before the settlement;
* `save(folder)` writes the tables as CSV.

A model built on the class that has integer variables gets the same tables, without the duals. Not implemented: an export cap, several scenarios of one member.

### 6.4 Validation

`python main.py --verify` (`src/validation.py`). Run it after any change to `src/model.py`; every check must PASS:

* R against the reference values of `dso_contract_params.json`: the cost on d1-d7 (the daily import may differ slightly from the file: R has alternative optima, with the same cost);
* R and I: the joint model equals the members solved one by one;
* I with beta = 0 and no cap equals R (limit case);
* the exceedance variables of I and C equal max(0, import - cap); the balance at the community meter of C holds;
* C <= I under the same contract; C at the cap in 18 hours of d1 (assignment text);
* the manager's surplus in the settlement of C equals the cap rent sum_t dual_cap_t x cap_t (complementary slackness).

## 7. Experiments

| command | what | outputs |
|---|---|---|
| `python main.py --verify` | validation checks (Section 6.4) | PASS / FAIL per check; exit code 1 on a failure |
| `python main.py --q1-base` (`--day d1`) | Question 1: R, I and C on the base community | `results/q1_base/`: `metrics_<day>.csv`, `cost_terms_<day>.csv`, `import_change_<I, C>_vs_R_<day>.csv`, `fig_import_vs_cap_<day>.png`, `fig_shift_<day>.png`, `shift_by_class_<day>.csv`, `settlement_<day>.csv`, `budget_<day>.csv`, `fig_internal_price_<day>.png`, `fig_member_gains_<day>.png`, and per setup `base_<day>_<setup>_hourly.csv`, `_members.csv`, `_member_costs.csv`, `_summary.txt` |
| `python main.py --plot-data` | input data of d1-d7: prices, reference load, PV available, cap (for inspection) | `results/data/data_<day>.png` |
| `python main.py --q1-sensitivity` | Question 1: sweeps of the discount beta and of the cap factor f (cap = f E_R / 24), one at a time, the other at its base value (d1; values in `q1.SWEEPS`) | `results/q1_sensitivity/`: `sweep_beta.csv`, `sweep_cap_factor.csv` (the metrics of R, I and C for every value), `fig_sensitivity.png` (the cap: value and max exceedance), `fig_sensitivity_full.png` (both terms, all three metrics) |

**Question 1 metrics** (`q1.metrics`), all against R (the members alone, no contract):

| metric | unit | point of view | definition |
|---|---|---|---|
| `value_DKK` | DKK/day | community | Cost_R - Cost_X; > 0: the contract is worth signing. value_C - value_I = Cost_I - Cost_C is the value of pooling |
| `max_exceedance_kW` | kW | DSO | max_t (import_t - cap_t)+ at the connection point |
| `hours_above_cap` | h | DSO | number of hours with import_t > cap_t |

The metrics table also gives `cost_DKK`, `exceedance_kWh`, `peak_import_kW` and `energy_shifted_kWh` (= sum |l - ref| / 2).

**Settlement of C** (`q1.settlement`, DKK/day): member i pays sum_t lambda_t q_i,t, with lambda_t the dual of the
community balance (the one Gurobi returns; it is not unique in the hours at the cap, the total surplus is). Its cost
in C is its disutility + PV cost + payment; `gain_vs_R` = cost_R - cost_C and `gain_vs_I` = cost_I - cost_C (> 0:
better off in C; `IR_vs_R`, `IR_vs_I`: gain >= 0). The budget compares what the manager collects with what it pays
the DSO (energy, tariff, penalty); the surplus equals the cap rent.
