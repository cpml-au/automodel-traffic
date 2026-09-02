# SR-Traffic

<p align="center">
<img src="readme_figure.png" alt="My figure" width="100%">
</p>

## Installation

The dependencies are collected in `environment.yaml` and can be installed, after cloning the repository, using [`mamba`]("https://github.com/mamba-org/mamba"):

```bash
mamba env create -f environment.yaml
```

Once the environment is installed and activated, install the library using

```bash
pip install -e .
```

## Usage

To reproduce the paper's figures and error tables using the precomputed model
parameters, run

```bash
python src/sr_traffic/fd/results.py --road_name {road_name} --task {task_name}
```

where `{road_name}` is either `US101` or `I80`, and `{task_name}` is either
`prediction` or `reconstruction`. This script does not perform parameter
calibration or symbolic-regression search: it reruns the traffic simulations
with fixed parameters, evaluates the models, and writes the figures under
`results/<road_name>/<task_name>/`. Each run also writes its error table in
LaTeX and Markdown formats as `error_table.tex` and `error_table.md` in the same
directory.

For `I80 prediction`, the report additionally simulates the five Phase-4
`automodel-*` corrections. It writes Automodel-only figures and error/score
tables alongside the combined baseline-versus-Automodel tables under
`results/I80/prediction/`.

To re-calibrate a given fundamental diagram, run

```bash
python src/sr_traffic/fd/calibration.py --config src/sr_traffic/fd/configs/{fnd_name}.yaml
```

where `{fnd_name}` is either `greenshields`, `triangular`, `weidmann`,
`del_castillo`, or `idm`.

This command calibrates only the predefined baseline diagrams; SR-discovered
models are searched for and fitted separately by the SR-Traffic pipeline below.

Each file in `src/sr_traffic/fd/configs/` defines one calibration problem:

- `road_name` selects the dataset (`I80` or `US101`).
- `task` selects the data split. `prediction` calibrates on earlier time points
  and tests on later ones, while `reconstruction` calibrates on a subset of
  spatial locations and tests on the held-out locations.
- `flux` is the case-sensitive name of the fundamental-diagram function in
  `src/sr_traffic/fd/diagrams.py`.
- `bounds` contains two lists: the lower bounds followed by the upper bounds.
  Their entries follow the order of the selected flux function's parameters
  after density, as listed below.
- `opt.num_ind` is the population size used by the PyGMO simple evolutionary
  algorithm, and `opt.num_gen` is the number of generations. The supplied
  values (`1000` and `100`) are full calibration settings and can be reduced
  for a quick trial.

| Config | Parameter order in `bounds` | Meaning |
| --- | --- | --- |
| `greenshields` | `v_max`, `rho_max` | Free-flow speed and jam density |
| `triangular` | `V_0`, `l_eff`, `T` | Free-flow speed, effective vehicle length, and time headway |
| `weidmann` | `v_max`, `rho_max`, `lambda_w` | Free-flow speed, jam density, and shape parameter |
| `del_castillo` | `C_jam`, `V_max`, `rho_max`, `theta` | Congested-wave speed scale, free-flow speed, jam density, and shape parameter |
| `idm` | `s0`, `T`, `delta`, `v0` | Minimum gap, time headway, acceleration exponent, and desired speed |

Calibration minimizes the average of the normalized squared density and
velocity errors on the training split. The data and parameters are
nondimensionalized by the preprocessing code, so the bounds are expressed in
the model's normalized units.

### Selecting the baseline fundamental diagram for SR

SR-Traffic searches for a multiplicative correction to a predefined baseline
fundamental diagram. If the symbolic-regression individual is denoted by
`g_SR`, the flux used in the traffic simulation is

```text
q_SR(rho) = q_baseline(rho; opt_coeffs) * g_SR(rho).
```

The baseline is selected by the `gp.ansatz` entry in
`src/sr_traffic/sr/sr_traffic.yaml`. The default is

```yaml
ansatz:
  flux: triangular_flux
  v: triangular_v
  opt_coeffs: [0.37013956, 1.48964708, 6.59672108]
```

Thus, the default search modifies the triangular fundamental diagram. The
three coefficients are, in order, `V_0`, `l_eff`, and `T`. The baseline
coefficients are kept fixed during the SR run; the search changes `g_SR` and
fits the constants occurring in that expression.

To select another baseline, change all three fields in `ansatz`: `flux` is the
case-sensitive flux-function name, `v` is its matching velocity function, and
`opt_coeffs` follows the parameter order below.

| Baseline | `flux` | `v` | Order of `opt_coeffs` |
| --- | --- | --- | --- |
| Greenshields | `Greenshields_flux` | `Greenshields_v` | `v_max`, `rho_max` |
| Triangular | `triangular_flux` | `triangular_v` | `V_0`, `l_eff`, `T` |
| Weidmann | `Weidmann_flux` | `Weidmann_v` | `v_max`, `rho_max`, `lambda_w` |
| Del Castillo | `del_castillo_flux` | `del_castillo_v` | `C_jam`, `V_max`, `rho_max`, `theta` |
| IDM | `IDM_flux` | `IDM_v` | `s0`, `T`, `delta`, `v0` |

For example, an IDM baseline can be selected with

```yaml
ansatz:
  flux: IDM_flux
  v: IDM_v
  opt_coeffs: [s0, T, delta, v0]
```

where the placeholders must be replaced by numeric coefficients calibrated for
the selected `road_name` and `task`.

Notice that the `opt_coeffs` are obtained by calibrating the chosen baseline before running
symbolic regression. Run `src/sr_traffic/fd/calibration.py` with the appropriate
fundamental-diagram config, as described above. The calibration minimizes the
training density and velocity error and prints `pop.champion_x`; copy that
printed vector into `gp.ansatz.opt_coeffs`. This copy is currently manual: the
calibration script does not update `sr_traffic.yaml` automatically. Make sure
the calibration config's `road_name` and `task` match the corresponding SR
settings.

### SR model score

The data error used to evaluate an SR model is the average of the relative
squared density and velocity errors:

```text
E_rho  = 100 * sum((rho_computed - rho_data)^2) / sum(rho_data^2)
E_v    = 100 * sum((v_computed - v_data)^2) / sum(v_data^2)
E_data = 0.5 * (E_rho + E_v).
```

Lower values are better. The factors of `100` express the two relative squared
errors as percentages. This quantity is a relative squared error, not a mean
squared error: the squared residuals are normalized by the squared norm of the
observations rather than by the number of observations. Although the simulated
flow is also computed, it is not included in `E_data`.

During the symbolic-regression search, the expression-tree length penalty from
`gp.penalty.reg_param` is added to the data error:

```text
E_fitness = E_data + reg_param * number_of_tree_nodes.
```

The default `reg_param` is `0.01`. The search minimizes `E_fitness`, thereby
trading off agreement with the observed density and velocity against symbolic
expression complexity. A candidate receives a data error of `100` if it fails
the velocity feasibility check, which requires velocity to be non-increasing
with density. Invalid expressions and expressions rejected by the tree checks
are penalized similarly.

The final test error printed after the search is `E_data` alone; it does not
include the expression-length penalty.

### Automodel I80 prediction corrections

The validation-selected multiplicative corrections and Phase-4 refit results
are recorded in `automodel/final_candidates.json` and `automodel/phase4/`.
Production-ready parameterized flux entry points are available in
`sr_traffic.fd.diagrams`:

```python
from sr_traffic.fd.diagrams import triangular_corrected_flux

# The final I80 correction coefficient is the default fourth argument.
flux = triangular_corrected_flux(rho, V_0, l_eff, T)
```

Equivalent corrected entry points are provided for Greenshields, IDM,
Weidmann, and Del Castillo. Their correction parameters remain optional so a
different calibration can override the I80 prediction defaults.

### Complete I80 prediction results

`E_data` below is the unpenalized score on the interior rows `1:-3`. Baseline
and Automodel results come from this checkout. The `SR-*` results were
reproduced by running the official
[`cpml-au/SR-Traffic`](https://github.com/cpml-au/SR-Traffic) results script at
commit `3bf285ab1d6b2b61f0c0f27c0d80e42633b84ac4` in an isolated clone. Bold test
scores mark the best model within each baseline family.

| Baseline family | Model | Source | Training $E_{data}$ | Test $E_{data}$ |
|---|---|---|---:|---:|
| Greenshields | Greenshields | Baseline | 8.166908 | 9.230426 |
| Greenshields | SR-Greenshields | Upstream SR | 7.464482 | 8.602553 |
| Greenshields | automodel-Greenshields | Automodel | 5.821048 | **7.091936** |
| IDM | IDM | Baseline | 7.319251 | 6.957441 |
| IDM | SR-IDM | Upstream SR | 6.597450 | **6.225629** |
| IDM | automodel-IDM | Automodel | 6.368446 | 7.117303 |
| Weidmann | Weidmann | Baseline | 6.858360 | 7.651073 |
| Weidmann | SR-Weidmann | Upstream SR | 5.957339 | **6.860641** |
| Weidmann | automodel-Weidmann | Automodel | 6.713568 | 7.458993 |
| Triangular | Triangular | Baseline | 7.988255 | 7.918859 |
| Triangular | SR-Triangular | Upstream SR | 7.229762 | **6.727777** |
| Triangular | automodel-Triangular | Automodel | 6.673826 | 7.118426 |
| Del Castillo | Del Castillo | Baseline | 6.824368 | 7.196903 |
| Del Castillo | SR-Del Castillo | Upstream SR | 6.098916 | 6.800595 |
| Del Castillo | automodel-Del Castillo | Automodel | 5.811699 | **6.779614** |

The upstream and current reports use the same chronological I80 prediction
split and score definition, but were executed from different code commits.
The reproduced SR-only relative-error table and additional generated figures
are available under `results/I80/prediction/`.

## Citing

## Acknowledgements

This work is supported by the European Union (European Research Council (ERC), ALPS, 101039481). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the ERC Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
