# Automodel Traffic

<p align="center">
  <img src="readme_figure.png" alt="Automodel Traffic logo" width="100%">
</p>

Automodel Traffic provides fitted, interpretable corrections to five classical
traffic fundamental diagrams for the NGSIM I80 prediction benchmark. It starts
from the hypothesis that a classical diagram captures the main traffic physics
but may retain a systematic, data-dependent bias. Instead of replacing that
diagram, Automodel multiplies its flux by a learned correction:

```text
q_corrected(rho) = q_baseline(rho; theta) * g(rho; c)
```

Here `q_baseline` is Greenshields, IDM, Weidmann, Triangular, or Del Castillo;
`theta` contains its calibrated coefficients; and `g` is a compact correction
with fitted coefficients `c`. Setting `g = 1` recovers the original fundamental
diagram. The selected corrections use positive exponential multipliers, which
preserve the baseline's units and zeros while allowing its magnitude and shape
to change. Some corrections also use short-range density contrasts so that the
flux can respond to the traffic state around a mesh node. Finite simulations,
non-negative velocity, and the relevant monotonicity checks are enforced
separately.

The repository includes the corrected flux functions, frozen coefficients,
scripts for reproducing the simulations and tables, and complete evaluation
records. The model structures were discovered using the
[`automodel` skill](https://github.com/Unlayer-AI/automodel).

The final correction structures and parameter bounds are defined in
[`automodel/final_models.py`](automodel/final_models.py). Their I80 refit
coefficients are the defaults in
[`src/sr_traffic/fd/diagrams.py`](src/sr_traffic/fd/diagrams.py).

## Installation

Create the supplied Python 3.12 environment with Mamba or Micromamba:

```bash
micromamba env create -f environment.yaml
micromamba activate automodel-traffic
```

The numerical package uses a `src/` layout, while the Automodel utilities live
at the repository root. Set both on `PYTHONPATH` when running commands from a
checkout:

```bash
export PYTHONPATH="$PWD/src:$PWD"
```

## Reproduce the I80 results

Run all five classical baselines and all five frozen Automodel corrections:

```bash
python src/sr_traffic/fd/results.py --road_name I80 --task prediction
```

The command writes plots and Markdown/LaTeX tables to
`results/I80/prediction/`, including:

- `score_table.md`: classical and Automodel `E_data` scores;
- `automodel_score_table.md`: Automodel-only scores;
- `error_table.md`: held-out density, velocity, `E_data`, baseline-improvement,
  and TTS errors;
- flux, velocity, predicted-versus-observed, and space-time plots.

These files are generated artifacts and are intentionally ignored by Git. The
command uses the frozen coefficients; it does not repeat model discovery or
refit against the held-out test interval.

## Use a corrected flux in Python

The corrected functions accept a
[`DCTKit`](https://github.com/cpml-au/dctkit) primal 0-cochain for density,
followed by the usual coefficients of the chosen baseline:

```python
from sr_traffic.fd.diagrams import triangular_corrected_flux

# rho is a dctkit.dec.cochain.CochainP0.
# The final I80 correction coefficient is used by default.
q = triangular_corrected_flux(rho, V_0, l_eff, T)
```

Available entry points are:

```python
Greenshields_corrected_flux(rho, v_max, rho_max)
IDM_corrected_flux(rho, s0, T, delta, v0)
Weidmann_corrected_flux(rho, v_max, rho_max, lambda_w)
triangular_corrected_flux(rho, V_0, l_eff, T)
del_castillo_corrected_flux(rho, C_jam, V_max, rho_max, theta)
```

Each function also accepts its correction coefficients as optional trailing
arguments. For example:

```python
q = triangular_corrected_flux(rho, V_0, l_eff, T, a=-500.0)
```

The bundled defaults are specific to the I80 prediction protocol. Fit and
validate new correction coefficients before applying these structures to a
different road, split, or normalization.

## Recalibrate a classical baseline

The five YAML files under `src/sr_traffic/fd/configs/` define the baseline
calibration problem. Each file has the same fields:

| YAML field | Meaning |
|---|---|
| `road_name` | Dataset passed to preprocessing. The supplied configs use `I80`. |
| `task` | Split strategy: `prediction` splits chronologically; `reconstruction` holds out spatial locations. |
| `flux` | Case-sensitive flux function from `src/sr_traffic/fd/diagrams.py`. |
| `bounds` | Two arrays, `[lower_bounds, upper_bounds]`, ordered like the selected flux function's parameters after `rho`. |
| `opt.num_ind` | PyGMO population size for the simple evolutionary algorithm. |
| `opt.num_gen` | Number of evolutionary generations. |

The parameters represented by `bounds` are:

| Config | Parameter order | Meaning |
|---|---|---|
| `greenshields.yaml` | `v_max`, `rho_max` | Free-flow speed; jam density |
| `idm.yaml` | `s0`, `T`, `delta`, `v0` | Minimum gap; time headway; acceleration exponent; desired speed |
| `weidmann.yaml` | `v_max`, `rho_max`, `lambda_w` | Free-flow speed; jam density; curve-shape parameter |
| `triangular.yaml` | `V_0`, `l_eff`, `T` | Free-flow speed; effective vehicle length; time headway |
| `del_castillo.yaml` | `C_jam`, `V_max`, `rho_max`, `theta` | Congested-wave speed scale; free-flow speed; jam density; curve-shape parameter |

The data and model variables are nondimensionalized during preprocessing, so
the numeric bounds in these files are expressed in normalized units. The
supplied `num_ind: 1000` and `num_gen: 100` values are full calibration settings
and can be reduced for a quicker trial.

To recalibrate one baseline:

```bash
python src/sr_traffic/fd/calibration.py \
  --config src/sr_traffic/fd/configs/triangular.yaml
```

Replace `triangular.yaml` with `greenshields.yaml`, `idm.yaml`,
`weidmann.yaml`, or `del_castillo.yaml` as needed. The script minimizes the
average normalized density and velocity error on the training partition and
prints the best parameter vector in the order shown above.

## Classical and Automodel results

The primary outcomes are relative L2 density and velocity errors on the held-out
test interval. Their predefined aggregate is
`E_data = 50 * (E_rho^2 + E_v^2)`; models are ranked only by this held-out score
using full-precision values. `vs. baseline` is the percentage reduction in test
`E_data` relative to the corresponding uncorrected diagram, so positive values
mean that the correction helped. These three errors use only scored interior
rows, excluding the supplied boundary and ghost values.

`E_TTS` is the relative total-travel-time error on the test interval,
`|TTS_model - TTS_data| / TTS_data`, where TTS is the space-time integral of
density over the full road. Lower is better for every error column.

All values come from this checkout. Bold entries mark the best held-out value
in each error column.

| Model | $E^{\mathrm{ts}}_\rho$ | $E^{\mathrm{ts}}_v$ | $E^{\mathrm{ts}}_{data}$ (rank) | vs. baseline | $E^{\mathrm{ts}}_{TTS}$ |
|---|---:|---:|---:|---:|---:|
| Greenshields | 0.294 | 0.313 | 9.230 (10) | — | 0.152 |
| automodel-Greenshields | 0.255 | 0.277 | 7.092 (3) | +23.17% | 0.121 |
| IDM | 0.258 | 0.270 | 6.957 (2) | — | 0.116 |
| automodel-IDM | 0.256 | 0.277 | 7.117 (4) | -2.30% | 0.125 |
| Weidmann | 0.265 | 0.288 | 7.651 (8) | — | **0.115** |
| automodel-Weidmann | 0.265 | 0.281 | 7.459 (7) | +2.51% | 0.117 |
| Triangular | 0.269 | 0.294 | 7.919 (9) | — | 0.129 |
| automodel-Triangular | 0.256 | 0.277 | 7.118 (5) | +10.11% | 0.127 |
| Del Castillo | 0.257 | 0.279 | 7.197 (6) | — | 0.119 |
| automodel-Del Castillo | **0.253** | **0.267** | **6.780 (1)** | +5.80% | 0.123 |

Automodel-Del Castillo has the lowest held-out density, velocity, and aggregate
errors in this comparison. Automodel-Greenshields provides the largest
improvement over its classical baseline. Automodel-IDM is the only correction
that increases held-out `E_data` relative to its baseline, while the uncorrected
Weidmann model has the lowest TTS error.

## Acknowledgements

This work is supported by the European Union (European Research Council (ERC),
ALPS, 101039481). Views and opinions expressed are those of the authors only and
do not necessarily reflect those of the European Union or the ERC Executive
Agency. Neither the European Union nor the granting authority can be held
responsible for them.
