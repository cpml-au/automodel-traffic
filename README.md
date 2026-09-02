# Automodel Traffic

<p align="center">
  <img src="readme_figure.png" alt="Automodel Traffic logo" width="100%">
</p>

Automodel Traffic provides fitted, interpretable corrections to five classical
traffic fundamental diagrams for the NGSIM I80 prediction benchmark. The
repository includes the corrected flux functions, the frozen coefficients,
scripts for reproducing the simulations and tables, and the complete evaluation
records. The model structures were discovered using the
[`automodel` skill](https://github.com/Unlayer-AI/automodel).

## Multiplicative-correction hypothesis

The project assumes that a classical fundamental diagram captures the main
traffic physics but may have a systematic, data-dependent bias. Instead of
replacing it, Automodel multiplies its flux by a learned correction:

```text
q_corrected(rho) = q_baseline(rho; theta) * g(rho; c)
```

- `q_baseline` is Greenshields, IDM, Weidmann, Triangular, or Del Castillo.
- `theta` contains the calibrated coefficients of that baseline.
- `g` is a compact correction with fitted coefficients `c`.
- `g = 1` recovers the original fundamental diagram.

The selected corrections use positive exponential multipliers. This preserves
the baseline's units and zeros while allowing its magnitude and shape to change.
Some corrections also use short-range density contrasts, so the corrected flux
can respond to the local traffic state around a mesh node. Finite simulations,
non-negative velocity, and the relevant monotonicity checks are enforced
separately.

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
- `error_table.md`: density and velocity component errors;
- flux, velocity, predicted-versus-observed, and space-time plots.

These files are generated artifacts and are intentionally ignored by Git. The
command uses the frozen coefficients; it does not repeat model discovery or
refit against the held-out test interval.

## Use a corrected flux in Python

The corrected functions accept a DCTKit primal 0-cochain for density, followed
by the usual coefficients of the chosen baseline:

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
calibration bounds and optimizer settings. To recalibrate one baseline:

```bash
python src/sr_traffic/fd/calibration.py \
  --config src/sr_traffic/fd/configs/triangular.yaml
```

Replace `triangular.yaml` with `greenshields.yaml`, `idm.yaml`,
`weidmann.yaml`, or `del_castillo.yaml` as needed. The supplied population and
generation counts are full calibration settings and can be reduced for a quick
trial.

## Classical, SR, and Automodel results

The score is the average of normalized density and velocity squared errors:

```text
E_rho  = 100 * sum((rho_model - rho_data)^2) / sum(rho_data^2)
E_v    = 100 * sum((v_model - v_data)^2) / sum(v_data^2)
E_data = 0.5 * (E_rho + E_v)
```

Lower is better. The table reports unpenalized `E_data` on the held-out
chronological test interval. Classical and Automodel values come from this
checkout. The SR values were reproduced with the official
[`cpml-au/SR-Traffic`](https://github.com/cpml-au/SR-Traffic) results script at
commit `3bf285ab1d6b2b61f0c0f27c0d80e42633b84ac4`; those models are benchmarks and
are not bundled here.

| Baseline family | Classical baseline | SR benchmark | Automodel | Best method |
|---|---:|---:|---:|---|
| Greenshields | 9.230426 | 8.602553 | **7.091936** | Automodel |
| IDM | 6.957441 | **6.225629** | 7.117303 | SR |
| Weidmann | 7.651073 | **6.860641** | 7.458993 | SR |
| Triangular | 7.918859 | **6.727777** | 7.118426 | SR |
| Del Castillo | 7.196903 | 6.800595 | **6.779614** | Automodel |

Automodel is best for Greenshields and Del Castillo and improves four of the
five classical baselines. The SR benchmark is best for IDM, Weidmann, and
Triangular. Del Castillo has the lowest Automodel test score, while the IDM
correction is the only Automodel result that is worse than its classical
baseline.

The SR benchmark and this checkout use the same chronological I80 task and score
definition, but they were executed from different commits. The comparison is
therefore a reproduced benchmark, not a claim that all implementations share an
identical code path.

## Evaluation scope

The 180 I80 samples are split chronologically: times 0--63 for training, 64--107
for validation, and 108--179 for the final test. The correction structures were
selected without test access, refitted on training plus validation, and then
evaluated once on the test block.

The test block is now consumed and should not be used for further model
selection. Full-series density normalization and observed future boundary
density are retained for benchmark compatibility, so this is a conditional
prediction benchmark rather than a fully autonomous forecast.

Detailed final coefficients, component errors, runtime, memory, and feasibility
results are in [`automodel/phase4/`](automodel/phase4/). The complete experiment
record is in [`CONTEXT.md`](CONTEXT.md).

## Tests

```bash
pytest -q
```

## Acknowledgements

This work is supported by the European Union (European Research Council (ERC),
ALPS, 101039481). Views and opinions expressed are those of the authors only and
do not necessarily reflect those of the European Union or the ERC Executive
Agency. Neither the European Union nor the granting authority can be held
responsible for them.
