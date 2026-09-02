# Automodel Traffic

<p align="center">
  <img src="readme_figure.png" alt="Automodel Traffic logo" width="100%">
</p>

Automodel Traffic is a completed agent-guided model-discovery experiment for
macroscopic freeway traffic. Starting from five classical fundamental diagrams,
we searched for small, interpretable corrections that improve I80 traffic
prediction while preserving physical feasibility and keeping the final test
interval isolated from model selection.

The experiment is fully recorded in this repository: 225 candidate fits, their
plans and diagnostics, the five frozen models, the one-shot held-out evaluation,
and production-ready corrected flux functions.

## What we did

For each Greenshields, IDM, Weidmann, Triangular, and Del Castillo baseline, we
searched for a multiplicative correction

```text
q_corrected(rho) = q_baseline(rho; fixed baseline coefficients) * g(rho; c).
```

The baseline traffic solver, calibrated baseline coefficients, initial and
boundary conditions, mesh, and data preprocessing stayed fixed. Only the form
of `g` and its constants changed.

The committed experiment used a structured Automodel loop:

1. Three agents explored different correction families in parallel.
2. Each agent made three sequential structural attempts per round, using the
   previous attempt's diagnostics to choose the next form.
3. Every structure was fitted on the chronological training interval and ranked
   on a separate validation interval using error plus a complexity penalty.
4. A root review reproduced the strongest candidates and chose the next round's
   search directions.
5. After five rounds, one structure per baseline was frozen, refitted on
   training plus validation data, and evaluated exactly once on the held-out
   test interval.

That is `5 rounds x 3 agents x 3 attempts x 5 baselines = 225` fitted
candidates. Agents proposed explicit, reviewable structures, while deterministic
bounded optimization fitted their constants. The search moved from pointwise
exponential corrections to
short-range discrete-convolution and Hodge-star features when validation results
supported that added structure.

All candidates had to produce finite simulations and physically admissible
velocity behavior. Nonlocal corrections also used a conservative row-wise flux
Jacobian bound so their spatial coupling was represented in the traffic solver.

## Evaluation protocol

The tracked dataset is the I80 4 p.m. series with 180 chronological samples.
There was no shuffling:

| Split | Time indices | Samples | Purpose |
|---|---:|---:|---|
| Train | 0--63 | 64 | Fit candidate constants |
| Validation | 64--107 | 44 | Select correction structures |
| Test | 108--179 | 72 | One final check after freezing |

Density and velocity errors are normalized relative squared errors:

```text
E_rho  = 100 * sum((rho_model - rho_data)^2) / sum(rho_data^2)
E_v    = 100 * sum((v_model - v_data)^2) / sum(v_data^2)
E_data = 0.5 * (E_rho + E_v)
```

Lower is better. During structure selection, validation fitness was
`E_data + 0.01 * tree_nodes`; the final test table reports unpenalized `E_data`.

## Held-out I80 results

Four of the five Automodel corrections improved their paired baseline on the
held-out test interval. The IDM correction did not, which is retained here
rather than hidden or selected away after seeing the test result.

| Baseline | Baseline test `E_data` | Automodel test `E_data` | Relative change |
|---|---:|---:|---:|
| Greenshields | 9.230426 | **7.091936** | -23.17% |
| IDM | **6.957441** | 7.117303 | +2.30% |
| Weidmann | 7.651073 | **7.458993** | -2.51% |
| Triangular | 7.918859 | **7.118426** | -10.11% |
| Del Castillo | 7.196903 | **6.779614** | -5.80% |

Del Castillo achieved the lowest absolute test error. The test block is now
consumed: any further structural search needs a new external dataset or a new
predeclared check.

Detailed refit coefficients, component errors, runtime, memory, and feasibility
results are in [`automodel/phase4/`](automodel/phase4/). The full selection
history is available in [`automodel/leaderboard.csv`](automodel/leaderboard.csv)
and [`automodel/leaderboard.json`](automodel/leaderboard.json).

## Final corrections

| Baseline | Selected structure | Fitted constants |
|---|---|---:|
| Greenshields | Density polynomial plus short-range spatial contrast | 4 |
| IDM | Spatial contrast plus a quadratic Hodge-gated contrast | 2 |
| Weidmann | Pointwise term anchored at jam density | 1 |
| Triangular | Density-gated spatial contrast | 1 |
| Del Castillo | Short convolution plus spatial contrast | 3 |

Exact expressions and parameter bounds are defined in
[`automodel/final_models.py`](automodel/final_models.py). Their Phase-4 refit
coefficients are the defaults of the corrected flux functions in
[`src/sr_traffic/fd/diagrams.py`](src/sr_traffic/fd/diagrams.py):

```python
from sr_traffic.fd.diagrams import triangular_corrected_flux

# Uses the frozen I80 Phase-4 correction coefficient by default.
flux = triangular_corrected_flux(rho, V_0, l_eff, T)
```

Equivalent entry points are provided for Greenshields, IDM, Weidmann, and Del
Castillo, and their correction coefficients can be overridden explicitly.

## Reproduce the final benchmark

Create the Python 3.12 environment with Mamba or Micromamba:

```bash
micromamba env create -f environment.yaml
micromamba activate automodel-traffic
```

Run the frozen baselines and Automodel corrections on I80 prediction:

```bash
PYTHONPATH="$PWD/src:$PWD" \
python src/sr_traffic/fd/results.py --road_name I80 --task prediction
```

Figures and Markdown/LaTeX score tables are written to
`results/I80/prediction/`. Generated result files are intentionally ignored by
Git.

Run the regression tests with:

```bash
PYTHONPATH="$PWD/src:$PWD" pytest -q
```

## Repository map

- [`CONTEXT.md`](CONTEXT.md) is the complete experimental narrative, protocol,
  decisions, results, and limitations.
- [`automodel/final_candidates.json`](automodel/final_candidates.json) is the
  validation-time freeze created before test access.
- [`automodel/final_models.py`](automodel/final_models.py) defines the five
  selected structures.
- [`automodel/phase4/`](automodel/phase4/) contains the final refit and held-out
  evaluation records.
- `meta_1/` through `meta_5/` contain every agent plan, implementation, fit,
  diagnostic, and round review.
- [`src/sr_traffic/fd/diagrams.py`](src/sr_traffic/fd/diagrams.py) exposes the
  production corrected fluxes; `sr_traffic` remains the internal numerical
  package namespace.

## Scope and limitations

- The committed data and discovered correction coefficients cover I80
  prediction only.
- Full-series density normalization and observed future boundary density are
  retained for benchmark compatibility, so this is a conditional prediction
  benchmark rather than a fully autonomous forecast.
- Validation gains were optimistic for some model families; the table above is
  the final held-out result and should not be reused for selection.
- The correction search improved four baselines, not all five.

## Acknowledgements

This work is supported by the European Union (European Research Council (ERC),
ALPS, 101039481). Views and opinions expressed are those of the authors only and
do not necessarily reflect those of the European Union or the ERC Executive
Agency. Neither the European Union nor the granting authority can be held
responsible for them.
