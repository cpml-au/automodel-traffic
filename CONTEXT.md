# Automodel Traffic — project context

> **Living document.** Each Automodel phase updates this file. A fresh agent must be
> able to resume the search from this file without conversation history.

## Goal

Discover an interpretable multiplicative correction for each of the five basic
fundamental diagrams used by this repository (Greenshields, IDM, Weidmann,
Triangular, and Del Castillo) on the I80 prediction problem. Each corrected model
has the README ansatz

```text
q_corrected(rho) = q_baseline(rho; fixed_baseline_coefficients) * g(rho; c),
```

where `g` may use the typed algebraic/DEC primitives already exposed by
`src/sr_traffic/sr/sr_traffic.yaml` and
`src/sr_traffic/sr/primitives.py`. Track every attempted expression, fitted
constants, complexity, runtime, memory, and train/validation score.

### Non-negotiable search-integrity constraint

Never inspect, import, search for, or use recovered symbolic models or model
information from the upstream SR-Traffic repository or the SR-Traffic paper.
Existing baseline FD definitions and calibrated baseline coefficients in this
local task repository may be used. DCTKit documentation and source may be used
to understand DEC operators. The final I80 test interval must remain untouched
until Automodel Phase 4.

## Problem definition

- **Input:** dimensionless density `rho` on the nodes of the one-dimensional
  simplicial complex, with observed range approximately `[0.0084, 1.0]`.
- **Output:** a dimensionless multiplicative cochain `g(rho; c)` used to correct
  each baseline flux. The traffic solver then predicts density and velocity.
- **Fixed components:** the I80 preprocessing, boundary/initial conditions,
  DEC mesh, Godunov solver, basic FD definition, and each FD's existing I80
  prediction coefficients remain fixed during structural search. Only constants
  inside `g` are tuned.
- **Physical/implementation constraints:** expressions and simulations must be
  finite. Corrected velocity must be non-negative on simulated states and
  non-increasing on each baseline's physical density domain (up to its calibrated
  jam/support density). Several fixed baselines are negative or undefined beyond
  that domain, so a correction is not required to repair their out-of-domain
  extension. Prefer corrections near one and parsimonious expressions. Any
  additional feasibility checks must be logged.

## Data

- **Files:** `src/sr_traffic/data/I80/NGSIM_I80_4pm_{Density,Velocity,Flow}_Data.txt`.
- **Preprocessing:** `preprocess_data("I80")` nondimensionalizes density by its
  observed maximum and velocity by 100 ft/s, creates the 1-D simplicial complex,
  interpolates boundary conditions, and refines each 5-second data interval to
  40 Godunov steps.
- **Key variables:** density and velocity determine the score. Flow is computed
  and reported diagnostically but is not part of the selection score.
- **Dataset shape:** 79 spatial rows by 180 sampled times; the scored interior
  has 75 spatial positions.
- **Selection score:** validation `E_fitness = E_data + 0.01 * tree_nodes`, where
  `E_data = 0.5 * (E_rho + E_v)` and each component is 100 times relative squared
  error. Report unpenalized `E_data` and node count separately. Rank candidates
  primarily by validation `E_fitness`, using validation `E_data` and simplicity
  to break practically negligible differences.
- **Splits:** chronological, as implemented by `build_dataset(...,
  task="prediction")`: train times 0–63 (64 times, 4,800 rows), validation times
  64–107 (44 times, 3,300 rows), train+validation times 0–107 (108 times, 8,100
  rows), and test times 108–179 (72 times, 5,400 rows). No shuffling.
- **External check (Phase 4):** the final 72-time I80 test block. Do not compute
  its score during Phases 1–3. After choosing one expression per baseline using
  validation only, refit its correction constants on train+validation and perform
  one test evaluation.
- **Benchmark caveat:** preprocessing scales density by the maximum across all
  180 times and simulations use observed boundary density throughout the future
  interval. This behavior is preserved for repository benchmark compatibility,
  but the external check is therefore conditional on known boundaries and uses
  test-aware normalization.

## Objective and metrics

- **Constant-fitting objective:** training `E_data` for a candidate structure.
- **Structural selection:** validation `E_fitness` with `reg_param = 0.01` per
  expression-tree node.
- **Reported metrics:** train and validation `E_data`, validation `E_fitness`,
  density and velocity error components, expression/tree size, number of fitted
  constants, feasibility status, optimizer convergence/restarts, wall time, and
  peak RSS. Phase 4 additionally reports test `E_data` and component errors.
- **Optimizer-noise controls:** fixed and logged seeds, multiple starts for
  promising candidates, and re-evaluation before accepting small improvements.
- **Baseline resource measurement:** identity-correction evaluations took
  approximately 0.13–0.69 seconds per split after compilation on CPU; the
  measurement process peaked at about 732 MB RSS.

## Parameter optimization routine

The repository implementation in `src/sr_traffic/sr/sr_traffic.py` compiles a
typed GP expression, tunes its constants in `[-10, 10]` with PyGMO's simple
evolutionary algorithm (population 10, 10 generations), rejects velocity curves
with positive derivative, and evaluates the resulting Godunov simulation. The
Automodel search may use the same routine or a reproducible multi-start SciPy/
PyGMO alternative with bounds tailored to a proposed expression. All settings,
seeds, convergence information, and scores must be logged.

## Phase 1 feasibility

- **Runtime:** Python 3.12 in Micromamba environment `sr_traffic`.
- **Key libraries verified:** JAX 0.11.1, DCTKit 0.0.0, Flex 0.3.0, DEAP 1.4,
  PyGMO 2.19.8, NumPy 2.5.2, SciPy 1.18.0, scikit-learn 1.9.0, SymPy 1.14.0.
- **Execution:** CPU-only JAX is available. Agents share read/write/execute access
  to the repository. The Automodel workflow may therefore run parallel,
  isolated candidate explorations.

## Mock model / baseline

The Phase 2 mock implementation is in `automodel/model.py`, with calibration and
evaluation in `automodel/pipeline.py`. It uses the explicit parameterized
correction `g(rho; c0) = c0`, whose default `c0 = 1` is the identity correction.
Baseline scores below were measured without touching the test split:

| Baseline | Train `E_data` | Validation `E_data` | Train+validation `E_data` |
|---|---:|---:|---:|
| Greenshields | 7.203543 | 9.610405 | 8.210600 |
| IDM | 8.438750 | 5.945012 | 7.392908 |
| Weidmann | 7.232718 | 6.480163 | 6.918967 |
| Triangular | 9.223384 | 6.447489 | 8.059334 |
| Del Castillo | 7.048800 | 6.628030 | 6.872657 |

### Phase 2 end-to-end verification

An independent sub-agent ran the Triangular mock pipeline on eight train and
eight validation times. Bounded scalar optimization converged after eight
evaluations to `c0 = 0.7781295656`. Training `E_data` was 11.329039 and
validation `E_data` was 9.103994; total reported pipeline runtime was 2.003 s and
peak RSS was 623.36 MB. Density, velocity, and flow were finite and non-negative.
The identity correction scored 4.308484 on the same small validation subset, so
the fitted constant clearly overfit; Phase 3 must use the complete train and
validation intervals and accept structures based on validation transfer, not
training improvement. Structured output is stored in
`automodel/phase2_results.json`. The sub-agent verified read/write/execute access
and confirmed that no test prediction or score was computed.

### Known pipeline constraints for Phase 3

- Force this checkout with `PYTHONPATH="$PWD/src:$PWD"`; the environment contains
  a stale editable install elsewhere.
- Do not use the stock `validate: false` SR CLI in Phases 1–3 because it evaluates
  the test split automatically.
- The stock validation `score()` omits the configured length penalty, and PyGMO
  constant fitting does not set an explicit seed. The Automodel harness must add
  the penalty itself and log deterministic seeds.
- Spatial DEC corrections (`cob`, `del`, flats, convolution) create a nonlocal
  flux Jacobian, while the current Godunov implementation keeps only its
  diagonal. Search pointwise corrections first; treat nonlocal DEC terms only as
  a separately labeled experiment after repairing the wave-speed handling.
- The grammar requests primitive base name `St1`, but the installed Flex version
  exposes `St_one`; direct Hodge-star primitives are absent until that mismatch
  is corrected. `del` remains metric-aware internally.

## Search plan

- **M × S × I:** 2 meta-iterations × 3 parallel sub-agents × 3 sequential
  structural attempts. Each attempt fits all five basic FDs.
- **Directory layout:**
  `meta_m/agent_s/attempt_i/{plan.md,model.py,training.md,evaluation.md,results.json}`.
- **Common protocol:** full times 0–63 for fitting; full times 64–107 for
  validation; deterministic Powell optimization with two starts and logged
  bounds/seeds; no access to times 108–179. Re-evaluate shortlisted models from
  the root process before choosing winners.
- **Diversification:** meta-iteration 1 compares (1) positive exponential
  polynomials in density, (2) positive exponentials anchored to one at the
  baseline's physical jam/support density, and (3) saturating positive
  rational-exponential terms. Meta-iteration 2 will refine the families that
  transfer best per baseline and probe coefficient/complexity reductions.

## Hypotheses for meta_1

- Greenshields and Triangular need asymmetric curvature or capacity adjustment;
  a one- or two-term exponential correction may improve validation while
  retaining their baseline zeros/support.
- Jam-anchoring may prevent corrections from compensating for baseline boundary
  behavior and therefore generalize better, even if its training score is worse.
- Saturating exponents may avoid excessive high-density distortion and yield a
  simpler validation improvement for IDM, Weidmann, and Del Castillo.
- Pointwise corrections are the defensible first search because the current
  Rusanov derivative is consistent only with a local fundamental diagram.

## Meta_1 outcome (2026-09-01)

All 45 planned fits completed with finite, physically feasible results and no
test access. Root re-evaluation reproduced the shortlist. See
`meta_1/review.md` and each attempt's `results.json` for every expression and
coefficient set.

| Baseline | Best meta-1 correction | Validation `E_data` | Validation `E_fitness` | Identity `E_fitness` |
|---|---|---:|---:|---:|
| Greenshields | `exp(0.146288*rho - 0.885592*rho^2 + 0.939214*rho^3)` | 8.045311 | 8.205311 | 9.620405 |
| IDM | identity | 5.945012 | 5.955012 | 5.955012 |
| Weidmann | `exp(0.136544*rho*(1-rho/0.806121))` | 6.222293 | 6.322293 | 6.490163 |
| Triangular | identity | 6.447489 | 6.457489 | 6.457489 |
| Del Castillo | identity | 6.628030 | 6.638030 | 6.638030 |

**Key findings:** positive pointwise corrections clearly help Greenshields and a
simple jam-anchored term helps Weidmann. Added flexibility overfits IDM and
Triangular. Del Castillo's best raw gain is erased by its expression penalty.
Many multi-parameter fits exhausted their strict evaluation budget, so meta 2
will recondition the basis and use larger budgets for promising simple forms.

**Hypotheses for meta_2:** compare intercept-bearing exponentials, bases centered
at each baseline's characteristic density, and low-node direct polynomial or
rational multipliers. These may decouple global scaling from shape and allow
small gains to survive the complexity penalty.

## Meta_2 outcome (2026-09-01)

All 45 additional fits completed successfully (90 total), again with no test
access. Root re-evaluation reproduced the three non-identity winners. See
`meta_2/review.md`, `automodel/final_candidates.json`, and the consolidated
`automodel/leaderboard.{csv,json}`.

| Baseline | Selected correction after meta 2 | Validation `E_data` | Validation `E_fitness` | Relative fitness change from identity |
|---|---|---:|---:|---:|
| Greenshields | `exp(0.023123 - 0.088317*rho - 0.150081*rho^2)` | 7.507391 | 7.627391 | -20.72% |
| IDM | identity | 5.945012 | 5.955012 | 0.00% |
| Weidmann | `exp(0.136544*rho*(1-rho/0.806121))` | 6.222293 | 6.322293 | -2.59% |
| Triangular | identity | 6.447489 | 6.457489 | 0.00% |
| Del Castillo | `exp(0.005495)` | 6.583956 | 6.603956 | -0.51% |

**Key findings:** the intercept-bearing quadratic is the dominant Greenshields
improvement; the meta-1 jam-anchored Weidmann term remains best; Del Castillo
benefits only from a tiny constant scale. None of the 18 tried structures beats
identity for IDM or Triangular under chronological validation. Centered and
direct rational forms did not improve an incumbent.

## Extended search plan (meta 3–4)

At the user's request, Phase 3 continues for two additional meta-iterations with
the same `S=3`, `I=3` structure, fitting every candidate to all five FDs. Meta 3
diversifies across convolution-only, explicit Hodge-star, and incumbent-plus-
convolution families. Meta 4 will refine the strongest DEC structures and retain
a non-convolution control direction.

### DEC infrastructure fixes before meta 3

- Changed the YAML primitive base from obsolete `St1` to Flex's generated base
  `St_one`. The SR grammar now registers `St_oneP0`, `St_oneP1`, `St_oneD0`, and
  `St_oneD1`; a regression test compiles and evaluates
  `St_oneD1(St_oneP0(rho))`.
- Replaced the diagonal-only flux-Jacobian derivative with the row-wise absolute
  Jacobian sum used as a conservative Rusanov speed bound. This is unchanged for
  local pointwise FDs after the solver absolute value and includes off-diagonal
  coupling introduced by convolution.
- Added homogeneous-state feasibility checks for nonlocal corrections so the
  valid-convolution downstream padding is not confused with a constitutive
  monotonicity failure. Full simulations remain the final finite/feasibility
  gate.
- `tests/test_sr_primitives.py` and `tests/test_flux_speed.py` pass (4 tests).

### Hypotheses for meta 3

- `conv_3(rho, ones)` can represent a short downwind density average unavailable
  to pointwise polynomials and may help IDM/Triangular transfer.
- A contrast between `conv_3` and `conv_1` can encode a local queue-front signal
  while cancelling much of the raw density level.
- Explicit Hodge-star compositions can supply metric-weighted linear/quadratic
  features; on this uniform 1-D mesh some will be redundant, which the validation
  penalty should expose.
- Adding a single convolution term to the previous per-FD incumbent is a lower-
  variance alternative to rebuilding each multiplier from scratch.

## Meta_3 outcome (2026-09-01)

All 45 fits completed; 44 were feasible and one non-monotone Weidmann hybrid was
rejected. No test access occurred. Root re-evaluation reproduced the winners.
See `meta_3/review.md` and the per-attempt artifacts.

| Baseline | Best correction after meta 3 | Validation `E_data` | Validation `E_fitness` | Fitness change from meta 2 |
|---|---|---:|---:|---:|
| Greenshields | `g_meta2*exp(-293.458109*(conv_3-3*conv_1))` | 6.703732 | 6.953732 | -8.83% |
| IDM | `exp(-46.991436*(conv_3-3*conv_1))` | 5.277018 | 5.417018 | -9.03% |
| Weidmann | retained incumbent | 6.222293 | 6.322293 | 0.00% |
| Triangular | identity | 6.447489 | 6.457489 | 0.00% |
| Del Castillo | `g_meta2*exp(-2.121812*conv_3-256.869230*(conv_3-3*conv_1))` | 5.520221 | 5.730221 | -13.23% |

**Key findings:** convolution contrast, not raw density averaging, supplies the
gain. Explicit `St_one` expressions are now fully usable but did not win. Meta 4
will vary the contrast normalization/gating, combine it with a Hodge-weighted
feature, and run a pointwise nonlinear control family for every FD.

## Meta_4 outcome (2026-09-01)

All 45 additional fits completed (180 across meta 1--4); 42 were feasible and
three Weidmann Hodge/convolution or contrast candidates were rejected for
non-monotonicity or non-finite simulations. No test access occurred. See
`meta_4/review.md`, `automodel/final_candidates.json`, and the consolidated
leaderboard for the complete expression history.

| Baseline | Selected correction after meta 4 | Validation `E_data` | Validation `E_fitness` | Fitness change from identity |
|---|---|---:|---:|---:|
| Greenshields | retained meta-3 convolution contrast | 6.703732 | 6.953732 | -27.72% |
| IDM | `g_meta3*exp(-119537.388349*hquad*(conv_3-3*conv_1))` | 4.728745 | 5.048745 | -15.22% |
| Weidmann | retained jam-anchored incumbent | 6.222293 | 6.322293 | -2.59% |
| Triangular | identity | 6.447489 | 6.457489 | 0.00% |
| Del Castillo | retained meta-3 convolution model | 5.520221 | 5.730221 | -13.68% |

Here `hquad = St_oneD1(SquareD1(St_oneP0(rho)))`. Root x64 evaluation of the
IDM winner reproduced validation `E_data = 4.728762`, fitness `5.048762`, and
the finite/nonlocal feasibility result. This demonstrates that the repaired
Hodge-star primitives are usable both in the grammar and in a selected model.
Alternative contrast normalizations and pointwise square-root controls did not
beat the retained incumbents.

### Current best-model pointers

The five Phase-3 selections and their constants are recorded in
`automodel/final_candidates.json`. The full 180-fit expression trail is in
`automodel/leaderboard.csv` and `automodel/leaderboard.json`; per-attempt plans,
training metrics, evaluations, diagnostics, and expressions remain under
`meta_1` through `meta_4`. The held-out I80 test interval (times 108--179) has
not been evaluated and remains available for Phase 4 after user review.

## Meta_5 outcome and Phase-3 freeze (2026-09-01)

The user requested one final meta-iteration followed by Phase 4. Meta 5 ran 45
more fits for all five FDs (225 total search fits): 33 were feasible and 12
pathological/non-monotone endpoints were rejected. No test access occurred.

The new winner is the 16-node Triangular correction

```text
g(rho) = exp(-921.970199792*rho*(conv_3(rho,ones)-3*conv_1(rho,ones))).
```

It scores validation `E_data = 5.067508` and fitness `5.227508`, versus identity
fitness `6.457489` (19.05% lower). Root JAX x64 evaluation reproduced
`E_data = 5.067495`, fitness `5.227495`, and the feasibility result. A distinct
signed-plus-quadratic contrast also improved Triangular but was worse in both
raw error and complexity. No meta-5 expression beat the other four incumbents.

### Frozen structures for Phase 4

| Baseline | Source model | Validation E_data | Validation fitness |
|---|---|---:|---:|
| Greenshields | `meta_3/agent_3/attempt_2/model.py` | 6.703732 | 6.953732 |
| IDM | `meta_4/agent_2/attempt_2/model.py` | 4.728745 | 5.048745 |
| Weidmann | `meta_1/agent_2/attempt_1/model.py` | 6.222293 | 6.322293 |
| Triangular | `meta_5/agent_2/attempt_1/model.py` | 5.067508 | 5.227508 |
| Del Castillo | `meta_3/agent_3/attempt_3/model.py` | 5.520221 | 5.730221 |

`automodel/final_candidates.json` is the machine-readable freeze. The complete
225-fit history is in `automodel/leaderboard.{csv,json}` and the review is in
`meta_5/review.md`. The stopping rationale is the user's explicit instruction
to move to Phase 4 after this round. Test times 108--179 remain untouched.

## Phase 4 — Finalize (2026-09-01)

### Final decision and usage

The five frozen structures were refit on train+validation times 0--107, then
each received exactly one evaluation on test times 108--179. All five are
accepted as the completed Automodel correction set: simulations were finite,
the relevant physical feasibility check passed, and no test result was used to
change a structure or coefficient. The parameterized production entry points
are `Greenshields_corrected_flux`, `IDM_corrected_flux`,
`Weidmann_corrected_flux`, `triangular_corrected_flux`, and
`del_castillo_corrected_flux` in `src/sr_traffic/fd/diagrams.py`. Their default
correction coefficients are the Phase-4 I80 prediction refits.

| Baseline | Train+validation E_data | Test E_rho | Test E_v | Test E_data | Test runtime (s) | Peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Greenshields | 5.878854 | 6.483692 | 7.700181 | 7.091936 | 0.926 | 2135.6 |
| IDM | 6.441920 | 6.570096 | 7.664510 | 7.117303 | 2.273 | 3091.1 |
| Weidmann | 6.777875 | 7.015417 | 7.902568 | 7.458993 | 0.628 | 772.7 |
| Triangular | 6.750185 | 6.547068 | 7.689783 | 7.118426 | 0.978 | 831.1 |
| Del Castillo | 5.868820 | 6.421893 | 7.137335 | 6.779614 | 1.081 | 1780.4 |

Detailed optimizer runs, final coefficients, component errors, total fit
runtime, and memory are in `automodel/phase4/{baseline}.json`; the concise
report is `automodel/phase4/README.md`.

### Interpretation and acceptance criteria

- All hard criteria passed: finite multiplier/simulation, positive multiplier,
  non-negative/non-increasing homogeneous corrected velocity, and exactly one
  test evaluation per frozen model.
- Test E_data is 5.79%, 50.51%, 19.88%, 40.47%, and 22.81% above the respective
  validation selection score for Greenshields, IDM, Weidmann, Triangular, and
  Del Castillo. The IDM and Triangular selection gains were therefore
  optimistic across the chronological shift.
- Against the post-refit train+validation objective, the respective test gaps
  are 20.63%, 10.48%, 10.05%, 5.46%, and 15.52%. This is a more direct measure
  of the final fixed-parameter distribution shift.
- Del Castillo has the lowest test E_data (6.779614), but this descriptive
  ordering must not be used for another round of model selection on the same
  test block.

### Limitations

- The test split is now consumed. Any future structural search needs fresh
  external data or a new predeclared check; reusing this block would overfit it.
- The protocol did not spend a second test evaluation on uncorrected baselines,
  so it does not claim a paired test-set improvement for every correction.
- I80 preprocessing uses full-series density normalization and observed future
  boundary density, making this a conditional benchmark rather than a fully
  autonomous forecast.
- IDM's feasible coefficient basin was too narrow for the scaled Powell steps;
  the robust explicit-start safeguard retained its already frozen coefficients.

### Results-script integration

`src/sr_traffic/fd/results.py` now registers the five I80 prediction models as
`automodel-Greenshields`, `automodel-IDM`, `automodel-Weidmann`,
`automodel-Triangular`, and `automodel-Del Castillo`. It uses the complete
nonlocal Jacobian speed bound, produces separate Automodel figures, and writes
both the legacy relative-error tables and README-defined `E_data` score tables.
Generated artifacts are under `results/I80/prediction/`.

The combined held-out `E_data` comparison is: Greenshields 9.230 versus
automodel-Greenshields 7.092; IDM 6.957 versus automodel-IDM 7.117; Weidmann
7.651 versus automodel-Weidmann 7.459; Triangular 7.919 versus
automodel-Triangular 7.118; and Del Castillo 7.197 versus automodel-Del Castillo
6.780. Thus four of five corrections improve their paired baseline on test;
the IDM correction does not.

### Upstream SR comparison

After the Automodel search and test protocol were closed, the user explicitly
requested a reproduced comparison with the official upstream SR models. The
`cpml-au/SR-Traffic` results script was run from commit
`3bf285ab1d6b2b61f0c0f27c0d80e42633b84ac4` in an isolated temporary clone.
Only result tables were retained in this repository; no upstream expressions or
implementation files were copied. The five SR-only legacy error rows are in
`results/I80/prediction/upstream_sr_error_table.md`, and the common interior
`E_data` comparison of all five baselines, five upstream SR models, and five
Automodel models is in `results/I80/prediction/model_comparison_table.md`.
