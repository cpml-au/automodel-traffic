# Agent 2 jam-anchored lineage summary

All 15 prescribed fits completed on the full I80 training split and were scored
on the full validation split. No held-out test prediction or score was computed.
All candidates were finite, strictly positive in their multiplier, and passed
the physical-domain velocity feasibility check.

## Best attempt per baseline

Ranking below is by validation `E_fitness = E_data + 0.01*tree_nodes` within
this three-attempt lineage. The identity `E_data` values are copied from the
Phase 2 project context only to show validation transfer; they were not rerun.

| Baseline | Best attempt | Constants | Validation E_rho | Validation E_v | Validation E_data | E_fitness | Identity E_data | Data-error change |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Greenshields | 2 | a=0.05870392846, b=-0.4948637636 | 9.938209 | 7.595424 | 8.766816 | 8.916816 | 9.610405 | -0.843589 |
| IDM | 3 | a=-1.436972907, b=-0.6452147119, c=4.435581740 | 6.711102 | 6.083894 | 6.397498 | 6.607498 | 5.945012 | +0.452486 |
| Weidmann | 1 | a=0.1365440521 | 5.923585 | 6.521000 | 6.222293 | 6.322293 | 6.480163 | -0.257870 |
| Triangular | 3 | a=-0.04847206901, b=-2.622664330, c=4.921472140 | 6.540417 | 7.919237 | 7.229827 | 7.439827 | 6.447489 | +0.782338 |
| Del Castillo | 1 | a=-0.05711308322 | 6.753607 | 6.547346 | 6.650476 | 6.750476 | 6.628030 | +0.022446 |

Negative data-error change is an improvement. This lineage improves raw
validation error for Greenshields and Weidmann. The best anchored forms for IDM,
Triangular, and Del Castillo do not beat the corresponding identity baselines.

## Diagnostics and failures

- Attempt runtimes were 69.279 s, 83.658 s, and 83.294 s, respectively, after
  evaluator construction.
- There were no simulation, finiteness, or feasibility failures.
- Twelve of 15 best-restart Powell results reported that the 30-evaluation cap
  was reached (two in attempt 1, all five in attempts 2 and 3). Their fitted
  candidates and final full-split evaluations are valid, but the stop reason
  should not be interpreted as optimizer convergence.
- Attempt 3 Weidmann reduces exactly to attempt 1 (`b=c=0`) and therefore loses
  by its larger tree penalty. Attempt 3 Del Castillo similarly reduces to
  attempt 2 (`c=0`); attempt 1 remains best for that baseline by validation
  fitness.

Per-attempt expressions, exact constants, component/data errors, seeds,
optimizer messages/evaluation counts, runtime/RSS, and feasibility flags are in
each `results.json`, with readable training and validation tables in
`training.md` and `evaluation.md`.
