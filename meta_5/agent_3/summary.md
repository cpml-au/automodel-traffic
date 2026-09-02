# Meta-iteration 5 agent 3 summary

All 15 requested mixed repaired-Hodge/convolution fits completed against the
fixed global incumbents in `automodel/final_candidates.json`. Constants were
fitted on full train times 0--63 and structures compared on full validation
times 64--107. Test times 108--179 were never evaluated.

## Best candidate in this lineage versus global incumbent

| FD | Best attempt | Added factor and effective parameters | Nodes | Train E_data | Validation E_data | Fitness | Incumbent fitness | Change | Feasible |
|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| Greenshields | 1 | `exp(a*C+b*hquad*C)`, `a=48.62308613`, `b=-42852.50609` | 55 | 6.255026 | 6.577243 | 7.127243 | 6.953732 | +0.173510 | yes |
| IDM | 1 | `exp(a*C+b*hquad*C)`, `a=86.08646158`, `b=-29600.21354` | 62 | 7.405681 | 4.634424 | 5.254424 | 5.048745 | +0.205680 | yes |
| Weidmann | 3 | `exp(a*(79*hquad-rho^2)*C)`, `a=0` | 33 | 7.205753 | 6.222293 | 6.552293 | 6.322293 | +0.230000 | no |
| Triangular | 3 | `exp(a*(79*hquad-rho^2)*C)`, `a=0` | 24 | 9.223384 | 6.447489 | 6.687489 | 6.457489 | +0.230000 | yes |
| Del Castillo | 1 | `exp(a*C+b*hquad*C)`, `a=71.14073361`, `b=-54261.81513` | 51 | 6.232488 | 5.440630 | 5.950630 | 5.730221 | +0.220409 | yes |

Lower fitness is better. No candidate displaces a global incumbent. Attempts 1
and 2 reduce raw validation error for several FDs, but their complete repeated
Hodge/convolution trees cost 30 and 34 additional nodes. The centered
one-parameter attempt 3 retains `a=0` for every FD and therefore only adds its
23-node penalty.

## Diagnostics and search decisions

- `hlin = St_oneD1(St_oneP0(rho))` and
  `hquad = St_oneD1(SquareD1(St_oneP0(rho)))` compiled against the repaired live
  grammar. The factor sizes were independently compiled as 29, 33, and 22
  nodes, with one additional `CMulP0` attachment in every candidate.
- Powell used two deterministic starts and at most 60 evaluations per start.
  Strongly different coefficient scales were reconditioned internally; all
  stored coefficients are the effective `a,b` appearing in the expressions.
- The corrected homogeneous nonlocal audit finds the global Weidmann incumbent
  has a small low-density velocity increase. Every new term is multiplied by
  `C`, which vanishes at the homogeneous central node, so this lineage cannot
  repair it. All Weidmann entries therefore explicitly record the evaluated
  zero/incumbent fallback as infeasible rather than claiming a feasible fit.
- Attempt 2 Del Castillo reached a finite training candidate but produced a
  nonfinite validation simulation. It is rejected and fully retained in the
  attempt result for auditability.
- All other final candidates listed in the attempt files have recorded finite
  train/validation status and homogeneous feasibility diagnostics. Wave speeds
  use the row-wise absolute sum of the full flux Jacobian, including off-diagonal
  Hodge/convolution coupling.

Aggregate recorded fit runtime was 963.361 seconds, and peak RSS was 4364.4 MB.
Complete coefficients, component errors, full expressions, optimizer messages
and evaluation counts, runtime/RSS, feasibility details, and
`test_evaluated=false` flags are in each attempt's `results.json`,
`training.md`, and `evaluation.md`.
