# Meta 5 agent 1 summary: nonlinear convolution-contrast shapes

All 15 accepted fits completed synchronously on CPU using the current global
incumbents from `automodel/final_candidates.json`. Full train times 0--63 were
used for coefficient fitting and full validation times 64--107 for selection.
Every fit used exact `dctkit.dec.cochain.convolution`, two deterministic Powell
starts, at most 60 evaluations per start, the corrected homogeneous nonlocal
feasibility audit, and the row-wise absolute sum of the full flux Jacobian as
the Rusanov speed bound. Test times 108--179 were never accessed.

The initial attempt-1 coefficient range `[-1e6,1e6]` was rejected as an
optimizer preflight after every endpoint overflowed the full simulation. The
accepted square range `[-2e4,2e4]` follows the measured homogeneous contrast
scale and all accepted runs explicitly retain the finite zero start if bounded
Powell returns an invalid or worse endpoint. Details are in
`attempt_1/preflight.md`.

## Best feasible result per FD

| FD | Best lineage result | New coefficients | Total nodes | Validation E_data | Validation fitness | Global incumbent fitness | Fitness delta |
|---|---|---|---:|---:|---:|---:|---:|
| Greenshields | attempt 1, `g_inc*exp(a*C^2)` | `a=15665.902501` | 39 | 6.713621 | 7.103621 | 6.953732 | +0.149889 |
| IDM | attempt 1, `g_inc*exp(a*C^2)` | `a=-5623.601642` | 46 | 4.728441 | 5.188441 | 5.048745 | +0.139697 |
| Weidmann | no feasible nonlinear candidate | all three rejected as non-monotone | -- | -- | -- | 6.322293 | -- |
| Triangular | **attempt 2, `g_inc*exp(a*C+b*C^2)`** | **`a=-207.794278`, `b=14690.289134`** | **27** | **5.138741** | **5.408741** | 6.457489 | **-1.048748** |
| Del Castillo | attempt 1, `g_inc*exp(a*C^2)` | `a=19984.007284` | 35 | 5.508327 | 5.858327 | 5.730221 | +0.128106 |

Here `C = conv_3(rho,ones) - 3*conv_1(rho,ones)`. The only selected update is
the attempt-2 Triangular expression. It improves validation fitness by 16.24%
relative to identity and reduces raw validation `E_data` by 1.308748. Its train
`E_data` is 8.151970, and it passed positivity, monotone homogeneous velocity,
and finite train/validation simulation checks.

The squared IDM and Del Castillo terms produce small raw validation gains
(0.000303 and 0.011894 respectively), but these do not survive the complete
tree penalty. Greenshields worsens. Every Weidmann nonlinear endpoint violates
homogeneous velocity monotonicity, including the superficially low-error square
result, and is rejected. The cubic family supplies no feasible fitness gain;
for IDM the explicit zero start was retained.

Complete expressions, fitted constants, component errors, optimizer messages,
tree sizes, runtimes, peak RSS, feasibility diagnostics, and
`test_evaluated=false` flags are recorded in each attempt's `results.json`,
`training.md`, and `evaluation.md`.
