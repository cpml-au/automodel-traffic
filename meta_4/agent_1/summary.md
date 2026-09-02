# Meta 4 agent 1 summary: alternative convolution contrasts

All 15 prescribed fits completed synchronously on CPU with exact
`dctkit.dec.cochain.convolution`, full train times 0--63, and full validation
times 64--107. Fourteen fitted candidates passed the homogeneous nonlocal
feasibility check and both simulations; the attempt-3 Weidmann residual was
infeasible and had pathological simulation error. The row-wise sum of absolute
values of the full flux Jacobian was used as the nonlocal Rusanov speed bound.
The held-out test was never evaluated.

Each candidate kept the selected per-FD meta-3 incumbent and its coefficients
fixed. The appended factor had 13 nodes, and fitness counted the entire
unsimplified incumbent-plus-increment tree. The best feasible attempt in this
lineage for each FD was:

| FD | Best attempt | Appended factor and fitted `a` | Nodes | Validation E_data | Validation fitness | Meta-3 incumbent fitness | Fitness delta |
|---|---:|---|---:|---:|---:|---:|---:|
| Greenshields | 3 | `exp(0.2169680955*(conv_3-3*conv_1))` | 38 | 6.703686 | 7.083686 | 6.953732 | +0.129954 |
| IDM | 3 | `exp(-0.1546941142*(conv_3-3*conv_1))` | 27 | 5.275521 | 5.545521 | 5.417018 | +0.128503 |
| Weidmann | 2 | `exp(10.58117363*(conv_3-4*conv_1))` | 23 | 7.575684 | 7.805684 | 6.322293 | +1.483391 |
| Triangular | 3 | `exp(50.95367165*(conv_3-3*conv_1))` | 14 | 7.661809 | 7.801809 | 6.457489 | +1.344321 |
| Del Castillo | 1 | `exp(0.1878953909*(conv_3-2*conv_1))` | 34 | 5.519536 | 5.859536 | 5.730221 | +0.129315 |

No candidate beats its meta-3 incumbent on validation fitness. Residual
refinement gives tiny unpenalized improvements for Greenshields (0.000046) and
IDM (0.001497), and the window-2 contrast gives Del Castillo a 0.000685 raw
improvement. All are far smaller than the 0.13 complexity penalty. The
alternative window-2/window-4 normalizations substantially worsen Weidmann and
Triangular. The correct selection from this lineage is therefore to retain all
five meta-3 incumbents unchanged.

Complete expressions, constants, component errors, optimizer outcomes,
feasibility diagnostics, runtimes, peak RSS, and test-access flags are in each
attempt's `results.json`, `training.md`, and `evaluation.md`.
