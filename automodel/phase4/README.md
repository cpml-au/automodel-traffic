# Phase 4 final evaluation

The five structures were frozen after meta-iteration 5, their correction
coefficients were refit on I80 prediction times 0--107, and each model was then
evaluated exactly once on held-out times 108--179. Test error did not influence
structure or coefficient selection.

| Baseline | Train+validation E_data | Test E_rho | Test E_v | Test E_data | Test runtime (s) | Peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Greenshields | 5.878854 | 6.483692 | 7.700181 | 7.091936 | 0.926 | 2135.6 |
| IDM | 6.441920 | 6.570096 | 7.664510 | 7.117303 | 2.273 | 3091.1 |
| Weidmann | 6.777875 | 7.015417 | 7.902568 | 7.458993 | 0.628 | 772.7 |
| Triangular | 6.750185 | 6.547068 | 7.689783 | 7.118426 | 0.978 | 831.1 |
| Del Castillo | 5.868820 | 6.421893 | 7.137335 | **6.779614** | 1.081 | 1780.4 |

## Refit coefficients

- Greenshields `(c0, rho, rho_squared, contrast)`:
  `(0.177745502, -0.307502565, -0.090030856, -224.635576032)`.
- IDM `(contrast, quadratic_hodge_contrast)`:
  `(-46.991435975, -119537.388349330)`. The scaled Powell runs left the
  narrow feasible basin, so explicit feasible-start retention correctly kept
  the selection coefficients.
- Weidmann `(jam_anchored)`: `(0.209108182)`.
- Triangular `(density_contrast)`: `(-921.972502729)`.
- Del Castillo `(c0, conv_3, contrast)`:
  `(0.073885438, -3.328070298, -227.983101130)`.

Full optimizer, component-error, runtime, and memory records are stored in the
five neighboring JSON files. All final models were finite and passed their
pointwise or nonlocal homogeneous-state feasibility audit.

## Interpretation

Relative to the validation scores used for structural selection, test E_data
increased by 5.79% (Greenshields), 50.51% (IDM), 19.88% (Weidmann), 40.47%
(Triangular), and 22.81% (Del Castillo). Relative to the post-refit
train+validation score, the gaps are 20.63%, 10.48%, 10.05%, 5.46%, and 15.52%,
respectively. IDM and Triangular therefore show substantial selection-to-test
optimism even though their post-refit external gaps are moderate.

The test block is now consumed. Any further structural iteration must use a
fresh external dataset or a new predeclared check rather than repeatedly
selecting against these test results.
