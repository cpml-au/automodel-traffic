# Attempt 3 training

Expression: `exp(a*conv_1(rho,ones)+b*conv_3(rho,ones))` (12 GP tree nodes).

All values use the complete I80 prediction training split (times 0-63).
Each fit used two deterministic Powell starts with at most 45 function
evaluations per start. Bounds: `[[-500.0, 500.0], [-250.0, 250.0]]`.
Nonlocal feasibility used homogeneous states; full simulations supplied
the finite gate. The flux speed bound was `row_abs_jacobian_sum`.

| Baseline | Parameters | E_rho | E_v | E_data | Finite/feasible | Evaluations | Restarts | Optimizer status | Fit runtime (s) | Split runtime (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|:---:|---:|---:|---|---:|---:|---:|
| Greenshields | a=-4.710309853, b=-0.1568147475 | 7.489501 | 6.810893 | 7.150197 | yes | 90 | 2 | False: Maximum number of function evaluations has been exceeded. | 20.173 | 0.249 | 3183.5 |
| IDM | a=-23.96934913, b=-0.7346754944 | 7.189480 | 7.478495 | 7.333987 | yes | 90 | 2 | False: Maximum number of function evaluations has been exceeded. | 272.279 | 0.581 | 3286.5 |
| Weidmann | a=58.55936855, b=-19.51949472 | 8.169318 | 5.813401 | 6.991360 | yes | 90 | 2 | False: Maximum number of function evaluations has been exceeded. | 27.707 | 0.272 | 3467.4 |
| Triangular | a=436.6349402, b=-149.1320997 | 7.158822 | 9.612005 | 8.385413 | yes | 90 | 2 | False: Maximum number of function evaluations has been exceeded. | 27.247 | 0.289 | 3742.0 |
| Del Castillo | a=-6.346689871, b=-0.4440752208 | 7.222665 | 6.765142 | 6.993904 | yes | 90 | 2 | False: Maximum number of function evaluations has been exceeded. | 37.239 | 0.321 | 4126.1 |

Attempt wall time: 384.651 seconds.
