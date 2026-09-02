# Attempt 2 training

Expression: `exp(a*conv_3(rho,ones))` (6 GP tree nodes).

All values use the complete I80 prediction training split (times 0-63).
Each fit used two deterministic Powell starts with at most 45 function
evaluations per start. Bounds: `[[-250.0, 250.0]]`.
Nonlocal feasibility used homogeneous states; full simulations supplied
the finite gate. The flux speed bound was `row_abs_jacobian_sum`.

| Baseline | Parameters | E_rho | E_v | E_data | Finite/feasible | Evaluations | Restarts | Optimizer status | Fit runtime (s) | Split runtime (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|:---:|---:|---:|---|---:|---:|---:|
| Greenshields | a=-1.662726641 | 7.479270 | 6.782038 | 7.130654 | yes | 90 | 2 | False: Maximum number of function evaluations has been exceeded. | 26.933 | 0.242 | 2172.4 |
| IDM | a=-8.669891236 | 7.124703 | 7.355361 | 7.240032 | yes | 90 | 2 | False: Maximum number of function evaluations has been exceeded. | 250.587 | 0.500 | 3228.7 |
| Weidmann | a=-0.0003757845476 | 8.326882 | 6.138577 | 7.232730 | yes | 75 | 2 | True: Optimization terminated successfully. | 18.887 | 0.257 | 2385.9 |
| Triangular | a=-5.089466615 | 7.737935 | 10.344676 | 9.041306 | yes | 87 | 2 | True: Optimization terminated successfully. | 26.758 | 0.261 | 2693.6 |
| Del Castillo | a=-2.490014657 | 7.206776 | 6.735792 | 6.971284 | yes | 85 | 2 | True: Optimization terminated successfully. | 27.298 | 0.302 | 2990.1 |

Attempt wall time: 350.468 seconds.
