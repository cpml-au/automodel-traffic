# Attempt 1 training

Expression: `exp(a*conv_1(rho,ones))` (6 GP tree nodes).

All values use the complete I80 prediction training split (times 0-63).
Each fit used two deterministic Powell starts with at most 45 function
evaluations per start. Bounds: `[[-500.0, 500.0]]`.
Nonlocal feasibility used homogeneous states; full simulations supplied
the finite gate. The flux speed bound was `row_abs_jacobian_sum`.

| Baseline | Parameters | E_rho | E_v | E_data | Finite/feasible | Evaluations | Restarts | Optimizer status | Fit runtime (s) | Split runtime (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|:---:|---:|---:|---|---:|---:|---:|
| Greenshields | a=-4.693158198 | 7.507726 | 6.794644 | 7.151185 | yes | 90 | 2 | False: Maximum number of function evaluations has been exceeded. | 20.845 | 0.150 | 1000.0 |
| IDM | a=-23.93975411 | 7.147620 | 7.529872 | 7.338746 | yes | 89 | 2 | True: Optimization terminated successfully. | 230.302 | 0.462 | 3157.8 |
| Weidmann | a=0.0009458826666 | 8.326837 | 6.138566 | 7.232701 | yes | 76 | 2 | True: Optimization terminated successfully. | 13.724 | 0.160 | 1235.8 |
| Triangular | a=-15.19070443 | 7.765910 | 10.349447 | 9.057678 | yes | 90 | 2 | False: Maximum number of function evaluations has been exceeded. | 20.518 | 0.175 | 1558.4 |
| Del Castillo | a=-6.349669457 | 7.248189 | 6.744154 | 6.996172 | yes | 82 | 2 | True: Optimization terminated successfully. | 22.176 | 0.248 | 1873.8 |

Attempt wall time: 307.603 seconds.
