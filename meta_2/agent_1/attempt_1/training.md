# Attempt 1 training

Expression: `exp(c0)` (2 tree nodes).

All values use the full I80 prediction training split (times 0-63).
Fits used two deterministic Powell starts and at most 45 function
evaluations per start. Parameter bounds: `[[-1.0, 1.0]]`.

| Baseline | Constants | E_rho | E_v | E_data | Feasible | Evaluations | Optimizer status | Fit runtime (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|:---:|---:|---|---:|---:|
| Greenshields | c0=0.02263538121 | 7.776723 | 6.578376 | 7.177549 | yes | 67 | False: Maximum number of function evaluations has been exceeded. | 9.782 | 819.8 |
| IDM | c0=-0.1883932724 | 7.256054 | 6.095199 | 6.675627 | yes | 90 | False: Maximum number of function evaluations has been exceeded. | 51.667 | 1436.5 |
| Weidmann | c0=-0.01971739213 | 8.341578 | 6.088132 | 7.214855 | yes | 64 | True: Optimization terminated successfully. | 9.879 | 1630.5 |
| Triangular | c0=-0.09737332019 | 7.776231 | 9.697797 | 8.737014 | yes | 73 | True: Optimization terminated successfully. | 11.656 | 1831.0 |
| Del Castillo | c0=0.005495202656 | 7.384300 | 6.710897 | 7.047598 | yes | 56 | True: Optimization terminated successfully. | 12.816 | 2051.1 |

Attempt wall time: 95.814 seconds.
