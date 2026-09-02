# Attempt 3 training

Expression: `exp(c0 + a*rho + b*rho^2)` (12 tree nodes).

All values use the full I80 prediction training split (times 0-63).
Fits used two deterministic Powell starts and at most 45 function
evaluations per start. Parameter bounds: `[[-1.0, 1.0], [-5.0, 5.0], [-5.0, 5.0]]`.

| Baseline | Constants | E_rho | E_v | E_data | Feasible | Evaluations | Optimizer status | Fit runtime (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|:---:|---:|---|---:|---:|
| Greenshields | c0=0.02312297116, a=-0.08831739726, b=-0.150081221 | 7.331324 | 6.764349 | 7.047837 | yes | 90 | False: Maximum number of function evaluations has been exceeded. | 15.889 | 3945.0 |
| IDM | c0=-0.1651545221, a=-0.05973336478, b=-0.0247305532 | 7.156401 | 6.096211 | 6.626306 | yes | 90 | False: Maximum number of function evaluations has been exceeded. | 46.556 | 4435.4 |
| Weidmann | c0=-0.01955409838, a=0.2296388975, b=0.2669364909 | 7.753528 | 6.140423 | 6.946976 | yes | 90 | False: Maximum number of function evaluations has been exceeded. | 15.574 | 4709.5 |
| Triangular | c0=-0.09737331875, a=0.08180344902, b=0.07213459761 | 7.880916 | 9.531303 | 8.706110 | yes | 90 | False: Maximum number of function evaluations has been exceeded. | 15.713 | 4960.1 |
| Del Castillo | c0=0.2372542188, a=-1.534809737, b=2.479218285 | 7.279193 | 6.478570 | 6.878881 | yes | 90 | False: Maximum number of function evaluations has been exceeded. | 26.627 | 5314.6 |

Attempt wall time: 120.361 seconds.
