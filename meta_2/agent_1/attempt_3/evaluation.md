# Attempt 3 evaluation

All values use the full I80 prediction validation split (times 64-107).
The held-out test split was not evaluated. Fitness is validation
`E_data + 0.01 * tree_nodes`; lower is better.

| Baseline | Expression | Constants | E_rho | E_v | E_data | Fitness | Finite/feasible | Runtime (s) | Peak RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `exp(c0 + a*rho + b*rho^2)` | c0=0.02312297116, a=-0.08831739726, b=-0.150081221 | 7.525036 | 7.489747 | 7.507391 | 7.627391 | yes | 0.165 | 3952.9 |
| IDM | `exp(c0 + a*rho + b*rho^2)` | c0=-0.1651545221, a=-0.05973336478, b=-0.0247305532 | 6.090361 | 7.221606 | 6.655983 | 6.775983 | yes | 0.553 | 4444.8 |
| Weidmann | `exp(c0 + a*rho + b*rho^2)` | c0=-0.01955409838, a=0.2296388975, b=0.2669364909 | 12.202457 | 9.980204 | 11.091331 | 11.211331 | yes | 0.170 | 4711.3 |
| Triangular | `exp(c0 + a*rho + b*rho^2)` | c0=-0.09737331875, a=0.08180344902, b=0.07213459761 | 8.000912 | 7.274188 | 7.637550 | 7.757550 | yes | 0.185 | 4968.7 |
| Del Castillo | `exp(c0 + a*rho + b*rho^2)` | c0=0.2372542188, a=-1.534809737, b=2.479218285 | 6.306695 | 8.435495 | 7.371096 | 7.491096 | yes | 0.285 | 5323.8 |

Complexity penalty: 0.12.
`test_evaluated = false`.
