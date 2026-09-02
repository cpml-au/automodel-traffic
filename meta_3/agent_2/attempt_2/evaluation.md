# Attempt 2 evaluation

Selection uses full I80 validation times 64--107 and `E_fitness = E_data + 0.01*tree_nodes`; lower is better.

| FD | Typed GP expression | Nodes | E_rho | E_v | E_data | E_fitness | Meta-2 fitness | Change | Runtime (s) | Peak RSS (MB) | Finite | Feasible | Seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| Greenshields | `ExpP0(MFP0(St_oneD1(SquareD1(St_oneP0(rho))), a))` | 7 | 9.033915 | 7.586672 | 8.310293 | 8.380293 | 7.627391 | +0.752902 | 0.221 | 3927.78 | yes | yes | 8120 |
| IDM | `ExpP0(MFP0(St_oneD1(SquareD1(St_oneP0(rho))), a))` | 7 | 6.831917 | 7.641999 | 7.236958 | 7.306958 | 5.955012 | +1.351946 | 0.739 | 3955.52 | yes | yes | 8121 |
| Weidmann | `ExpP0(MFP0(St_oneD1(SquareD1(St_oneP0(rho))), a))` | 7 | 5.796900 | 7.156356 | 6.476628 | 6.546628 | 6.322293 | +0.224335 | 0.235 | 3955.52 | yes | yes | 8122 |
| Triangular | `ExpP0(MFP0(St_oneD1(SquareD1(St_oneP0(rho))), a))` | 7 | 6.418392 | 6.487388 | 6.452889 | 6.522889 | 6.457489 | +0.065400 | 0.222 | 3955.52 | yes | yes | 8123 |
| Del Castillo | `ExpP0(MFP0(St_oneD1(SquareD1(St_oneP0(rho))), a))` | 7 | 6.096465 | 8.194215 | 7.145340 | 7.215340 | 6.603956 | +0.611384 | 0.274 | 3980.61 | yes | yes | 8124 |

Registered live Hodge-star names: `St_oneP0`, `St_oneP1`, `St_oneD0`, `St_oneD1`.

`test_evaluated = false`; no test prediction or score was computed.
