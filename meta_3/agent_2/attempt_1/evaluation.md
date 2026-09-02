# Attempt 1 evaluation

Selection uses full I80 validation times 64--107 and `E_fitness = E_data + 0.01*tree_nodes`; lower is better.

| FD | Typed GP expression | Nodes | E_rho | E_v | E_data | E_fitness | Meta-2 fitness | Change | Runtime (s) | Peak RSS (MB) | Finite | Feasible | Seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| Greenshields | `ExpP0(MFP0(St_oneD1(St_oneP0(rho)), a))` | 6 | 10.014572 | 7.653971 | 8.834271 | 8.894271 | 7.627391 | +1.266880 | 0.475 | 986.29 | yes | yes | 8110 |
| IDM | `ExpP0(MFP0(St_oneD1(St_oneP0(rho)), a))` | 6 | 6.905678 | 8.497431 | 7.701554 | 7.761554 | 5.955012 | +1.806542 | 0.691 | 3190.53 | yes | yes | 8111 |
| Weidmann | `ExpP0(MFP0(St_oneD1(St_oneP0(rho)), a))` | 6 | 5.796083 | 7.166179 | 6.481131 | 6.541131 | 6.322293 | +0.218838 | 0.194 | 3303.39 | yes | yes | 8112 |
| Triangular | `ExpP0(MFP0(St_oneD1(St_oneP0(rho)), a))` | 6 | 6.501461 | 8.969658 | 7.735559 | 7.795559 | 6.457489 | +1.338070 | 0.196 | 3484.90 | yes | yes | 8113 |
| Del Castillo | `ExpP0(MFP0(St_oneD1(St_oneP0(rho)), a))` | 6 | 6.038464 | 7.398791 | 6.718627 | 6.778627 | 6.603956 | +0.174671 | 0.249 | 3708.40 | yes | yes | 8114 |

Registered live Hodge-star names: `St_oneP0`, `St_oneP1`, `St_oneD0`, `St_oneD1`.

`test_evaluated = false`; no test prediction or score was computed.
