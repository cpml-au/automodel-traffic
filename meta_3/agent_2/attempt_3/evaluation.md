# Attempt 3 evaluation

Selection uses full I80 validation times 64--107 and `E_fitness = E_data + 0.01*tree_nodes`; lower is better.

| FD | Typed GP expression | Nodes | E_rho | E_v | E_data | E_fitness | Meta-2 fitness | Change | Runtime (s) | Peak RSS (MB) | Finite | Feasible | Seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| Greenshields | `ExpP0(AddCP0(MFP0(St_oneD1(St_oneP0(rho)), a), MFP0(St_oneD1(SquareD1(St_oneP0(rho))), b)))` | 13 | 9.451165 | 7.683293 | 8.567229 | 8.697229 | 7.627391 | +1.069838 | 0.253 | 4173.36 | yes | yes | 8130 |
| IDM | `ExpP0(AddCP0(MFP0(St_oneD1(St_oneP0(rho)), a), MFP0(St_oneD1(SquareD1(St_oneP0(rho))), b)))` | 13 | 6.788590 | 7.888933 | 7.338762 | 7.468762 | 5.955012 | +1.513750 | 0.806 | 4197.74 | yes | yes | 8131 |
| Weidmann | `ExpP0(AddCP0(MFP0(St_oneD1(St_oneP0(rho)), a), MFP0(St_oneD1(SquareD1(St_oneP0(rho))), b)))` | 13 | 5.796083 | 7.166179 | 6.481131 | 6.611131 | 6.322293 | +0.288838 | 0.256 | 4197.74 | yes | yes | 8132 |
| Triangular | `ExpP0(AddCP0(MFP0(St_oneD1(St_oneP0(rho)), a), MFP0(St_oneD1(SquareD1(St_oneP0(rho))), b)))` | 13 | 6.500090 | 8.954266 | 7.727178 | 7.857178 | 6.457489 | +1.399689 | 0.281 | 4197.74 | yes | yes | 8133 |
| Del Castillo | `ExpP0(AddCP0(MFP0(St_oneD1(St_oneP0(rho)), a), MFP0(St_oneD1(SquareD1(St_oneP0(rho))), b)))` | 13 | 6.092255 | 8.369377 | 7.230816 | 7.360816 | 6.603956 | +0.756860 | 0.325 | 4197.74 | yes | yes | 8134 |

Registered live Hodge-star names: `St_oneP0`, `St_oneP1`, `St_oneD0`, `St_oneD1`.

`test_evaluated = false`; no test prediction or score was computed.
