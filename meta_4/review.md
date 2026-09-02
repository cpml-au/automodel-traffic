# Meta-iteration 4 root review

Meta 4 completed 45 fits across alternative convolution contrasts, repaired
Hodge-star/convolution gates, and pointwise square-root controls. Forty-two
final candidates were feasible; three Weidmann candidates were rejected for
non-monotonicity or non-finite simulations. No test data was evaluated.

The only new winner is the IDM quadratic-Hodge-gated convolution contrast.
An independent root-process evaluation reproduced it as finite and feasible:
validation `E_data = 4.728762` and fitness `5.048762` in JAX x64, versus the
agent's float32 `4.728745` and `5.048745`.

| Baseline | Selected multiplier after meta 4 | Train E_data | Validation E_data | Validation fitness | Previous fitness |
|---|---|---:|---:|---:|---:|
| Greenshields | retained meta-3 convolution contrast | 6.334473 | 6.703732 | 6.953732 | 6.953732 |
| IDM | `g_meta3*exp(-119537.388349*hquad*(conv_3-3*conv_1))` | 7.679225 | 4.728745 | 5.048745 | 5.417018 |
| Weidmann | retained jam-anchored incumbent | 7.205753 | 6.222293 | 6.322293 | 6.322293 |
| Triangular | identity | 9.223384 | 6.447489 | 6.457489 | 6.457489 |
| Del Castillo | retained meta-3 convolution model | 6.364161 | 5.520221 | 5.730221 | 5.730221 |

Here `hquad = St_oneD1(SquareD1(St_oneP0(rho)))`. The IDM expression has 32
nodes, so its reported fitness includes a `0.32` complexity penalty.

## Findings

- The repaired Hodge-star primitives are not merely callable: a quadratic
  Hodge feature gated by the level-cancelling convolution contrast improves IDM
  validation fitness by 6.80% relative to meta 3 and 15.22% relative to the
  identity baseline.
- Alternative contrast normalizations produced only negligible raw changes
  that were outweighed by complexity. The pointwise square-root controls did
  not beat any incumbent.
- Several Hodge/convolution candidates for Weidmann were unstable or substantially
  worse. The physical and finite-simulation checks correctly rejected the
  invalid cases.
- Across all four meta-iterations, 180 fitted baseline/expression pairs were
  evaluated using train and validation only. Times 108--179 remain untouched.
