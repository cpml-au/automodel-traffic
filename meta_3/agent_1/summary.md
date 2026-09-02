# Meta 3 agent 1 summary — direct DEC convolutions

All 15 requested train fits and validation evaluations completed. Every final
simulation was finite and passed `is_nonlocal_feasible`. The held-out I80 test
split was not evaluated (`test_evaluated=false`). All candidates used exact
`dctkit.dec.cochain.convolution` with a P0 ones kernel and the
`row_abs_jacobian_sum` nonlocal flux-speed bound.

| Baseline | Best direct-convolution candidate | Parameters | Validation E_data | Fitness | Meta-2 incumbent fitness | Fitness delta |
|---|---|---|---:|---:|---:|---:|
| Greenshields | `exp(a*conv_3(rho,ones))` | `a=-1.662726641` | 8.747316 | 8.807316 | 7.627391 | +1.179925 |
| IDM | `exp(a*conv_3(rho,ones))` | `a=-8.669891236` | 7.675070 | 7.735070 | 5.955012 | +1.780058 |
| Weidmann | `exp(a*conv_1(rho,ones)+b*conv_3(rho,ones))` | `a=58.55936855`, `b=-19.51949472` | 6.251344 | 6.371344 | 6.322293 | +0.049051 |
| Triangular | `exp(a*conv_1(rho,ones)+b*conv_3(rho,ones))` | `a=436.6349402`, `b=-149.1320997` | 6.567798 | 6.687798 | 6.457489 | +0.230309 |
| Del Castillo | `exp(a*conv_1(rho,ones))` | `a=-6.349669457` | 6.719110 | 6.779110 | 6.603956 | +0.175154 |

Positive deltas are regressions. Thus no direct-convolution-only candidate
replaces a meta-2 incumbent after the configured complexity penalty. The
combined convolution is directionally useful for Weidmann: its unpenalized
validation `E_data=6.251344` is better than the identity data error, but its
12-node fitness remains 0.049051 above the jam-anchored incumbent.

The Powell evaluation cap was reached in nine fits, while six reported
optimizer success. This status is recorded per baseline in each `results.json`;
all capped fits still produced finite, feasible final candidates.
