# Attempt 2 evaluation

The fitted constants were frozen and evaluated once on the complete validation
split (times 64--107), never on test. Fitness adds the 0.13 complexity penalty
(`0.01 * 13` nodes) to validation data error.

| Baseline | Validation density error | Validation velocity error | Validation data error | Validation fitness | Peak RSS (MB) | Feasible |
|---|---:|---:|---:|---:|---:|:---:|
| Greenshields | 10.473258 | 7.642451 | 9.057855 | 9.187855 | 890.773 | yes |
| IDM | 6.762498 | 8.130500 | 7.446499 | 7.576499 | 1313.422 | yes |
| Weidmann | 6.800902 | 6.386293 | 6.593598 | 6.723598 | 1505.750 | yes |
| Triangular | 6.515769 | 9.196405 | 7.856087 | 7.986087 | 1736.742 | yes |
| Del Castillo | 6.051924 | 7.966357 | 7.009141 | 7.139141 | 2011.008 | yes |

The quadratic term improved IDM over attempt 1, but its validation error still
exceeded the uncorrected IDM benchmark. It worsened Weidmann and Del Castillo,
and collapsed to zero for Greenshields and Triangular, where it only adds a
complexity penalty. Peak RSS is the cumulative process high-water mark.
