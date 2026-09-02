# Meta 2 agent 2 — centered-exponential summary

All 15 requested fits completed on the current checkout and CPU. Every final
model was finite and physically feasible under the shared 79-point domain check.
No test prediction or test score was computed.

The center for each baseline was fixed to
`r* = 0.5*physical_density_limit(baseline)`:

| Baseline | r* |
|---|---:|
| Greenshields | 0.279975615 |
| IDM | 0.347375764723 |
| Weidmann | 0.403060485 |
| Triangular | 0.335649971536 |
| Del Castillo | 0.307660845 |

## Validation results

Fitness is `validation E_data + 0.01*tree_nodes`.

| Baseline | Attempt 1 fitness | Attempt 2 fitness | Attempt 3 fitness | Branch best | Meta-1 incumbent | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Greenshields | **8.343199** | 12.708665 | 8.491098 | 8.343199 | 8.205311 | no improvement |
| IDM | **12.562173** | 12.667339 | 12.857800 | 12.562173 | 5.955012 identity | no improvement |
| Weidmann | 11.542300 | 11.351727 | **11.270119** | 11.270119 | 6.322293 | no improvement |
| Triangular | 8.615415 | **6.693149** | 7.222813 | 6.693149 | 6.457489 identity | no improvement |
| Del Castillo | 12.662561 | 12.813254 | **7.879776** | 7.879776 | 6.638030 identity | no improvement |

The centered-linear Greenshields candidate was the branch's strongest absolute
result:

`exp(-0.3764529309*(rho-0.279975615))`

with validation `E_data=8.283199` and `E_fitness=8.343199`. It remains
`0.137888` fitness above the meta-1 Greenshields incumbent. None of this
lineage's candidates replaces a meta-1 incumbent.

The optimizer reported budget exhaustion for 10 of 15 final fits, but all saved
fits completed their final full-train/full-validation evaluations and passed
feasibility. The behavior is consistent with strong training-to-validation
drift rather than numerical failure; centering at half the physical density
limit did not improve transfer in this parameterization.
