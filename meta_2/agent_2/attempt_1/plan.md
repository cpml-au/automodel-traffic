# Attempt 1 plan

Fit the prescribed minimal centered positive multiplier

`g(rho) = exp(a*(rho-r*))`, where `r* = 0.5*physical_density_limit(baseline)`,

separately for all five fixed FDs. Fit `a` on the full I80 training split via
the shared `fit_candidate` utility, using two deterministic Powell restarts,
45 evaluations per restart, bounds `[-5,5]`, and seed
`5100 + 1*10 + baseline_index`. Score only the full validation split with
`E_data + 0.01*6`. Log exact per-FD centers and expressions, components/data,
constants, diagnostics, runtime, RSS, feasibility, and `test_evaluated=false`.

The exponential guarantees a positive multiplier. The shared feasibility check
must additionally confirm finite values and non-negative, non-increasing
corrected velocity on each FD's physical domain. Do not access the test split.
