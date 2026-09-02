# Attempt 3 plan

Add centered skew/asymmetry to the previous candidate:

`g(rho) = exp(a*(rho-r*) + b*(rho-r*)^2 + c*(rho-r*)^3)`, where
`r* = 0.5*physical_density_limit(baseline)`.

Fit `a,b,c` separately for all five fixed FDs on the full training split via
the shared utility, using two Powell restarts, 45 evaluations per restart,
bounds `[-5,5]`, and seed `5100 + 3*10 + baseline_index`. Evaluate only full
validation and score `E_data + 0.01*22`. Log exact per-FD centers/expressions,
constants, metrics, optimizer status, diagnostics, runtime, RSS, feasibility,
and `test_evaluated=false`. The held-out test split must remain untouched.
