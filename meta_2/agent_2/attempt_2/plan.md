# Attempt 2 plan

Extend the centered exponent with curvature:

`g(rho) = exp(a*(rho-r*) + b*(rho-r*)^2)`, with
`r* = 0.5*physical_density_limit(baseline)`.

Fit `a,b` separately for all five fixed FDs on the complete training split via
the shared utility, with two Powell restarts, 45 evaluations per restart,
`[-5,5]` bounds, and seed `5100 + 2*10 + baseline_index`. Score only complete
validation using `E_data + 0.01*14`. Preserve the same fixed solver, data, and
baseline components and tune only correction constants. Record exact per-FD
expressions and centers, metrics, optimizer diagnostics, runtime, RSS,
feasibility, and `test_evaluated=false`; never access test data.
