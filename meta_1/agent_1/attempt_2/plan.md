# Attempt 2 plan

- Candidate: `g(rho) = exp(a*rho + b*rho^2)`.
- Fit `a, b` independently for all five calibrated baselines on the complete
  I80 prediction training split.
- Use two Powell starts, at most 30 function evaluations per start, bounds
  `[-4, 4]` for each parameter, and seed `1120 + baseline_index`.
- Apply the existing positivity, finiteness, nonnegative-velocity, and
  non-increasing-velocity feasibility checks.
- Evaluate the fitted parameters once on the complete validation split. Never
  access or evaluate the test split.
- Selection fitness: validation data error plus `0.01 * 10` tree nodes.
