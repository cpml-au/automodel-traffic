# Attempt 3 plan

- Candidate: `g(rho) = exp(a*rho + b*rho^2 + c*rho^3)`.
- Fit `a, b, c` independently for all five calibrated baselines on the complete
  I80 prediction training split.
- Use two Powell starts, at most 30 function evaluations per start, bounds
  `[-4, 4]` for each parameter, and seed `1130 + baseline_index`.
- Apply the existing positivity, finiteness, nonnegative-velocity, and
  non-increasing-velocity feasibility checks.
- Evaluate the fitted parameters once on the complete validation split. Never
  access or evaluate the test split.
- Selection fitness: validation data error plus `0.01 * 16` tree nodes.
