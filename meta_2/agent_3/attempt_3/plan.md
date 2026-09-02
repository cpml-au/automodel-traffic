# Attempt 3 plan

- Candidate: `g(rho) = (1 + a*rho + b*rho^2)/(1 + c*rho)` (17 tree
  nodes).
- The quadratic numerator adds one curvature degree of freedom to attempt 2
  while preserving `g(0)=1` and pointwise locality.
- Fit `a,b,c` independently for all five fixed calibrated baselines using the
  full I80 prediction training split (times 0--63).
- Use two deterministic Powell starts, at most 45 objective evaluations per
  start, bounds `[-0.9, 4.0]` for each parameter, and seed
  `6130 + baseline_index`.
- Reject non-finite or non-positive multipliers, near-zero/non-positive
  denominators, and corrected velocities that are negative or increase on the
  baseline's physical domain.
- Score exactly once on the full validation split (times 64--107) after fitting.
  Never evaluate the held-out test split.
- Selection fitness: validation data error plus `0.01 * 17`.
