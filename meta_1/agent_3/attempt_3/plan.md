# Attempt 3 plan

- Structure: `g(rho) = exp((a*rho + b*rho^2 + c*rho^3)/(1+rho))`.
- Hypothesis: cubic curvature can add a second bend to the multiplier when the
  quadratic family cannot fit both transition and congested regimes, while the
  positive exponential and denominator retain controlled pointwise behavior.
- Fit each of the five fixed I80 prediction baselines independently on all train
  times with two deterministic Powell starts, 30 evaluations per restart, and
  all coefficients in `[-5, 5]`.
- Evaluate the fitted model once on the full validation interval. Never inspect
  or evaluate the held-out test interval.
- Complexity convention: 20 expression-tree nodes, counting operators and
  leaves and expressing `rho^3` as `CMul(Square(rho),rho)`.
