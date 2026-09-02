# Attempt 2 plan

- Structure: `g(rho) = exp((a*rho + b*rho^2)/(1+rho))`.
- Hypothesis: quadratic curvature lets the correction distinguish free-flow and
  congested-density behavior while retaining positivity, saturation relative to
  an ordinary exponential polynomial, and the anchor `g(0)=1`.
- Fit each of the five fixed I80 prediction baselines independently on all train
  times with two deterministic Powell starts, 30 evaluations per restart, and
  both coefficients in `[-5, 5]`.
- Evaluate the fitted model once on the full validation interval. Never inspect
  or evaluate the held-out test interval.
- Complexity convention: 13 expression-tree nodes, counting operators and
  leaves, with `rho^2` represented by the available `Square` primitive.
