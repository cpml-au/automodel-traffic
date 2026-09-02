# Attempt 1 rejected optimizer preflight

The initial squared-contrast range `a in [-1e6,1e6]` was rejected before the
accepted attempt. Bounded Powell returned approximately `a = +999999.9785` for
all five FDs; every resulting full train/validation simulation was non-finite
(`NaN` or `inf`). No candidate from this preflight was compared for selection.

At homogeneous `rho = 1`, the largest observed `C^2` was `0.00144208`, so the
rejected endpoint creates an exponent near `1442`, beyond float32 exponential
range. The accepted fit uses `[-2e4,2e4]` and explicitly compares Powell's
endpoint with the finite zero start, which exactly recovers the fixed global
incumbent. The held-out test was not evaluated.
