# Attempt 2 plan

Fit all five fixed meta-3 incumbents multiplied by
`exp(a*SqrtP0(rho) + b*rho)`. In executable JAX, `SqrtP0(rho)` is protected as
`sqrt(max(rho, 0))`; this documents the corresponding GP primitive.

Use full I80 train times 0--63 and full validation times 64--107, two
deterministic Powell restarts, 60 evaluations per restart, coefficient bounds
`[-5, 5]`, and seeds `12120 + baseline_index`. Count the complete incumbent and
the ten-node added factor. Use `is_nonlocal_feasible` for every baseline so the
protocol is valid for convolution incumbents. Do not evaluate the test split.
