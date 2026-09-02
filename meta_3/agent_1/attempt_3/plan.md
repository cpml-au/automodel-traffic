# Attempt 3 plan

Test the combined direct DEC convolution multiplier
`g = exp(a * conv_1(rho, ones) + b * conv_3(rho, ones))` on all five fixed
fundamental diagrams. Both features call the exact
`dctkit.dec.cochain.convolution` operator with primal P0 ones kernels.

The two coefficients can separate the one-node density response from the
three-node neighborhood response. Positivity follows from the exponential.
Fit on full training with the nonlocal feasibility gate and evaluate full
validation only; never evaluate the held-out test interval.

GP node count: 12 (`exp`, addition, two multiplications, `a`, `b`, two
convolutions, two `rho` terminals, and two `ones` terminals).
