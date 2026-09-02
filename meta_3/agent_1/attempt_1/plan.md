# Attempt 1 plan

Test the smallest direct DEC convolution multiplier,
`g = exp(a * conv_1(rho, ones))`, on every fixed fundamental diagram.
`conv_1` is the exact `dctkit.dec.cochain.convolution` operator with
`kernel_window=1`; `ones` is a primal zero-cochain on the I80 complex.

The exponential guarantees a positive multiplier. Fit only `a` on the full
training interval, use homogeneous-state nonlocal feasibility plus full finite
simulation gates, and select only on the full validation interval. The held-out
test interval must not be evaluated.

GP node count: 6 (`exp`, multiplication, `a`, `conv_1`, `rho`, `ones`).
