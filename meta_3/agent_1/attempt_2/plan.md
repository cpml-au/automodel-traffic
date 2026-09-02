# Attempt 2 plan

Test a short downwind DEC convolution multiplier,
`g = exp(a * conv_3(rho, ones))`, on every fixed fundamental diagram.
`conv_3` is the exact `dctkit.dec.cochain.convolution` operator with
`kernel_window=3`; `ones` is a primal zero-cochain on the I80 complex.

Compared with attempt 1, this exposes the model to a three-node density
neighborhood. The exponential guarantees positivity. Fit only `a` on full
training, use the nonlocal feasibility gate, and score full validation only.
The held-out test interval must not be evaluated.

GP node count: 6 (`exp`, multiplication, `a`, `conv_3`, `rho`, `ones`).
