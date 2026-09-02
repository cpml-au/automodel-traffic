# Attempt 1 plan

Augment every fixed meta-3 incumbent with `exp(a*hlin*(conv_3-3*conv_1))`.
Exact typed factor: `ExpP0(CMulP0(MFP0(St_oneD1(St_oneP0(rho)),a),SubCP0(conv_3P0(rho,ones),MFP0(conv_1P0(rho,ones),three))))` (16 nodes).
A `CMulP0` attachment makes 17 added nodes; incumbent
nodes are included separately in each total. The implementation uses
`C.star`, `C.cochain_mul`, and `C.convolution`. Fit only the new constants
on full train, select only on full validation, and never evaluate test.
