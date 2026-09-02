# Attempt 3 plan

Keep all incumbent coefficients fixed and combine the convolution density-level
and convolution-contrast directions:

`g = g_inc * exp(a*conv_3(rho,ones) + b*(conv_3(rho,ones)-3*conv_1(rho,ones)))`.

The two newly fitted coefficients can separate short-range density level from a
local queue-front signal. Both are calibrated exclusively on full training with
two deterministic Powell starts. Full validation is used for selection, with
the complete incumbent-plus-increment tree penalty. Nonlocal homogeneous-state
feasibility and the convolution-aware Rusanov speed bound are used. No test data
are evaluated.
