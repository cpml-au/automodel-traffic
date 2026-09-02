# Attempt 1 plan

Starting from each baseline's selected meta-2 multiplier, with every incumbent
coefficient fixed, add one positive exponential factor based on a three-cell DEC
convolution:

`g = g_inc * exp(a * conv_3(rho, ones))`.

This probes whether a short downwind density aggregate supplies information not
available to the incumbent pointwise multiplier. Only `a` is fitted, on the full
I80 training interval. The full validation interval is used once for structural
selection; the held-out test interval is not accessed. The convolution-aware
homogeneous feasibility check and the row-wise absolute flux-Jacobian-sum
Rusanov speed bound are used.
