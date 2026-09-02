# Attempt 2 plan

Keep each selected meta-2 incumbent and its coefficients fixed, then fit one
coefficient multiplying a convolution contrast:

`g = g_inc * exp(a * (conv_3(rho, ones) - 3*conv_1(rho, ones)))`.

The contrast largely cancels density level on a homogeneous uniform mesh and is
intended to encode short-range queue-front structure. Calibration uses only the
full training interval, and selection uses only full validation. Feasibility is
checked on homogeneous states and simulations use the repaired row-wise
absolute flux-Jacobian-sum wave-speed bound. The test interval remains untouched.
