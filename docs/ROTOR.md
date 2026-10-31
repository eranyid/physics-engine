# Rotor / propeller BEM (M11 aero)

Blade-element / momentum theory in `pe_aero::bem`. SI; portable `libm`; no FMA.
Citations in [`THIRD_PARTY.md`](../THIRD_PARTY.md).

## Actuator disk (Glauert momentum)

Hover induced velocity and ideal power:

\[
v_h = \sqrt{\frac{T}{2\rho A}},\qquad
P_i = T\, v_h,\qquad
\mathrm{FM} = \frac{T v_h}{P_\mathrm{shaft}}
\]

Climb / advance (axial):

\[
v_i = -\frac{V}{2} + \sqrt{\Bigl(\frac{V}{2}\Bigr)^2 + \frac{T}{2\rho A}}
\]

These identities are tested bit-exact / 1e-12 relative.

## Blade element + Prandtl tip loss

At radius \(r\): inflow \(\phi = \mathrm{atan2}(V_\infty+v_i,\,\Omega r)\),
AoA \(\alpha = \theta - \phi\). Section \(C_L,C_D\) from existing [`WingPolar`].
Prandtl tip loss:

\[
f = \frac{N_b}{2}\frac{R-r}{r\sin\phi},\qquad
F = \frac{2}{\pi}\arccos(e^{-f})
\]

Induced velocity: fixed-point annulus momentum with under-relaxation (default
0.35), max 80 iterations, relative tol \(10^{-5}\). Convergence is guarded;
callers should check `BemResult::converged`.

## Nominal demo geometry (unverified)

`RotorGeometry::uniform_chord` and `nominal_rotor_polar` are **engineering
demos**, not certified blade / airfoil data:

| Parameter | Typical demo | Note |
| --- | --- | --- |
| Section \(C_{L\alpha}\) | \(2\pi\) / rad | thin-airfoil |
| Section \(C_{D0}\) | 0.01–0.015 | nominal |
| Twist schedule | linear root→tip | caller-chosen |

## Validation

| Check | Tolerance |
| --- | --- |
| \(v_h\), \(P_i\), \(\mathrm{FM}=1\) on ideal power | 1e-12 |
| Momentum climb identity \(T=2\rho A(V+v_i)v_i\) | 1e-10 rel |
| Helicopter hover demo: \(\mathrm{FM}\in(0.3,1.05)\), \(P\gtrsim P_i\) | broad physical band |
| Hover \(v_i\) vs \(v_h\) | within 50% (BEM vs disk) |
| Propeller thrust decreases with advance speed | monotonic trend |

## Force generator

`BemRotorForce` applies thrust along a body axis and equal-opposite shaft
torque. Axial freestream is \(-\mathbf{v}_\mathrm{rel}\cdot\hat{\mathbf{e}}\).

## Deferred

Tangential induction \(a'\); yawed / unsteady inflow; dynamic stall;
compressible tip corrections; wake contraction; verified airfoil polars.
