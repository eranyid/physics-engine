# Rotor / propeller BEM (M11 aero)

Blade-element / momentum theory in `pe_aero::bem`. SI; portable `libm`; no FMA.
Citations in [`THIRD_PARTY.md`](../THIRD_PARTY.md).

## Actuator disk (Glauert momentum)

Hover induced velocity and ideal power (annulus area \(A=\pi(R^2-r_\mathrm{hub}^2)\)):

\[
v_h = \sqrt{\frac{T}{2\rho A}},\qquad
P_i = T\, v_h,\qquad
\mathrm{FM} = \frac{T v_h}{P_\mathrm{shaft}}
\]

Climb / advance (axial):

\[
v_i = -\frac{V}{2} + \sqrt{\Bigl(\frac{V}{2}\Bigr)^2 + \frac{T}{2\rho A}}
\]

Closed-form identities: bit-exact / 1e-12 relative (measured).

## Blade element + Prandtl tip loss

At radius \(r\): inflow \(\phi = \mathrm{atan2}(V_\infty+v_i,\,\Omega r)\),
AoA \(\alpha = \theta - \phi\). Section \(C_L,C_D\) from existing [`WingPolar`].
Prandtl tip loss:

\[
f = \frac{N_b}{2}\frac{R-r}{r\sin\phi},\qquad
F = \frac{2}{\pi}\arccos(e^{-f})
\]

Induced velocity: fixed-point annulus momentum with under-relaxation (default
0.35), max 80 iterations (200 in ideal mode), relative tol \(10^{-5}\)
(\(10^{-8}\) ideal). Callers should check `BemResult::converged`.

For \(C_D=0\), the local identity \(dP = v_i\,dT\) holds; the solver reports
thrust-weighted \(\bar v_i^{tw}=\sum v_i dT/T\) with
\(|P - T\bar v_i^{tw}|/(T\bar v_i^{tw}) \sim 10^{-9}\) (measured).

## Ideal-twist / Betz oracle

`RotorGeometry::ideal_hover_twist` sets \(\theta(r)=\phi(r)+\alpha\) and Betz
chord \(c(r)=c_\mathrm{tip}(R/r)\). `BemRotor::ideal()` disables tip loss and
forces \(C_D=0\). Momentum reference uses the **annulus** area (hub cut-out).

## Nominal demo coefficients (unverified)

| Parameter | Value | Note |
| --- | --- | --- |
| Section \(C_{L\alpha}\) | \(2\pi\) / rad | thin-airfoil |
| Demo \(C_{D0}\) | 0.004–0.015 | nominal profile drag |
| Ideal α | 4° | test geometry |

## Validation (measured, Linux x86_64 release)

| Check | Tolerance | Measured |
| --- | --- | --- |
| \(v_h\), \(P_i\), \(\mathrm{FM}=1\) closed form | 1e-12 | bit-exact |
| Momentum climb \(T=2\rho A(V+v_i)v_i\) | 1e-10 rel | pass |
| Ideal BEM \(\|\bar v_i-v_h\|/v_h\) (area-wt) | **2%** | **0.0007%** |
| Ideal BEM \(\|\bar v_i^{tw}-v_h\|/v_h\) | **2%** | **0.002%** |
| Ideal \(P\) vs \(T v_h\) | **2%** | **0.002%** |
| Ideal \(P\) vs \(T\bar v_i^{tw}\) (\(C_D=0\)) | 0.5% | **~2e-9** |
| Ideal FM | within 2% of 1 | **0.999978** |
| Ideal iteration residual | \(<10^{-8}\) | **7.2e-9** (21 iters) |
| Tip loss only (\(C_D=0\), 2 blades): FM | [0.85, 0.99] | **0.9787** (drop **0.021** from 0.9995) |
| Profile drag only (\(C_{D0}=0.004\), no tip loss): FM | [0.70, 0.98] | **0.755** |
| Tip loss + \(C_{D0}=0.004\): FM | [0.65, FM_drag) | **0.742** |
| Propeller thrust vs advance | monotonic decrease | pass |

## Force generator

`BemRotorForce` applies thrust along a body axis and equal-opposite shaft
torque. Axial freestream is \(-\mathbf{v}_\mathrm{rel}\cdot\hat{\mathbf{e}}\).

## Deferred

Tangential induction \(a'\); yawed / unsteady inflow; dynamic stall;
compressible tip corrections; wake contraction; verified airfoil polars.
