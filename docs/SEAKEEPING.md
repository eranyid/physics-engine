# Seakeeping & Morison (M11 hydro)

Clean-room strip theory and slender-member wave loads in `pe-hydro`.
SI units; portable `libm` only; no FMA. Sources cited in [`THIRD_PARTY.md`](../THIRD_PARTY.md).

## Morison equation (`pe_hydro::morison`)

Force per unit length normal to a slender circular member:

\[
\mathrm{d}\mathbf{F}
  = \rho C_m A \mathbf{a}_n
  + \tfrac12 \rho C_d D\, |\mathbf{u}_{rn}|\,\mathbf{u}_{rn}
\]

with \(A = \pi D^2/4\), \(\mathbf{a}_n\) fluid acceleration normal to the member,
\(\mathbf{u}_{rn} = (\mathbf{u}_\mathrm{fluid}-\mathbf{u}_\mathrm{struct})_n\).
`C_m = 1 + C_a` includes Froude–Krylov. Structure acceleration is **not** folded
into the inertia term (caller may add \(-\rho C_a A \mathbf{a}_\mathrm{struct}\)).

Integration: equal segments along the member; kinematics from existing
[`Wave`](../crates/pe-hydro/src/waves/mod.rs) fields (Airy has analytic
acceleration; others use a central time difference).

### Nominal coefficients (unverified)

| Symbol | Value | Note |
| --- | --- | --- |
| `NOMINAL_CM_CIRCULAR` | 2.0 | API RP 2A / DNV-RP-C205 typical post-critical order |
| `NOMINAL_CD_CIRCULAR` | 1.0 | same — **not** fitted in this tree |

### Validation

- Closed-form Airy **inertia** force and mudline moment on a fixed vertical pile
  (finite depth and deep water) vs segment integration: **< 0.5%** relative at
  ≥ 64 segments; error decreases with segment count.
- Crest-phase **drag-only** closed form (`cos χ ≥ 0`, no reversal): **< 0.2%**
  at 128 segments.
- Full Morison: 64 segments within **1%** of a 256-segment reference.

### Deferred

Member interference / wake; depth-varying \(C_d,C_m\); MacCamy–Fuchs diffraction
for large \(D/\lambda\).

---

## Strip theory heave / pitch (`pe_hydro::seakeeping`)

Salvesen–Tuck–Faltinsen (1970) strip integrals for **head seas**, heave \(\xi_3\)
and pitch \(\xi_5\), zero forward speed.

### Sectional hydrodynamics

| Quantity | Formula | Status |
| --- | --- | --- |
| \(a_{33}^\infty\) | \(\rho\pi (B/2)^2\,\sigma\) (circular Lewis substitute) | cited (Faltinsen / Newman circle limit); \(\sigma\) clamp **nominal** for non-circles |
| \(a_{33}(\omega)\) | \(a_{33}^\infty\cdot\frac{(\omega/\omega_0)^2}{1+(\omega/\omega_0)^2}\), \(\omega_0=\sqrt{g/(B/2)}\) | smooth surrogate — **not** a Lewis frequency table |
| \(b_{33}\) | \(\rho\pi R^2\omega\cdot\beta\cdot e^{-2kT_\mathrm{eq}}\), \(\beta=0.5\) | **nominal / unverified** vs Ursell–Vugts |

### RAOs

Frequency-domain \(2\times 2\) system
\((-\omega^2(M+A)+i\omega B+C)\boldsymbol{\xi}=\mathbf{F}\) with Froude–Krylov
plus STF Haskind diffraction from sectional \(b_{33}\).

**Long-wave heave RAO → 1** (ship follows the wave): tested within **10%** of 1
at \(T=60\,\mathrm{s}\) for an 80 m box barge (engineering strip tolerance).

### Roll natural period

\[
T_4 = 2\pi\sqrt{\frac{I_{44}+A_{44}}{\rho g \nabla\,\mathrm{GM}}}
\]

\(A_{44}\) is **not** predicted by this heave/pitch module — pass 0 or a
documented value. Formula identity tested exactly.

### Box heave natural frequency

\[
\omega_3 = \sqrt{\frac{\rho g A_\mathrm{wp}}{m+A_{33}}}
\]

with \(A_{33}\) from strip \(a_{33}^\infty\); identity tested to 1e-12 relative.

### Time domain

Constant-coefficient integrator at a single frozen \(\omega_\mathrm{ref}\)
(semi-implicit Euler). **No Cummins retardation** — documented limit.

Free-decay envelope rate \(\beta = B_{33}/(2(m+A_{33}))\): peak-fit within
**15%** of analytic (discrete sampling).

### Deferred

Cummins memory; forward speed \(U\neq 0\); roll/sway/yaw; tabulated Vugts /
Frank close-fit sections; 3-D panel diffraction; viscous roll damping.
