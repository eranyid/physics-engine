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

### Validation (measured)

| Check | Tolerance | Measured |
| --- | --- | --- |
| Airy inertia force/moment vs closed form, ≥64 segs | **0.5%** | under tol; converges with \(N\) |
| Crest-phase drag-only vs closed form, 128 segs | **0.2%** | under tol |
| Full Morison 64 vs 256-seg reference | **1%** | under tol |

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

**Long-wave heave RAO → 1.** Residual at finite \(\lambda/L\) comes from:

1. \(-\omega^2(M+A)\) inertia still nonzero,
2. Haskind diffraction \(\propto i\omega B_{33}\to 0\) only as \(\omega\to 0\),
3. \(e^{ikx}\) phase variation along the hull (\(kL\not\to 0\)),
4. FK depth attenuation \(<1\) at finite \(kT\).

As \(T\) (and \(\lambda/L\)) grows, all four vanish. Measured convergence for an
80×12×3 m box barge (deep water):

| \(T\) [s] | \(\lambda/L\) | \(\|\\mathrm{RAO}-1\|\) |
| --- | --- | --- |
| 20 | 7.8 | 0.60% |
| 40 | 31 | 0.20% |
| 80 | 125 | 0.060% |
| 160 | 499 | 0.015% |
| 320 | 1998 | **0.0038%** |

Tolerance at very long waves: **2%**. Measured at \(T=320\,\mathrm{s}\): **0.0038%**.

### Roll natural period

\[
T_4 = 2\pi\sqrt{\frac{I_{44}+A_{44}}{\rho g \nabla\,\mathrm{GM}}}
\]

\(A_{44}\) is **not** predicted by this heave/pitch module — pass 0 or a
documented value. Formula identity: measured bit-exact / 1e-12.

### Box heave natural frequency

\[
\omega_3 = \sqrt{\frac{\rho g A_\mathrm{wp}}{m+A_{33}}}
\]

with \(A_{33}\) from strip \(a_{33}^\infty\); identity measured to 1e-12 relative.

### Time domain / free decay (frozen coefficient)

Constant-coefficient integrator at a single frozen \(\omega_\mathrm{ref}\)
(semi-implicit Euler). For retardation memory see **Cummins** below.

Free-decay uses RK4 + multi-cycle log-decrement on positive peaks
(\(\beta = B_{33}/(2(m+A_{33}))\)):

| Check | Tolerance | Measured |
| --- | --- | --- |
| \(\|\beta_\mathrm{fit}-\beta\|/\beta\) | **3%** | **~0%** (dt=5e-4, 80k steps) |

---

## Cummins retardation (`pe_hydro::cummins`)

Impulse-response memory for heave (Cummins 1962 / Ogilvie 1964):

\[
K(t)=\frac{2}{\pi}\int_0^{\omega_{\max}} B(\omega)\cos(\omega t)\,d\omega
\]

\[
(m+A_\infty)\ddot z+\int_0^t K(t-\tau)\dot z(\tau)\,d\tau+C z=F(t)
\]

`B(ω)` from strip `B₃₃`; `A_∞` from sectional `a₃₃∞`; trapezoidal cosine
transform; convolution with trapezoidal weights on a uniform `Δt` grid.

### Validation (measured)

| Check | Tolerance | Measured |
| --- | --- | --- |
| Analytic `B(ω)=e^{−αω}` → `K(t)=(2/π)α/(α²+t²)` (α=0.5) | **2%** | **≲ 6e-4** |
| Mid-band `K(t)↔B(ω)` relative L2 (box 40×8×2) | **8%** | **~0.95%** |
| Analytic `A(ω)−A_∞` sine transform of `K` (α=0.5) | **0.12%** | **≲ 5.3e-4** |
| Free-decay `β` TD vs Laplace pole of same `K` | **1%** | **~0.49%** |
| Free-decay `ω_d` TD vs Laplace pole of same `K` | **0.4%** | **~0.19%** |
| Peak energy proxy monotone after transient | no growth (>2% slack) | passes |
| Cross-OS `cummins_hash` | locked | `1ee2f57b9c89c3ae` |

**Why not `β = B(ωₙ)/(2(m+A_∞))`?** That single-frequency frozen reference
disagrees with the convolution free decay by **~22.6%** on this box grid: the
time-domain solution carries the full retardation spectrum and the Ogilvie
`A(ω)−A_∞` content of `K(t)`, while the frozen model keeps one `B` sample and
`A_∞` only. The matching independent oracle is the Laplace characteristic pole
`(m+A_∞)λ² + λ K̂(λ) + C = 0` of the **same** kernel (plus the sine-transform
`A−A_∞` check). CI tolerances are ~2× the measured dual-path residual.

### Deferred

Forward speed \(U\neq 0\); roll/sway/yaw; tabulated Vugts /
Frank close-fit sections; 3-D panel diffraction; viscous roll damping;
multi-DOF Cummins coupling matrix.
