# Seakeeping, Morison & ships/sea (M11 + M21)

Clean-room strip theory, slender-member wave loads, 6-DOF Cummins memory,
forward-speed STF terms, ITTC-57 resistance, and Nomoto/MMG maneuvering in
`pe-hydro`. SI units; portable `libm` only; no FMA. Sources cited in
[`THIRD_PARTY.md`](../THIRD_PARTY.md).

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

**Diffraction class:** STF strip Haskind approximation (not 3-D panel).

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

## Cummins retardation — heave (`pe_hydro::cummins`)

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

---

## M21 — 6-DOF Cummins (`pe_hydro::cummins6`)

Coupled Cummins equation for surge, sway, heave, roll, pitch, yaw:

\[
(M+A_\infty)\ddot{\boldsymbol{\xi}}
  +\int_0^t K(t-\tau)\,\dot{\boldsymbol{\xi}}(\tau)\,d\tau
  +C\boldsymbol{\xi}
  =\mathbf{F}(t)
\]

DOF order: `|0 surge|1 sway|2 heave|3 roll|4 pitch|5 yaw|`.

| Operator | Source |
| --- | --- |
| \(A_\infty\) | Strip Lewis-circle \(a_{33}^\infty\), \(a_{22}^\infty\); surge \(C_x\rho\nabla\) with **nominal** \(C_x=0.5\) (**unverified**) |
| \(K_{ij}(t)\) | Cosine transform of strip \(B_{ij}(\omega)\); active P–S / F–A symmetry channels only |
| \(C\) | Waterplane heave–pitch from strips; roll \(C_{44}=\rho g\nabla\mathrm{GM}_T\) |
| \(\mathbf{F}\) | Froude–Krylov + **STF strip Haskind diffraction** (head seas → heave/pitch) |

### Bounded memory / truncation

Kernels truncate at horizon \(T=(n-1)\Delta t\). Engineering L1 tail estimate
`truncation_tail_l1` assumes late-time exponential decay from the last two
samples: \(\int_T^\infty |K|\approx |K(T)|\,\tau\). Convolution history is
fixed-length (deterministic, no `HashMap`).

### Validation (measured)

| Check | Tolerance | Measured |
| --- | --- | --- |
| Roll decay RK4 vs analytic (\(\dot\phi(0)=0\)) | **0.05%** | **~1e-14** |
| Heave natural \(\omega_3\) (coupled \(M+A_\infty,C\)) vs strip | **0.5%** | **~0.078%** |
| Long-wave head-seas heave RAO \|RAO−1\| (strip FD) | **2%** | **~0.0038%** |
| Undamped (\(K\equiv0\)) energy envelope, RK4, dt=0.01, ~40 s | **1e-8** | **~2.2e-10** |
| RK4 dt-halving order vs analytic HO (zero-crossing) | **[3.5, 4.5]** | **~4.00** |
| Kernel truncation: \(\max\|\Delta z\|\) vs \(20\cdot\mathrm{L1}_{tail}\cdot\|v\|_\infty T/(m+A_\infty)\) | factor **20** | passes (≪) |
| Box barge heave/pitch RAO vs closed-form strip integrals (incl. resonance) | **0.5% rel** (abs floor \(5\cdot10^{-4}\)) | **≲ 0.4%** |
| \(A_\infty\), \(C\) matrix asymmetry | **1e-12** rel | **~0** |
| Cross-OS `cummins6_hash` (semi-implicit path; unchanged) | locked | `b72a60e18efd983d` |

**Energy-drift root cause (M23).** The M21 ~1.3% undamped envelope drift at
`dt=0.01` was **semi-implicit Euler** (order-1), not a non-symmetric
\(A_\infty\)/\(C\) coupling or kernel artefact. Matrices are symmetric to machine
epsilon. Default [`Cummins6Dof::step`] stays semi-implicit so the locked hash is
unchanged; oracles use [`Cummins6Dof::step_rk4`].

---

## Forward speed (`pe_hydro::forward_speed`)

Encounter frequency:

\[
\omega_e = \omega - U k \cos\chi
\]

(\(\chi=0\) head seas). Deep-water shortcut \(k=\omega^2/g\).

STF (1970) forward-speed strip corrections to heave–pitch radiation damping
(end-station + \(\int a_{33}\,dx\) terms). At \(U=0\) matrices match zero-speed
strip bit-for-bit. For a uniform box, \(B_{33}/B_{55}\) end-difference terms
cancel; \(B_{35}/B_{53}\) pick up \(\pm U\int a_{33}\).

---

## Resistance (`pe_hydro::resistance`)

### ITTC-1957 friction line (exact)

\[
C_f = \frac{0.075}{(\log_{10} R_n - 2)^2},\qquad R_n = VL/\nu
\]

| Check | Tolerance | Measured |
| --- | --- | --- |
| \(C_f(R_n=10^9)=0.075/49\) | **1e-14 rel** | bit-exact algebra |

`HullDrag::ittc57` evaluates \(C_f(R_n)\) each step. Residual \(C_r\) remains
caller-supplied. **Holtrop–Mennen** full coefficient tables are **not**
implemented (not verified against a primary citation in-tree). Linear-Froude
residual surrogate is marked **unverified**.

Nominal \(\nu\) defaults (`NU_FRESHWATER_20C_NOMINAL`, …) are **unverified**
handbook order — override for exact oracle work.

---

## Maneuvering (`pe_hydro::maneuvering`)

### Nomoto first-order K–T (validated)

\[
T\dot r + r = K\delta
\]

Analytic step (\(r(0)=0\)): \(r(t)=K\delta(1-e^{-t/T})\),
\(\psi(t)=K\delta\bigl(t-T(1-e^{-t/T})\bigr)\). Steady turning radius
\(R=U/|r_\infty|\).

| Check | Tolerance | Measured |
| --- | --- | --- |
| Numerical Nomoto step vs analytic (\(dt=10^{-3}\), 40 s) | **0.2%** | **~2e-4% / 0.003%** |
| Zig-zag 10°/10° rudder switch | qualitative | passes |

### KVLCC2 MMG (Yasukawa & Yoshimura 2015)

Coefficients transcribed from the **primary** paper Table 3 (Springer PDF),
including \(w_{P0}=0.40\), \((k_0,k_1,k_2)=(0.2931,-0.2753,-0.1385)\),
\(x'_R=-0.5\). L7-model geometry from Table 1. Turning indices vs Table 4 Exp.
(MARIN / SIMMAN L7 free-running):

| Check | Tolerance | Measured |
| --- | --- | --- |
| \(+35^\circ\) advance \(A_D'/L_{pp}\) vs Exp. 3.25 | **12%** | **~10.4%** (got ≈2.91) |
| \(+35^\circ\) tactical diameter \(D_T'/L_{pp}\) vs Exp. 3.34 | **12%** | **~6.4%** (got ≈3.13) |

Yasukawa’s own Cal. was 3.31 / 3.36 (~2% vs Exp.). The 12% tol covers
clean-room discretisation / self-propulsion \(n_P\) solve without loosening
past a documented band. Demo `nominal_linear_mmg_unverified` remains for
tutorials only.

Propeller / rudder **helper** APIs outside the KVLCC2 path stay marked
**unverified** unless the caller supplies a cited map.

---

## Scene / Python wiring

- **Python (`physics_engine_rs`):** `canonical_cummins6_hash`, `ittc57_friction_cf`,
  `nomoto_r_step` / `nomoto_psi_step`, `encounter_frequency_py`, plus existing
  Morison / strip / Cummins heave helpers. Tests: `tests/test_m21_seakeeping_rust.py`.
- **Scene (`pe-scene`):** ContactWorld schema still carries stage-1 forces only
  (see [`SCENE.md`](SCENE.md)). M21 ship models are constructed via the Rust /
  Python API (`Cummins6Dof`, `NomotoState`, `HullDrag::ittc57`) and can sit
  beside a scene-loaded rigid body; full hydro `ForceDesc` variants remain
  deferred to avoid schema churn on locked scene hashes.

---

## Deferred

Tabulated Vugts / Frank close-fit sections; 3-D panel diffraction;
MacCamy–Fuchs; **Holtrop–Mennen** full resistance tables (1982 ISP formulas /
coefficients not verified against a primary reproduction in-tree — left
unimplemented); viscous roll damping maps; additional propeller open-water
curves beyond the KVLCC2 \(K_T\) polynomial.
