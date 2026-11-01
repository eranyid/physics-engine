# Aerodynamics & 6-DOF flight dynamics (`pe-aero`)

SI units; portable `libm`; no FMA. Body axes: `+x` nose, `+y` right, `+z` up
(world z-up when level). `α = atan2(−v_z, v_x)`, `β = asin(v_y/V)`.

Stage-3 polar / thrust / ISA: see [`DESIGN.md`](DESIGN.md). Compressibility /
NASA check-cases: M9. BEM rotors: [`ROTOR.md`](ROTOR.md).

## 6-DOF stability-derivative FDM

Module: `pe_aero::fdm`.

| Piece | Role |
| --- | --- |
| `StabilityDerivatives` | `C_L,C_D,C_Y,C_l,C_m,C_n` vs α, β, α̇, `p̂,q̂,r̂`, δe/δa/δr |
| `SixDofModel` / `SixDofAero` | Body forces/moments + thrust with COM offset; ISA or constant ρ; steady wind; optional held Dryden gust |
| `trim_steady_level` | Newton solve for `(α, δe, T)` at fixed `V` (wings-level) |
| `linearise_longitudinal` | Numerical `A` for state `[u,w,q,θ]` (α̇ fixed-point) |
| `linearise_lateral` | Numerical `A` for state `[v,p,r,φ]` |
| `classify_*_modes` | Bairstow eigen → phugoid / short-period; roll / spiral / Dutch roll |
| `phugoid_approximation` | Lanchester / Etkin sanity only: `ω ≈ g√2/V`, `ζ ≈ C_D/(√2 C_L)` |
| `short_period_approximation[_with_alphadot]` | Reduced-order α–q; shows `C_mα̇` damping effect |
| `wieselsberger_sigma` / `mccormick_lift_factor` | Ground effect vs `h/b` |
| `navion_nelson` | Published Navion table (verified) |

Nondimensional rates: `p̂ = p b/(2V)`, `q̂ = q c̄/(2V)`, `r̂ = r b/(2V)`,
`α̂̇ = α̇ c̄/(2V)`.

### Navion data provenance (`verified: true`)

| Quantity | Value | Source |
| --- | --- | --- |
| `W` | 2750 lbf | Nelson App. B Table B.1 |
| `I_x,I_y,I_z`, `I_xz` | 1048, 3000, 3530 slug·ft²; `I_xz=0` | Nelson App. B |
| `S,b,c̄` | 184 ft², 33.4 ft, 5.7 ft | Nelson geometry; NASA TN D-6643 Table I |
| `V` (derivative set) | 176 ft/s, sea level | Nelson / Teper |
| `C_L,C_D` trim | 0.41, 0.05 | Nelson Table B.1 |
| `C_Lα,C_Dα,C_mα,C_Lα̇,C_mα̇,C_Lq,C_mq,…` | `4.44, 0.33, −0.683, 0.0, −4.36, 3.8, −9.96, …` | Nelson Table B.1 |
| Lateral set | `C_Yβ=−0.564`, `C_nβ=+0.071`, … | Nelson / Teper reproduction |

Anything not listed above that appears in demos (`unverified_demo_model`) is
**unverified**.

JSBSim example aircraft were **not** vendored (LGPL); only published numeric
comparisons against Nelson/NASA tables are used.

### Published eigenroots (oracle targets)

Cal State thesis *Stability and control characteristics of the Navion aircraft*
(LONG.for / LAT.for on Nelson App. B / Teper STI TR 176-1 data), Appendix II:

| Mode | Root | Derived |
| --- | --- | --- |
| Phugoid | `−0.01640041 ± 0.2135064 i` | `ω_n=0.2135064`, `ζ=0.076589` |
| Short-period | `−2.444170 ± 2.537795 i` | `ω_n=3.5228`, `ζ=0.693696` |
| Spiral | `−0.0088` | — |
| Roll subsidence | `−8.2550` | — |
| Dutch roll | `−0.4743 ± 2.3109 i` | `ω_n=2.3591`, `ζ=0.2010` |

### Validation (measured, Linux x86_64)

| Check | Tolerance | Measured |
| --- | --- | --- |
| Navion phugoid vs published root | `ω` rel ≤ 0.5%; `ζ` abs ≤ 0.012 | `ω≈0.21322` (0.13%), `ζ≈0.0817` |
| Navion short-period vs published | `ω` rel ≤ 3%; `ζ` abs ≤ 0.012 | `ω≈3.573` (1.4%), `ζ≈0.6987` |
| Lanchester phugoid sanity (not primary) | `ω` rel ≤ 0.25; `ζ` abs ≤ 0.05 | `ω` rel ≈ 0.175; `ζ` abs ≈ 0.009 |
| `C_mα̇` raises short-period `ζ` | Δζ > 0.05 | `0.575 → 0.699` |
| Lateral roll vs `−8.255` | rel ≤ 8% | `≈ −8.533` (3.4%) |
| Lateral spiral vs `−0.0088` | abs ≤ 0.002 | `≈ −0.00878` |
| Dutch roll vs `ω=2.3591`, `ζ=0.2010` | `ω` rel ≤ 12%; `ζ` abs ≤ 0.02 | `ω≈2.230` (5.5%), `ζ≈0.2005` |
| Wieselsberger `σ(h/b)` / McCormick `K_L=2−σ` | rel 1e-12 | bit-level vs formula |
| Dryden `v`/`w` RMS vs `σ` | rel ≤ 8% / 5% | `rms_v≈1.452` (`σ=1.5`); `rms_w≈1.001` |
| Dryden `v` PSD log-slope in `[2,20]` rad/s | abs ≤ 0.25 from −2 | `≈ −1.925` |
| Steady glide `L/D` (algebraic polar max) | rel 1e-10 | bit-level |
| Spring+gravity energy oscillation (symplectic Euler, dt=1e-3, 10 s) | peak–peak / E₀ ≤ 1e-2 | ≈ 6.96e-3 |
| Quaternion `\|q\|−1` over 5 s RK4 FDM | ≤ 1e-12 | ~0 (renormalised) |
| Coordinated turn `n=1/cosφ`, `ψ̇=g tanφ/V` | rel ≤ 0.5% | exact (≤ 1e-15) |
| Trim residual `‖(F_x,F_z,M_y)‖` | ≤ 1e-3 | ≈ 5.6e-5 N / N·m |
| Thrust offset moment `r×T` | abs 1e-9 | exact |
| Cross-OS FDM body hash | locked | `ba6d68e43c69d315` |
| Cross-OS Dryden gust hash | locked | `2895b75a14f621c5` |
| Cross-OS FDM+held-gust hash | locked | `58f491fb7f3b34a6` |

Never loosen a bound silently — update this table with the new measurement.

### Dryden turbulence (MIL-F-8785C)

- `u`: first-order Dryden filter, variance-matched discrete OU.
- `v`, `w`: **second-order** Dryden
  `H(s)/σ = √(L/(πV)) (1+√3 τ s)/(1+τ s)²`, `τ=L/V`, partial-fraction
  cascade with stationary-variance gain so `RMS → σ`.
- RNG: seeded `xoshiro256**` (portable).

**Integrator coupling.** `HeldDrydenGust` advances once per major step
`n = ⌊t/Δt⌋` and **holds** the gust constant inside `[n Δt,(n+1)Δt)`. RK4
stages that land in the same interval therefore see identical wind, so replay
of a fixed seed/`Δt` schedule is bit-deterministic even though
`ForceGenerator::apply` is `&self` (interior `Mutex` keyed by step index;
rebuild-from-seed on rewind).

### Ground effect

Wieselsberger (NACA TM 77) / McCormick: induced-drag factor
`σ = (16 h/b)² / (1 + (16 h/b)²)`; lift factor `K_L = 2 − σ` at fixed
geometric α. Applied in `eval_aero_body` when `SixDofModel::ground_height` is set.

### Determinism helpers

```bash
cargo run -p pe-aero --bin fdm_hash --release
# line 1: FDM body; line 2: Dryden gust sequence; line 3: FDM+held gust
```

Python: `physics_engine_rs.canonical_fdm_hash()`,
`canonical_dryden_gust_hash()`, `canonical_fdm_gust_hash()`.

### Hash changelog

| Hash | Old → new | Reason |
| --- | --- | --- |
| FDM body | `fb8c25d5fe2184fd` → `ba6d68e43c69d315` | Body-+y `C_Y` side-force sign fix (Dutch-roll damping) |
| Dryden gust | `233148553b199f90` → `2895b75a14f621c5` | 2nd-order MIL-F-8785C `v`/`w` filters |
| FDM+gust | (new) `58f491fb7f3b34a6` | Held Dryden inside `SixDofAero` + RK4 |
| Ballistics volley | `a9ec62a39bbd3dc4` → `7b0162d346bf9f1b` | Same Dryden change (volley seeds `pe-aero` gusts) |

### Deferred

- Certified airfoil polars; compressibility coupling into the 6-DOF derivative set
- Lagged `α̇` inside `SixDofAero::apply` (linearisation already uses gravity-consistent fixed-point)
