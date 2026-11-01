# Aerodynamics & 6-DOF flight dynamics (`pe-aero`)

SI units; portable `libm`; no FMA. Body axes: `+x` nose, `+y` right, `+z` up
(world z-up when level). `α = atan2(−v_z, v_x)`, `β = asin(v_y/V)`.

Stage-3 polar / thrust / ISA: see [`DESIGN.md`](DESIGN.md). Compressibility /
Dryden / NASA check-cases: M9. BEM rotors: [`ROTOR.md`](ROTOR.md).

## 6-DOF stability-derivative FDM

Module: `pe_aero::fdm`.

| Piece | Role |
| --- | --- |
| `StabilityDerivatives` | `C_L,C_D,C_Y,C_l,C_m,C_n` vs α, β, `p̂,q̂,r̂`, δe/δa/δr |
| `SixDofModel` / `SixDofAero` | Body forces/moments + thrust with COM offset; ISA or constant ρ; steady wind |
| `trim_steady_level` | Newton solve for `(α, δe, T)` at fixed `V` (wings-level) |
| `linearise_longitudinal` | Numerical `A` for state `[u,w,q,θ]` |
| `classify_longitudinal_modes` | Bairstow eigen → phugoid / short-period |
| `phugoid_approximation` | Lanchester / Etkin: `ω ≈ g√2/V`, `ζ ≈ C_D/(√2 C_L)` |
| `coordinated_turn_*` | `n = 1/cos φ`, `ψ̇ = g tanφ / V` |
| `navion_nelson` | Published Navion table (verified) |

Nondimensional rates: `p̂ = p b/(2V)`, `q̂ = q c̄/(2V)`, `r̂ = r b/(2V)`.

### Navion data provenance (`verified: true`)

| Quantity | Value | Source |
| --- | --- | --- |
| `W` | 2750 lbf | Nelson App. B Table B.1 |
| `I_x,I_y,I_z` | 1048, 3000, 3530 slug·ft² | Nelson App. B |
| `S,b,c̄` | 184 ft², 33.4 ft, 5.7 ft | Nelson geometry; NASA TN D-6643 Table I |
| `V` (derivative set) | 176 ft/s, sea level | Nelson / Teper |
| `C_L,C_D` trim | 0.41, 0.05 | Nelson Table B.1 |
| `C_Lα,C_Dα,C_mα,C_mq,…` | per Table B.1 | Nelson; Teper STI TR 176-1; NASA CR-2144 |
| Lateral set | `C_Yβ=−0.564`, `C_nβ=+0.071`, … | Nelson / Teper reproduction |

Anything not listed above that appears in demos (`unverified_demo_model`) is
**unverified**.

JSBSim example aircraft were **not** vendored (LGPL); only published numeric
comparisons against Nelson/NASA tables are used.

### Validation (measured, Linux x86_64)

| Check | Tolerance | Measured |
| --- | --- | --- |
| Navion phugoid `ω_n` vs Lanchester `g√2/V` | rel ≤ 0.20 | ≈ 0.18 (`ω_eig≈0.213`, approx ≈ 0.259) |
| Navion phugoid `ζ` vs `C_D/(√2 C_L)` | abs ≤ 0.05 | ≈ 0.012 (`ζ_eig≈0.078`, approx ≈ 0.090) |
| Navion phugoid vs Nelson/thesis root `−0.0164±0.2135i` | `ω` rel ≤ 0.05; `ζ` abs ≤ 0.02 | match within 2% |
| Short-period `ω_n` > phugoid; `ζ ∈ (0.2, 1.5)` | band | `ω_sp≈3.57`, `ζ_sp≈0.58` |
| Steady glide `L/D` (algebraic polar max) | rel 1e-10 | bit-level |
| Spring+gravity energy oscillation (symplectic Euler, dt=1e-3, 10 s) | peak–peak / E₀ ≤ 1e-2 | ≈ 6.96e-3 |
| Quaternion `\|q\|−1` over 5 s RK4 FDM | ≤ 1e-12 | ~0 (renormalised) |
| Coordinated turn `n=1/cosφ`, `ψ̇=g tanφ/V` | rel ≤ 0.5% | exact (≤ 1e-15) |
| Trim residual `‖(F_x,F_z,M_y)‖` | ≤ 1e-3 | ≈ 5.6e-5 N / N·m |
| Thrust offset moment `r×T` | abs 1e-9 | exact |
| Cross-OS FDM body hash | locked | `fb8c25d5fe2184fd` |
| Cross-OS Dryden gust hash | locked | `233148553b199f90` |

Never loosen a bound silently — update this table with the new measurement.

### Determinism helpers

```bash
cargo run -p pe-aero --bin fdm_hash --release
# line 1: FDM body hash; line 2: Dryden gust-sequence hash
```

Python: `physics_engine_rs.canonical_fdm_hash()`,
`physics_engine_rs.canonical_dryden_gust_hash()`.

### Deferred

- Full 2nd-order Dryden `v/w` filters; `α̇` derivatives; ground effect
- Lateral-directional eigenmodes (Dutch roll / roll / spiral) as a first-class API
- Certified airfoil polars; compressibility coupling into the 6-DOF derivative set
- Folding Dryden into the RK4 `ForceGenerator` path (apply is immutable; gusts
  hashed separately today)
