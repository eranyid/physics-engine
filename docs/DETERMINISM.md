# Determinism policy (stage 1 / M2)

## Guarantees

On the **Rust step path** (`pe-core` integrators + force apply + quaternion `exp_rotvec`):

1. **f64 only**; no `target-cpu=native`, no explicit FMA (`mul_add` unused).
2. Elementary functions go through the pure-Rust [`libm`](https://crates.io/crates/libm) crate
   (`pe_math::libm_ops`: `sin`/`cos`/`sqrt`/`exp`/`log`/`tanh`/`fabs`/`hypot`/`atan2`).
3. Fixed body and generator order (`Vec`, never `HashMap` iteration).
4. Identical inputs → **bit-identical** packed state on the same ISA across OS when (1)–(3) hold.

Cross-check helper: `cargo run -p pe-core --bin canonical_hash --release` and
`physics_engine_rs.canonical_determinism_hash()`.

### Canonical scenario hash (locked)

Scenario (must stay in sync in the binary, Rust test, and Python helper):

- Bodies: mass 1 at `(0,0,10)` with `v=(1,0,0)`; mass 2 at `(1,0,0)` with
  `v=(0,0.5,0)`, `I=diag(0.4,0.5,0.6)`, `ω=(0.1,0,0.2)`.
- Generators: `UniformGravity`, `LinearDrag(0.01)`, `Spring(0→(0,0,10), k=5)`,
  `ConstantTorque(1, (0,0,0.05))`.
- Integrate with **RK4**, `t_end=2`, `dt=1e-3`.
- Hash: FNV-1a 64 over little-endian IEEE bits of `State::pack()`.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `ca172fe3e3e57c79` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

If a platform ever diverges, **do not silently loosen**: record the OS, toolchain,
exact first differing packed index / float bits, and root cause (usually accidental
platform `libm` or FMA) in this table.

## Oracle vs Python reference

The Python package uses NumPy/SciPy (platform libm, vectorized reductions). Therefore:

| Scenario | Measured max \|Rust−Python\| (Linux x86_64) | Notes |
| --- | --- | --- |
| Projectile / free fall (RK4 / Verlet) | **0** (bit-identical in practice) | Polynomial / near-exact |
| Linear drag (RK4, 12 s) | **0** | Same force algebra; libm≈NumPy here |
| Harmonic oscillator (RK4) | **0** | Scalar spring |
| Torque-free `L` / KE (5 s RK4) | ~4e-16 abs / ~7e-16 rel KE | Quaternion `exp_rotvec` libm vs NumPy |
| Kepler half-period (RK4) | ≤ 1e-9 rel on separation (slow) | NumPy `einsum` pair loop ≠ scalar `i<j` order |
| Quadratic drag | ≤ 1e-10 rel (oracle test bound) | `sqrt` in \|u\| |

Within Rust, `test_rust_determinism_repeat` and `determinism_bitwise_repeat` require
**bit-identical** packed state on repeat.
