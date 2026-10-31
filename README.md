# physics-engine

Rigorous, reusable real-world physics simulation (SI units). Correctness first.

**Today:** Python reference oracle (NumPy/SciPy), stages 1–3, plus a **Rust stages 1–3
core** (`pe-math` / `pe-core` / `pe-collision` / `pe-aero` / `pe-hydro`) with PyO3 bindings
(`physics_engine_rs`, abi3). Scene I/O (`pe-scene`), C ABI (`pe-ffi`), and a Godot 4
GDExtension (`godot/`) ship as M8. Python remains the oracle for cross-checks.

- Design: [docs/DESIGN.md](docs/DESIGN.md)
- Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- Determinism: [docs/DETERMINISM.md](docs/DETERMINISM.md)
- Benchmarks: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- Seakeeping / Morison: [docs/SEAKEEPING.md](docs/SEAKEEPING.md)
- Rotor BEM: [docs/ROTOR.md](docs/ROTOR.md)
- Scene I/O: [docs/SCENE.md](docs/SCENE.md)
- C ABI: [docs/FFI.md](docs/FFI.md)
- Godot: [docs/GODOT.md](docs/GODOT.md)

**Status:** M0–M9 done (incl. M6 convex); M11 Morison / seakeeping / BEM done; soft/fluids parallel tracks.
- Stage 1: rigid bodies, gravity, drag, springs, integrators, diagnostics (Python + Rust)
- Stage 2: collision shapes, broad/narrow phase, sequential-impulse contacts (Python + Rust)
- Stage 3: ISA / aero / hydro (Python + Rust)

## Run tests (Python oracle)

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest -q          # fast suite (skips @pytest.mark.slow)
.venv/bin/pytest -m slow -q  # long energy / stack / trim / roll runs
```

## Rust core + wheel

```bash
# Toolchain pinned in rust-toolchain.toml
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check

# abi3 wheel (requires maturin + venv)
.venv/bin/pip install maturin
.venv/bin/maturin build --release -m python/Cargo.toml
.venv/bin/pip install target/wheels/physics_engine_rs-*.whl   # or dist/*.whl
.venv/bin/pytest tests/test_oracle_rust.py tests/test_oracle_contacts_rust.py tests/test_oracle_aero_hydro_rust.py -q

# Cross-OS locked hashes
cargo run -p pe-core --bin canonical_hash --release
cargo run -p pe-collision --bin collision_hash --release
cargo run -p pe-hydro --bin aero_hydro_hash --release
cargo run -p pe-hydro --bin m11_hash --release
```

## Example (stage 1, Python oracle)

```python
from physics_engine import RigidBody, World, forces, integrators
w = World([RigidBody(mass=1.0, velocity=[10, 0, 10])], [forces.UniformGravity()])
w.run(t_end=1.0, dt=1e-3, integrator=integrators.velocity_verlet)
print(w.state.x)
```

## Example (stage 1, Rust wheel)

```python
import physics_engine_rs as pe
w = pe.World([pe.RigidBody(1.0, velocity=[10.0, 0.0, 10.0])], [pe.UniformGravity()])
w.run(1.0, 1e-3, "velocity_verlet")  # GIL released during stepping
print(w.x)
```

## Example (stage 2, Rust wheel)

```python
import physics_engine_rs as pe
body = pe.RigidBody(1.0, position=[0.0, 0.0, 1.0], inertia_body=[0.4, 0.4, 0.4])
w = pe.World([body], [pe.UniformGravity()])
cw = pe.ContactWorld(
    w,
    [pe.Sphere(0.1)],
    [pe.StaticCollider(pe.Plane([0.0, 0.0, 1.0]), material=pe.Material(0.8, 0.0))],
    [pe.Material(0.8, 0.0)],
)
cw.run(2.0, 1e-3)
print(cw.x)
```

## Example (stage 3 aero trim)

```python
from physics_engine import aero, atmosphere
polar = aero.WingPolar(S=16, c=1.6, b=10, CD0=0.025, e=0.8, CL0=0.2, CL_alpha=5.7)
trim = aero.level_flight_trim(mass=1000.0, polar=polar, rho=atmosphere.RHO0)
print(trim["V"], trim["alpha"], trim["thrust"], trim["L_over_D"])
```

## Example (stage 3, Rust wheel)

```python
import physics_engine_rs as pe
polar = pe.WingPolar(16.0, 1.6, 10.0, CD0=0.025, e=0.8, CL0=0.2, CL_alpha=5.7)
trim = pe.level_flight_trim(1000.0, polar, pe.RHO0)
hull = pe.BoxHull(20.0, 5.0, 3.0)
print(trim["V"], hull.upright_draft_for_mass(50_000.0, pe.RHO_FRESHWATER))
```
