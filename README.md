# physics-engine

Rigorous, reusable real-world physics simulation (SI units). Correctness first.

**Today:** Python reference oracle (NumPy/SciPy), stages 1–3 validated against
analytic results. **Direction:** Rust core + Python bindings (PyO3/maturin), with
Godot GDExtension later; Python kept to cross-check Rust.

- Design (as-built Python + invariants): [docs/DESIGN.md](docs/DESIGN.md)
- Roadmap (architecture, domains, milestones, survey): [docs/ROADMAP.md](docs/ROADMAP.md)

**Status:** stages 1–3 (Python).
- Stage 1: rigid bodies, gravity, drag, integrators, diagnostics
- Stage 2: collision shapes, broad/narrow phase, sequential-impulse contacts
- Stage 3: ISA atmosphere, aircraft aero (lift/drag/stall/thrust/pitch derivatives),
  ship hydrostatics (buoyancy, GM/GZ, heave/roll periods, hull drag)

Library layer only — no game, combat, or rendering code.

## Run tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest -q          # fast suite (skips @pytest.mark.slow)
.venv/bin/pytest -m slow -q  # long energy / stack / trim / roll runs
```

## Example (stage 1)

```python
import numpy as np
from physics_engine import RigidBody, World, forces, integrators
w = World([RigidBody(mass=1.0, velocity=[10, 0, 10])], [forces.UniformGravity()])
w.run(t_end=1.0, dt=1e-3, integrator=integrators.velocity_verlet)
print(w.state.x)
```

## Example (stage 3 aero trim)

```python
from physics_engine import aero, atmosphere
polar = aero.WingPolar(S=16, c=1.6, b=10, CD0=0.025, e=0.8, CL0=0.2, CL_alpha=5.7)
trim = aero.level_flight_trim(mass=1000.0, polar=polar, rho=atmosphere.RHO0)
print(trim["V"], trim["alpha"], trim["thrust"], trim["L_over_D"])
```

## Example (stage 3 barge GM)

```python
from physics_engine import hydro
hull = hydro.BoxHull(length=30, beam=8, depth=4)
draft, KG = 2.0, 1.5
print(hull.upright_GM(draft, KG))  # KB + BM - KG, BM = B^2/(12 T)
```
