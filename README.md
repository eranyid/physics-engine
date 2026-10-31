# physics-engine

Rigorous, reusable real-world physics simulation (Python, NumPy/SciPy, SI units).
Design: [docs/DESIGN.md](docs/DESIGN.md). Status: stage 1 (rigid bodies, gravity, drag, integrators).

## Run tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest -q
```

## Example

```python
import numpy as np
from physics_engine import RigidBody, World, forces, integrators
w = World([RigidBody(mass=1.0, velocity=[10, 0, 10])], [forces.UniformGravity()])
w.run(t_end=1.0, dt=1e-3, integrator=integrators.velocity_verlet)
print(w.state.x)
```
