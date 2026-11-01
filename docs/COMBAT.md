# Combat module (`pe-combat`)

Combat physics for games and studio simulation: what happens **after** a
projectile from [`pe-ballistics`](BALLISTICS.md) reaches a surface. Delivered
in phases:

| Phase | Contents | Status |
| --- | --- | --- |
| 1 | Hit-resolver hook, armor materials / stacks, plate perforation, media penetration, ricochet, post-perforation yaw, spall, energy ledger | **done** |
| 2 | Hit zones on characters, damage / bleeding / incapacitation, per-zone blast | planned |
| 3 | Humanoid ragdoll on `pe-collision` joints, active-ragdoll motors, flinch / stagger / knockdown | planned |
| 4 | Weapon handling: free recoil with gas jet, muzzle climb, sway, barrel heat; `combat_hash`, Python bindings | planned |

**Scope.** All material constants are caller-supplied or bundled as nominal
`generic_*` rows (`provenance.verified == false`), tuned for plausible game
behaviour. These are engineering approximations, not armor or weapon design
tools.

## Integration: `HitResolver`

`ProjectileSystem::step_with_resolver(dt, shapes, body_ids, &mut resolver)`
passes every swept hit to a `HitResolver`, which returns:

- `HitOutcome::Stop` — the bullet dies at the surface (the classic behaviour;
  `ProjectileSystem::step` uses the built-in `StopOnHit`, so existing
  results and the locked ballistics hash are unchanged), or
- `HitOutcome::Continue { position, velocity, mass_kg, drag_multiplier }` —
  perforation (exit point) or ricochet (surface point).

After `Continue` the bullet travels in a straight line for the rest of the
sub-step, with the shape it just hit excluded. Up to
`MAX_INTERACTIONS_PER_SUBSTEP = 8` surfaces can be crossed in one sub-step,
with no tunneling. `Projectile::drag_multiplier` (default 1) scales drag
after yaw or tumble. `HitEvent::shape_index` identifies the hit shape.

```rust
use pe_combat::{bundled_materials, ArmorStack, CombatResolver};

let mats = bundled_materials();
let wood = mats.iter().position(|m| m.name == "generic_wood").unwrap();
let mut resolver = CombatResolver::new(mats, vec![Some(ArmorStack::solid(wood))]);
sys.step_with_resolver(1.0 / 60.0, &shapes, &body_ids, &mut resolver);
for r in resolver.records.drain(..) {
    // r.kind, r.entry, r.exit, r.e_deposited_j, r.spall …
}
```

## Materials and stacks

`ArmorMaterial` (versioned serde JSON, `ARMOR_MATERIAL_VERSION = 1`) has one
of two penetration models:

| Model | Use | Law |
| --- | --- | --- |
| `Plate` | thin hard plates | Lambert–Jonas `V50`, Recht–Ipson residual |
| `Medium` | soil, wood, concrete, water, tissue simulant | Poncelet retardation along the real path |

An `ArmorStack` is an ordered list of `ArmorLayer { material, thickness_m }`,
outermost first. `thickness_m ≤ 0` means "use the shape's geometry": the
chord through the shape, found by a reverse cast (`hitscan::shape_exit`).
For plates the normal thickness is `chord · cos θ`; for media it is the
chord itself.

Bundled (nominal): `generic_hard_steel_plate`, `generic_mild_steel_plate`,
`generic_aluminium_plate`, `generic_concrete`, `generic_wood`,
`generic_soil`, `generic_water`, `generic_tissue_simulant`.
`cargo run -p pe-combat --example penetration_table` prints what each bundled
ammo class does against each of them.

## Models

| Quantity | Formula |
| --- | --- |
| Lambert `z` | `z = (t/d) sec^n θ` (`n = 0.75`, θ clamped to 89°) |
| Ballistic limit | `V50 = k (L/d)^0.15 √(f(z) d³/m)`, `f(z) = z + e^{−z} − 1` (`L/d ≥ 1`) |
| Residual speed | `v_r = a (v^p − V50^p)^{1/p}` (`recht_ipson_residual`) |
| Plate erosion | `m_out = m (1 − e_z z)`, floored at `0.2 m` |
| Medium exit speed | `v(x)² = (a/b + v₀²) e^{−2Abx/m} − a/b`; stops at `poncelet_depth` |
| Ricochet probability | `P = 1 / (1 + e^{(β − β_c)/w})`, graze `β = π/2 − θ` (outer layer only) |
| Ricochet velocity | `v' = e_t v_t − e_n v_n`, then a Gaussian jitter that stays off the surface |
| Yaw / tumble | `k_drag ← k_drag (1 + g · ΔE/E)` per layer; exit deflection std `σ · ΔE/E` |
| Spall | `E_spall = f_s ΔE_layer`; Mott masses up to `mass_factor · ρ_t A t_LOS`; uniform cone |

The SI Lambert constant relates to the classic cgs one (d in cm, m in g) by
`k_SI = k_cgs · √1000` (4000 → ≈ 126 491).

**Energy ledger.** Every `ImpactRecord` satisfies
`KE_in = KE_out + E_deposited + E_spall` to round-off. The total kinetic
energy of the spall fragments equals `E_spall`.

## Determinism

Every random choice (ricochet roll, deflection, spall masses and directions)
draws from the hit projectile's own per-shot RNG (`Projectile::rng`). The
outcome for a `ShotId` is therefore bit-identical regardless of how many other
bullets are in flight (`same_shot_same_outcome_regardless_of_crowd`).

## Validation (`crates/pe-combat/tests/penetration_oracles.rs`)

- `V50` matches the closed form; `z` scales as `sec^n θ` (and `1/cos θ` when `n = 1`).
- With `p = 2`, the plate residual equals `a √(v² − V50²)`; the plate stops just below `V50`.
- The Poncelet exit-speed closed form matches RK4 integration (1e-9).
  Exit speed vanishes at `poncelet_depth`, and the reported stop depth matches it.
- Grazing ricochet reproduces `e_t v_t − e_n v_n` exactly.
- Proptest (256 random stacks and hits): the energy ledger closes to 1e-9;
  ricochet never gains energy and leaves the surface; `drag_factor ≥ 1`.
- System tests: a rifle perforates a wood wall (exit at the far face) and
  stops in steel. A 1 mm plate at `dt = 1/60` is not tunneled. Shapes with no
  armor stop the bullet exactly like the classic path.

## Limits

- Point-mass projectile; the yaw after perforation is a drag multiplier, not 6-DOF.
- Lambert–Jonas was fitted to long-rod penetrators; for small arms it serves as
  a tunable scaling law, not a prediction.
- Medium layers stop at the first layer that the bullet does not exit.
- Spall fragments are returned as data. Phase 2 spawns them as projectiles.

## References

- W. P. Lambert, *Towards Simplified Penetration Equations*, BRL-R-2036 (1978).
- R. F. Recht, T. W. Ipson, *Ballistic perforation dynamics*, J. Appl. Mech. 30 (1963).
- J.-V. Poncelet, penetration resistance law (19th c.), engineering form.
- N. F. Mott, *Fragmentation of shell cases*, Proc. R. Soc. A 189 (1947).
