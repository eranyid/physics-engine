# Combat module (`pe-combat`)

Combat physics for games and studio simulation: what happens **after** a
projectile from [`pe-ballistics`](BALLISTICS.md) reaches a surface. Delivered
in phases:

| Phase | Contents | Status |
| --- | --- | --- |
| 1 | Hit-resolver hook, armor materials / stacks, plate perforation, media penetration, ricochet, post-perforation yaw, spall, energy ledger | **done** |
| 2 | Hit zones on characters, damage / bleeding / incapacitation, per-zone blast, spall and fragments as projectiles | **done** |
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
`ProjectileSystem::spawn_with_velocity` spawns a projectile with an explicit
state (fragments) and no noise.

`CombatResolver` looks up each shape index in a `TargetTable`:
`Target::Inert` (stop), `Target::Armor(ArmorStack)` or
`Target::Zone { character, zone, stack }`. `CombatResolver::new(materials,
Vec<Option<ArmorStack>>)` still builds an armor-only table.

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

## Characters and damage (phase 2)

`Character::humanoid(id, height_m, tissue_material)` builds 13 zones on body
pose 0. The character stands facing +x, z up, with the origin between the
feet. `Character::equip(&[ZoneKind…], &ArmorStack)` adds worn armor.

| Zone | Shape | Damage factor | Bleed [L/s/J] | Instant incap |
| --- | --- | --- | --- | --- |
| Head | sphere | 3.0 | 1e-5 | ≥ 40 J tissue |
| Neck | capsule | 2.0 | 6e-5 | ≥ 150 J |
| UpperTorso / LowerTorso / Pelvis | box | 1.0 / 0.8 / 0.7 | 2–2.5e-5 | – |
| Upper arms / forearms | capsule | 0.35 | 1.5e-5 | – |
| Thighs / shins | capsule | 0.5 / 0.35 | 3e-5 / 1e-5 | – |

```rust
let mut targets = TargetTable::new();
targets.push_character(0, &ch);                  // one Target::Zone per zone
let shapes = ch.zone_shapes(&[(pos, rot)]);      // same order
let mut res = CombatResolver::with_targets(bundled_materials(), targets);
sys.step_with_resolver(dt, &shapes, &ids, &mut res);
apply_damage_events(&mut characters, &res.damage_events);
characters[0].health.step(dt);                   // bleeding, pain
let m = characters[0].health.modifiers();        // aim sway, speed, arms
```

A zone resolves its worn armor followed by tissue (chord through the shape).
`StackResult::layer_dissipated_j` gives the energy left in each layer. Spall
from worn armor that the bullet breaches goes into the wearer.

| Quantity | Formula |
| --- | --- |
| Blunt | `blunt = blunt_factor · E_armor` |
| Damage | `HP = damage_factor · (E_tissue + 0.25 · blunt) / 8 J` |
| Bleeding | `r0 = bleed · E_tissue`, `r(t) = r0 e^{−t/τ}` (τ = 120 s), integrated exactly |
| Pain | `+= (E_tissue + blunt) / 400 J`, decays with τ = 30 s |
| Limbs | `function *= e^{−g (E_tissue + 0.25 blunt)}`, `g = 1/600 J` |
| Condition | Incapacitated at 100 HP, 30 % blood loss or pain 3. Dead at 150 HP or 45 % blood loss |

**Blast** (`blast_on_character`): for each zone, Kinney peak overpressure and
Friedlander impulse on the zone's projected area facing the charge, against a
pressure–impulse curve `(P − P0)(I − I0) = C`. Damage is
`damage_hp · (1 + ln excess)`, and the push impulse `I · A · d̂` is returned
for hit reactions. With 1 kg TNT-equivalent, 2 m is lethal, 3 m
incapacitates, and from 4 m on there is no injury. Shielding is not modelled.

**Fragments** (`fragments.rs`): `fragment_profile` (sphere-equivalent,
analytic drag × 3), `spawn_spall` / `spawn_all_spall` for `ImpactRecord`
spall, `spawn_fragment_burst` for an isotropic Mott burst. Fragment
`ShotId`s carry `FRAGMENT_ID_BIT` and are derived from the parent impact, so
they don't depend on how many other projectiles exist. Mass is scaled through
`drag_multiplier = (m/m_nom)^{2/3}`.

`cargo run -p pe-combat --example damage_demo` prints several shots and blasts.

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

Phase 2 (`crates/pe-combat/tests/damage.rs`):

- Proptest: the per-layer ledger `Σ layer + spall + KE_out = KE_in`, with one
  entry per layer reached.
- Bleeding matches `Σ r0 τ (1 − e^{−t/τ})` to 1e-12 with mixed `dt`; heavy
  bleeding goes Wounded → Incapacitated → Dead.
- A rifle through an unarmored torso leaves a wound track equal to the box
  depth and keeps flying. A 10 mm plate stops it, giving blunt damage only and
  no bleeding. With a 4 mm plate, `E_armor + E_tissue = KE_in − KE_out` exactly.
- A head shot incapacitates instantly. Repeated thigh hits reduce speed and
  increase sway monotonically.
- Blast damage is monotone in distance. Push equals `face_positive_impulse · A`
  and points away from the charge. The projected area is exact.
- Spawned spall KE equals `e_spall_j`. Spall from a plate reaches a character
  1 m behind it, and the outcome is identical with 200 other bullets in flight.
  Fragment bursts are deterministic.

## Limits

- Point-mass projectile; the yaw after perforation is a drag multiplier, not 6-DOF.
- Lambert–Jonas was fitted to long-rod penetrators; for small arms it serves as
  a tunable scaling law, not a prediction.
- Medium layers stop at the first layer that the bullet does not exit.
- Zones follow body poses, but there is no ragdoll yet (phase 3); zone shapes are rebuilt per step by the caller.
- Health and vulnerability constants are game tuning, not medical data.

## References

- W. P. Lambert, *Towards Simplified Penetration Equations*, BRL-R-2036 (1978).
- R. F. Recht, T. W. Ipson, *Ballistic perforation dynamics*, J. Appl. Mech. 30 (1963).
- J.-V. Poncelet, penetration resistance law (19th c.), engineering form.
- N. F. Mott, *Fragmentation of shell cases*, Proc. R. Soc. A 189 (1947).
