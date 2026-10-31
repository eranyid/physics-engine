# Ammo profiles (`AmmoProfile`)

Versioned, serde-loadable ammunition descriptors for per-bullet simulation.
Fields are **external-ballistics parameters only** — no design or manufacturing
details.

**Important:** Bundled ammo profiles are **nominal and unverified**
(`verified: false`). The `G1StandIn` / `G7StandIn` drag tags are **analytic
stand-ins** — they are **not** the published BRL/McCoy G1 or G7 standard drag
functions, and a BC of 1 does **not** reproduce an official G1/G7 deceleration
scale.

## Schema (`version = 1`)

| Field | SI unit | Meaning |
| --- | --- | --- |
| `name` | — | Display / lookup id |
| `caliber_m` | m | Bullet diameter |
| `mass_kg` | kg | Bullet mass |
| `muzzle_velocity_m_s` | m/s | Reference muzzle speed |
| `bc.value_si` | kg/m² | Sectional BC = mass / reference area |
| `bc.drag_model` | enum | `G1StandIn`, `G7StandIn`, `CustomTable`, `Analytic` |
| `bc.table` | optional | Custom `(mach[], cd[])` knots |
| `length_m` | m | Bullet length (Magnus / form estimates) |
| `twist_m_per_rev` | m/rev | Barrel twist; spin = `v / twist` |
| `form_factor` | — | Multiplier on `C_D` |
| `barrel_length_ref_m` | m? | Optional `v ∝ √(L/L_ref)` scaling |
| `velocity_std_m_s` | m/s | Per-shot muzzle speed σ |
| `dispersion_std_rad` | rad | Aim cone 1σ |
| `provenance` | — | `{ note, verified }` |

Every numeric field in bundled data carries
`note: "nominal, approximate, source: …"` and `verified: false` unless a
primary source was fetched and checked.

## Drag models

- **Analytic** — built-in `C_D(M)` (subsonic plateau, transonic rise, supersonic
  decay). Documented engineering model, not a published projectile law.
- **G1StandIn / G7StandIn** — **analytic stand-ins only** (same curve; G7
  scaled ×0.85). They are **not** verified G1/G7 standard drag functions from
  BRL/McCoy tables and must not be cited as such.
- **CustomTable** — caller-supplied Mach→`C_D` knots with linear interpolation.

## Adding custom ammo

JSON example:

```json
{
  "version": 1,
  "name": "my_round",
  "caliber_m": 0.00762,
  "mass_kg": 0.0095,
  "muzzle_velocity_m_s": 800.0,
  "bc": {
    "value_si": 4000.0,
    "drag_model": "Analytic",
    "table": null,
    "provenance": {
      "note": "nominal, approximate, source: studio tuning",
      "verified": false
    }
  },
  "length_m": 0.03,
  "twist_m_per_rev": 0.3048,
  "form_factor": 1.0,
  "barrel_length_ref_m": null,
  "velocity_std_m_s": 5.0,
  "dispersion_std_rad": 0.001,
  "provenance": {
    "note": "nominal, approximate, source: studio tuning",
    "verified": false
  }
}
```

Rust: `AmmoProfile::from_json` / `from_toml`. Python: `ammo_from_json_muzzle_velocity`
and `ammo_profile_json` for bundled names.

## Bundled class profiles

`generic_pistol`, `generic_smg`, `generic_intermediate_rifle`,
`generic_battle_rifle`, `generic_sniper_long_range`,
`generic_heavy_machine_gun`, `generic_shotgun_slug`, `generic_subsonic`.

All are **nominal / approximate / unverified** (`verified: false`), using
rounded public-domain / encyclopedia-class ballpark figures for game
simulation. **Do not treat as manufacturer load data** or as verified ballistic
coefficients.
