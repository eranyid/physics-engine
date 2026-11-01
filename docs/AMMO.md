# Ammo profiles (`AmmoProfile`)

Versioned, serde-loadable ammunition descriptors for per-bullet simulation.
Fields are **external-ballistics parameters only** — no design or manufacturing
details.

**Verification rule:** `verified: true` only when a primary / manufacturer /
SAAMI-class public source is cited with the **exact** number used. Otherwise
`verified: false`.

## Schema (`version = 1`)

| Field | SI unit | Meaning |
| --- | --- | --- |
| `name` | — | Display / lookup id |
| `caliber_m` | m | Bullet diameter |
| `mass_kg` | kg | Bullet mass |
| `muzzle_velocity_m_s` | m/s | Reference muzzle speed |
| `bc.value_si` | kg/m² | Sectional BC = mass / reference area (= imperial lb/in² × 703.069580719 × 4/π ≈ 894.665) |
| `bc.drag_model` | enum | `G1`, `G7`, `CustomTable`, `Analytic` (legacy `G1StandIn`/`G7StandIn` → `G1`/`G7`) |
| `bc.table` | optional | Custom `(mach[], cd[])` knots |
| `length_m` | m | Bullet length (Magnus / form estimates) |
| `twist_m_per_rev` | m/rev | Barrel twist; spin = `v / twist` |
| `form_factor` | — | Multiplier on `C_D` |
| `barrel_length_ref_m` | m? | Optional `v ∝ √(L/L_ref)` scaling |
| `velocity_std_m_s` | m/s | Per-shot muzzle speed σ |
| `dispersion_std_rad` | rad | Aim cone 1σ |
| `provenance` | — | `{ note, verified }` |

## Drag models

- **G1 / G7** — published BRL/JBM standard Cd–Mach tables; **linear**
  interpolation; classical BC scaling. See [`BALLISTICS.md`](BALLISTICS.md) and
  [`THIRD_PARTY.md`](../THIRD_PARTY.md).
- **Analytic** — built-in engineering `C_D(M)` (demo only).
- **CustomTable** — caller-supplied Mach→`C_D` knots (linear / monotone APIs).

## Bundled profiles

### Generic class demos (`verified: false`)

`generic_pistol`, `generic_smg`, `generic_intermediate_rifle`,
`generic_battle_rifle`, `generic_sniper_long_range`,
`generic_heavy_machine_gun`, `generic_shotgun_slug`, `generic_subsonic`.

### Sourced cartridges (retrieval date 2026-11-01)

| Name | Mass / MV / Ø | BC | Profile `verified` | BC `verified` |
| --- | --- | --- | --- | --- |
| `9x19_124gr_fmj_federal_ae9ap` | 124 gr / 1150 fps / 0.355 in (Federal AE9AP) | G1 0.15 nominal | **true** | false |
| `5.56_55gr_fmjbt_federal_ae193` | 55 gr / 3165 fps / 0.224 in (Federal AE193) | G1 **0.246** (Federal) | **true** | **true** |
| `5.56_62gr_m855` | 62 gr / ~3025 fps | G1 0.304 nominal | false | false |
| `7.62x51_m80_147gr` | 147 gr / 2750 fps @ 78 ft (MAS Handbook 2021) | G1 0.401 nominal | **true** | false |
| `308_168gr_sierra_matchking` | 168 gr / 2650 fps (Federal GMM class) | G1 **0.462** (Sierra / DTIC ADA554683) | **true** | **true** |
| `300_winmag_180gr` | 180 gr / 2960 fps class | G1 0.480 nominal | false | false |
| `50bmg_750gr` | 750 gr / ~2820 fps class | G1 1.050 (Hornady A-MAX class; not locked) | false | false |
| `12ga_1oz_slug` | 1 oz / 1600 fps class | G1 0.070 nominal | false | false |
| `45acp_230gr_fmj` | 230 gr / 890 fps (Federal AE class) | G1 0.160 nominal | **true** | false |
| `oracle_g1_bc0.5_2800fps` | synthetic oracle | G1 **0.500** | **true** | **true** |

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
    "value_si": 357.8662322877,
    "drag_model": "G1",
    "table": null,
    "provenance": {
      "note": "G1 BC 0.400 lb/in² converted to SI; cite manufacturer here",
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
    "note": "studio tuning",
    "verified": false
  }
}
```

Rust: `AmmoProfile::from_json` / `from_toml`. Python: `ammo_from_json_muzzle_velocity`
and `ammo_profile_json` for bundled names.
