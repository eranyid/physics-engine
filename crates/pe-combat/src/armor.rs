//! Armor / medium materials and layered armor stacks.
//!
//! Two penetration model classes:
//!
//! - [`PenetrationModel::Plate`] — thin, hard plates (steel, aluminium,
//!   ceramic-faced): perforation by **Lambert–Jonas** ballistic limit with
//!   obliquity, residual speed by **Recht–Ipson / Lambert** form.
//! - [`PenetrationModel::Medium`] — thick, soft media (soil, wood, concrete,
//!   water, tissue simulant): **Poncelet** retardation integrated along the
//!   actual path length through the shape.
//!
//! Bundled `generic_*` rows are nominal (`provenance.verified == false`);
//! they are tuned to give plausible game behaviour, not design data.

use pe_ballistics::ammo::FieldProvenance;

/// Schema version for serde material files.
pub const ARMOR_MATERIAL_VERSION: u32 = 1;

/// Penetration model for a material.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PenetrationModel {
    /// Thin hard plate (Lambert–Jonas ballistic limit).
    Plate {
        /// Lambert constant `k` in SI form [m/s · (kg/m³)^½].
        ///
        /// `V50 = k (L/d)^0.15 √(f(z) d³ / m)`, `f(z) = z + e^{−z} − 1`.
        /// The classic cgs constant 4000 (d in cm, m in g) is `≈ 126 491` here.
        lambert_k: f64,
        /// Obliquity exponent `n` in `z = (t/d) sec^n θ` (Lambert: 0.75).
        obliquity_exponent: f64,
        /// Residual-velocity scale `a` in `v_r = a (v^p − v_bl^p)^{1/p}`.
        residual_a: f64,
        /// Residual-velocity exponent `p`.
        residual_p: f64,
        /// Fractional projectile mass loss per unit `z` on perforation.
        erosion_per_z: f64,
    },
    /// Thick / soft medium (Poncelet `m dv/dx = −A (a + b v²)`).
    Medium {
        /// Strength-like term `a` [Pa].
        poncelet_a_pa: f64,
        /// Inertia-like term `b` [kg/m³].
        poncelet_b_kg_m3: f64,
    },
}

/// Ricochet parameters (graze angle measured from the surface plane).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RicochetParams {
    /// Critical graze angle `β_c` [rad]; below it ricochet is likely.
    /// `0` disables ricochet.
    pub critical_graze_rad: f64,
    /// Logistic transition width [rad] around `β_c` (`0` = sharp step).
    pub transition_width_rad: f64,
    /// Normal restitution `e_n ∈ [0,1]` (rebound normal speed fraction).
    pub normal_restitution: f64,
    /// Tangential retention `e_t ∈ [0,1]` (kept tangential speed fraction).
    pub tangential_retention: f64,
    /// Std-dev of random direction jitter after ricochet [rad].
    pub deflection_std_rad: f64,
}

impl RicochetParams {
    /// No ricochet.
    pub fn none() -> Self {
        Self {
            critical_graze_rad: 0.0,
            transition_width_rad: 0.0,
            normal_restitution: 0.0,
            tangential_retention: 0.0,
            deflection_std_rad: 0.0,
        }
    }
}

/// Spall (behind-plate debris) parameters.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpallParams {
    /// Fraction of the energy dissipated in the layer that goes into spall.
    /// `0` disables spall.
    pub energy_fraction: f64,
    /// Total spall mass as a multiple of the plug mass `ρ_t π (d/2)² t`.
    pub mass_factor: f64,
    /// Half-angle of the spall cone around the exit direction [rad].
    pub cone_half_angle_rad: f64,
    /// Mott characteristic fragment mass `μ` [kg].
    pub mott_mu_kg: f64,
    /// Maximum fragments per perforation.
    pub max_fragments: usize,
}

impl SpallParams {
    /// No spall.
    pub fn none() -> Self {
        Self {
            energy_fraction: 0.0,
            mass_factor: 0.0,
            cone_half_angle_rad: 0.0,
            mott_mu_kg: 0.0,
            max_fragments: 0,
        }
    }
}

/// Armor or medium material.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ArmorMaterial {
    /// Schema version ([`ARMOR_MATERIAL_VERSION`]).
    pub version: u32,
    /// Unique name.
    pub name: String,
    /// Density [kg/m³].
    pub density_kg_m3: f64,
    /// Penetration model.
    pub model: PenetrationModel,
    /// Ricochet behaviour.
    pub ricochet: RicochetParams,
    /// Spall behaviour.
    pub spall: SpallParams,
    /// Drag multiplier gain after perforation: `k_drag ← k_drag (1 + g · ΔE/E)`.
    pub tumble_drag_gain: f64,
    /// Std-dev of exit-direction deflection [rad] per unit energy-loss fraction.
    pub exit_deflection_std_rad: f64,
    /// Provenance of the constants.
    pub provenance: FieldProvenance,
}

impl ArmorMaterial {
    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self, String> {
        let m: Self = serde_json::from_str(s).map_err(|e| e.to_string())?;
        m.validate()?;
        Ok(m)
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("ArmorMaterial serializes")
    }

    /// Check physical sanity of all fields.
    pub fn validate(&self) -> Result<(), String> {
        if self.version != ARMOR_MATERIAL_VERSION {
            return Err(format!("unsupported material version {}", self.version));
        }
        if !(self.density_kg_m3 > 0.0) {
            return Err("density must be > 0".into());
        }
        match &self.model {
            PenetrationModel::Plate {
                lambert_k,
                obliquity_exponent,
                residual_a,
                residual_p,
                erosion_per_z,
            } => {
                if !(*lambert_k > 0.0 && *obliquity_exponent >= 0.0) {
                    return Err("plate: lambert_k > 0, obliquity_exponent ≥ 0".into());
                }
                if !(*residual_a > 0.0 && *residual_a <= 1.0 && *residual_p > 0.0) {
                    return Err("plate: 0 < residual_a ≤ 1, residual_p > 0".into());
                }
                if !(*erosion_per_z >= 0.0) {
                    return Err("plate: erosion_per_z ≥ 0".into());
                }
            }
            PenetrationModel::Medium {
                poncelet_a_pa,
                poncelet_b_kg_m3,
            } => {
                if !(*poncelet_a_pa > 0.0 && *poncelet_b_kg_m3 > 0.0) {
                    return Err("medium: poncelet a, b > 0".into());
                }
            }
        }
        let r = &self.ricochet;
        let unit = |x: f64| (0.0..=1.0).contains(&x);
        if !(r.critical_graze_rad >= 0.0 && r.transition_width_rad >= 0.0)
            || !unit(r.normal_restitution)
            || !unit(r.tangential_retention)
            || !(r.deflection_std_rad >= 0.0)
        {
            return Err("ricochet params out of range".into());
        }
        let s = &self.spall;
        if !unit(s.energy_fraction) || !(s.mass_factor >= 0.0) || !(s.mott_mu_kg >= 0.0) {
            return Err("spall params out of range".into());
        }
        if s.energy_fraction > 0.0 && !(s.mott_mu_kg > 0.0 && s.max_fragments > 0) {
            return Err("spall enabled requires mott_mu_kg > 0 and max_fragments > 0".into());
        }
        if !(self.tumble_drag_gain >= 0.0 && self.exit_deflection_std_rad >= 0.0) {
            return Err("tumble_drag_gain, exit_deflection_std_rad ≥ 0".into());
        }
        Ok(())
    }
}

/// One layer of an armor stack. `material` indexes the resolver's material table.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ArmorLayer {
    /// Index into the material table.
    pub material: usize,
    /// Normal (plate) thickness [m]. `≤ 0` means "use the shape's geometric
    /// path length" (plates: chord·cos θ; media: chord).
    pub thickness_m: f64,
}

/// Ordered layers, outermost first (e.g. ceramic face → steel backing).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ArmorStack {
    /// Layers, outermost first.
    pub layers: Vec<ArmorLayer>,
}

impl ArmorStack {
    /// Single-layer stack.
    pub fn single(material: usize, thickness_m: f64) -> Self {
        Self {
            layers: vec![ArmorLayer {
                material,
                thickness_m,
            }],
        }
    }

    /// Single layer whose thickness comes from the hit shape's geometry.
    pub fn solid(material: usize) -> Self {
        Self::single(material, 0.0)
    }
}

fn plate(
    name: &str,
    density: f64,
    lambert_k: f64,
    residual_a: f64,
    residual_p: f64,
    erosion_per_z: f64,
    ricochet: RicochetParams,
    spall: SpallParams,
    tumble: f64,
    defl: f64,
) -> ArmorMaterial {
    ArmorMaterial {
        version: ARMOR_MATERIAL_VERSION,
        name: name.to_string(),
        density_kg_m3: density,
        model: PenetrationModel::Plate {
            lambert_k,
            obliquity_exponent: 0.75,
            residual_a,
            residual_p,
            erosion_per_z,
        },
        ricochet,
        spall,
        tumble_drag_gain: tumble,
        exit_deflection_std_rad: defl,
        provenance: FieldProvenance::nominal(
            "Lambert–Jonas form (Lambert 1978, BRL-R-2036); generic game-tuned constants",
        ),
    }
}

fn medium(
    name: &str,
    density: f64,
    a: f64,
    b: f64,
    ricochet: RicochetParams,
    spall: SpallParams,
    tumble: f64,
    defl: f64,
) -> ArmorMaterial {
    ArmorMaterial {
        version: ARMOR_MATERIAL_VERSION,
        name: name.to_string(),
        density_kg_m3: density,
        model: PenetrationModel::Medium {
            poncelet_a_pa: a,
            poncelet_b_kg_m3: b,
        },
        ricochet,
        spall,
        tumble_drag_gain: tumble,
        exit_deflection_std_rad: defl,
        provenance: FieldProvenance::nominal(
            "Poncelet form; generic game-tuned constants (plausible depths, not test data)",
        ),
    }
}

fn rico(graze_deg: f64, width_deg: f64, e_n: f64, e_t: f64, std_deg: f64) -> RicochetParams {
    let d = core::f64::consts::PI / 180.0;
    RicochetParams {
        critical_graze_rad: graze_deg * d,
        transition_width_rad: width_deg * d,
        normal_restitution: e_n,
        tangential_retention: e_t,
        deflection_std_rad: std_deg * d,
    }
}

fn spall(energy_fraction: f64, mass_factor: f64, cone_deg: f64, mu_kg: f64) -> SpallParams {
    SpallParams {
        energy_fraction,
        mass_factor,
        cone_half_angle_rad: cone_deg * core::f64::consts::PI / 180.0,
        mott_mu_kg: mu_kg,
        max_fragments: 24,
    }
}

/// Bundled nominal materials (all `verified: false`).
///
/// | name | model |
/// | --- | --- |
/// | `generic_hard_steel_plate` | plate |
/// | `generic_mild_steel_plate` | plate |
/// | `generic_aluminium_plate` | plate |
/// | `generic_concrete` | medium |
/// | `generic_wood` | medium |
/// | `generic_soil` | medium |
/// | `generic_water` | medium |
/// | `generic_tissue_simulant` | medium |
pub fn bundled_materials() -> Vec<ArmorMaterial> {
    vec![
        plate(
            "generic_hard_steel_plate",
            7850.0,
            126_491.0,
            0.9,
            2.5,
            0.05,
            rico(20.0, 3.0, 0.15, 0.8, 2.0),
            spall(0.15, 0.5, 35.0, 2.0e-5),
            4.0,
            0.05,
        ),
        plate(
            "generic_mild_steel_plate",
            7850.0,
            95_000.0,
            0.9,
            2.5,
            0.05,
            rico(15.0, 3.0, 0.15, 0.75, 2.0),
            spall(0.10, 0.5, 40.0, 2.0e-5),
            4.0,
            0.05,
        ),
        plate(
            "generic_aluminium_plate",
            2700.0,
            60_000.0,
            0.95,
            2.0,
            0.02,
            rico(12.0, 3.0, 0.1, 0.75, 2.0),
            spall(0.05, 0.3, 45.0, 1.0e-5),
            3.0,
            0.04,
        ),
        medium(
            "generic_concrete",
            2400.0,
            1.5e8,
            2000.0,
            rico(17.0, 3.0, 0.1, 0.6, 4.0),
            spall(0.10, 0.4, 50.0, 5.0e-5),
            6.0,
            0.08,
        ),
        medium(
            "generic_wood",
            550.0,
            1.5e7,
            600.0,
            rico(6.0, 2.0, 0.05, 0.6, 4.0),
            SpallParams::none(),
            5.0,
            0.10,
        ),
        medium(
            "generic_soil",
            1600.0,
            2.0e7,
            1200.0,
            rico(9.0, 2.0, 0.05, 0.55, 5.0),
            SpallParams::none(),
            8.0,
            0.12,
        ),
        medium(
            "generic_water",
            1000.0,
            1.0e5,
            150.0,
            rico(7.0, 1.5, 0.05, 0.7, 3.0),
            SpallParams::none(),
            10.0,
            0.10,
        ),
        medium(
            "generic_tissue_simulant",
            1060.0,
            5.0e6,
            424.0,
            RicochetParams::none(),
            SpallParams::none(),
            8.0,
            0.15,
        ),
    ]
}

/// Look up a bundled material by name.
pub fn bundled_material_by_name(name: &str) -> Option<ArmorMaterial> {
    bundled_materials().into_iter().find(|m| m.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_validate_and_unverified() {
        for m in bundled_materials() {
            m.validate().unwrap_or_else(|e| panic!("{}: {e}", m.name));
            assert!(!m.provenance.verified, "{} must be nominal", m.name);
        }
    }

    /// Structural equality with 1e-15 relative float tolerance (serde_json's
    /// default float parser may be off by one ulp).
    fn approx_eq(a: &serde_json::Value, b: &serde_json::Value) -> bool {
        use serde_json::Value as V;
        match (a, b) {
            (V::Number(x), V::Number(y)) => {
                let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
                (x - y).abs() <= 1e-15 * x.abs().max(y.abs())
            }
            (V::Array(x), V::Array(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| approx_eq(p, q))
            }
            (V::Object(x), V::Object(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .all(|(k, v)| y.get(k).map(|w| approx_eq(v, w)).unwrap_or(false))
            }
            _ => a == b,
        }
    }

    #[test]
    fn json_roundtrip() {
        for m in bundled_materials() {
            let back = ArmorMaterial::from_json(&m.to_json()).unwrap();
            assert_eq!(back.name, m.name);
            let va = serde_json::to_value(&m).unwrap();
            let vb = serde_json::to_value(&back).unwrap();
            assert!(approx_eq(&va, &vb), "{}", m.name);
        }
    }

    #[test]
    fn invalid_rejected() {
        let mut m = bundled_material_by_name("generic_wood").unwrap();
        m.density_kg_m3 = -1.0;
        assert!(m.validate().is_err());
    }
}
