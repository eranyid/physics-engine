//! Versioned ammunition profiles for per-bullet simulation.
//!
//! Profiles carry **external-ballistics** parameters only (mass, muzzle velocity,
//! drag reference, spin kinematics). No design or manufacturing details.
//! Bundled class profiles use publicly published **nominal** figures and are
//! flagged `verified: false` unless a primary source was fetched and checked.

use pe_math::libm_ops;

use crate::drag::{analytic_cd, area_from_bc_si, caliber_area, DragModelRef, DragTable};

/// Schema version for serde ammo files.
pub const AMMO_PROFILE_VERSION: u32 = 1;

/// Provenance / verification metadata for a numeric field.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FieldProvenance {
    /// Human-readable note, e.g. `"nominal, approximate, source: …"`.
    pub note: String,
    /// `true` only when checked against a fetched primary / manufacturer source.
    pub verified: bool,
}

impl FieldProvenance {
    /// Unverified nominal figure.
    pub fn nominal(source: &str) -> Self {
        Self {
            note: format!("nominal, approximate, source: {source}"),
            verified: false,
        }
    }
}

/// Ballistic coefficient with its drag-model reference.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BallisticCoefficient {
    /// BC in SI sectional form [kg/m²] = mass / reference area.  
    /// **Not** the traditional lb/in² G1 BC unless converted by the caller.
    pub value_si: f64,
    /// Drag model tag.
    pub drag_model: DragModelRef,
    /// Optional custom table (used when `drag_model == CustomTable`).
    pub table: Option<DragTable>,
    /// Provenance.
    pub provenance: FieldProvenance,
}

/// Versioned ammo profile (serde JSON/TOML).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AmmoProfile {
    /// Schema version.
    pub version: u32,
    /// Display name.
    pub name: String,
    /// Caliber / bullet diameter [m].
    pub caliber_m: f64,
    /// Bullet mass [kg].
    pub mass_kg: f64,
    /// Muzzle velocity [m/s].
    pub muzzle_velocity_m_s: f64,
    /// Ballistic coefficient + drag model.
    pub bc: BallisticCoefficient,
    /// Bullet length [m] (for Magnus / form estimates).
    pub length_m: f64,
    /// Barrel twist as metres of travel per revolution [m/rev] (right-hand > 0).
    pub twist_m_per_rev: f64,
    /// Form / shape factor [dimensionless] multiplying `C_D` (1 = baseline).
    pub form_factor: f64,
    /// Optional barrel-length velocity scaling: `v = v0 · √(L / L_ref)` style
    /// reference length [m]; `None` disables.
    pub barrel_length_ref_m: Option<f64>,
    /// Muzzle-velocity standard deviation [m/s] for per-shot dispersion.
    pub velocity_std_m_s: f64,
    /// Angular dispersion std-dev [rad] (half-angle of cone, 1σ).
    pub dispersion_std_rad: f64,
    /// Field provenance bundle (summary).
    pub provenance: FieldProvenance,
}

impl AmmoProfile {
    /// Spin rate [rev/s] from muzzle speed and twist: `n = v / twist`.
    pub fn spin_revs_per_s(&self, muzzle_speed_m_s: f64) -> f64 {
        if self.twist_m_per_rev.abs() < 1e-12 {
            return 0.0;
        }
        muzzle_speed_m_s / self.twist_m_per_rev.abs()
    }

    /// Spin rate [rad/s].
    pub fn spin_rad_per_s(&self, muzzle_speed_m_s: f64) -> f64 {
        self.spin_revs_per_s(muzzle_speed_m_s) * 2.0 * core::f64::consts::PI
    }

    /// Reference area [m²] from BC (SI) or caliber disk if BC invalid.
    pub fn reference_area_m2(&self) -> f64 {
        let a = area_from_bc_si(self.mass_kg, self.bc.value_si);
        if a > 0.0 {
            a
        } else {
            caliber_area(self.caliber_m)
        }
    }

    /// `C_D` at Mach `m` using the profile's drag model × form factor.
    pub fn cd_at_mach(&self, mach: f64) -> f64 {
        let base = match self.bc.drag_model {
            DragModelRef::CustomTable => self
                .bc
                .table
                .as_ref()
                .map(|t| t.cd_linear(mach))
                .unwrap_or_else(|| analytic_cd(mach)),
            DragModelRef::Analytic | DragModelRef::G1StandIn | DragModelRef::G7StandIn => {
                // G7 stand-in: slightly lower CD than G1 stand-in.
                let scale = match self.bc.drag_model {
                    DragModelRef::G7StandIn => 0.85,
                    _ => 1.0,
                };
                analytic_cd(mach) * scale
            }
        };
        base * self.form_factor
    }

    /// Scale muzzle velocity for barrel length `l_m` [m] if a reference is set.
    pub fn muzzle_velocity_for_barrel(&self, barrel_length_m: f64) -> f64 {
        match self.barrel_length_ref_m {
            Some(lref) if lref > 0.0 && barrel_length_m > 0.0 => {
                self.muzzle_velocity_m_s * libm_ops::sqrt(barrel_length_m / lref)
            }
            _ => self.muzzle_velocity_m_s,
        }
    }

    /// Muzzle kinetic energy [J]: `½ m v²`.
    pub fn muzzle_energy_j(&self) -> f64 {
        0.5 * self.mass_kg * self.muzzle_velocity_m_s * self.muzzle_velocity_m_s
    }

    /// Parse JSON.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }

    /// Parse TOML.
    pub fn from_toml(s: &str) -> Result<Self, String> {
        toml::from_str(s).map_err(|e| e.to_string())
    }

    /// Serialize JSON.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }
}

fn profile(
    name: &str,
    caliber_mm: f64,
    mass_g: f64,
    v_mps: f64,
    bc_si: f64,
    drag: DragModelRef,
    length_mm: f64,
    twist_in_per_rev: f64,
    source: &str,
) -> AmmoProfile {
    // twist: inches/rev → m/rev
    let twist_m = twist_in_per_rev * 0.0254;
    AmmoProfile {
        version: AMMO_PROFILE_VERSION,
        name: name.to_string(),
        caliber_m: caliber_mm * 1e-3,
        mass_kg: mass_g * 1e-3,
        muzzle_velocity_m_s: v_mps,
        bc: BallisticCoefficient {
            value_si: bc_si,
            drag_model: drag,
            table: None,
            provenance: FieldProvenance::nominal(source),
        },
        length_m: length_mm * 1e-3,
        twist_m_per_rev: twist_m,
        form_factor: 1.0,
        barrel_length_ref_m: None,
        velocity_std_m_s: 0.02 * v_mps, // 2 % nominal
        dispersion_std_rad: 0.001,      // ~1 mrad
        provenance: FieldProvenance::nominal(source),
    }
}

/// Bundled generic ammo **class** profiles (nominal, `verified: false`).
///
/// Figures are rounded public-domain / encyclopedia-class ballpark values for
/// game and studio simulation — **not** manufacturer load data.
pub fn bundled_profiles() -> Vec<AmmoProfile> {
    vec![
        profile(
            "generic_pistol",
            9.0,
            8.0,
            350.0,
            2000.0,
            DragModelRef::G1StandIn,
            15.0,
            10.0,
            "public nominal class figures (pistol); verified: false",
        ),
        profile(
            "generic_smg",
            9.0,
            8.0,
            400.0,
            2200.0,
            DragModelRef::G1StandIn,
            15.0,
            10.0,
            "public nominal class figures (SMG); verified: false",
        ),
        profile(
            "generic_intermediate_rifle",
            5.56,
            4.0,
            900.0,
            3500.0,
            DragModelRef::G7StandIn,
            20.0,
            7.0,
            "public nominal class figures (intermediate rifle); verified: false",
        ),
        profile(
            "generic_battle_rifle",
            7.62,
            9.5,
            800.0,
            4000.0,
            DragModelRef::G7StandIn,
            30.0,
            12.0,
            "public nominal class figures (battle rifle); verified: false",
        ),
        profile(
            "generic_sniper_long_range",
            8.6,
            12.0,
            850.0,
            5500.0,
            DragModelRef::G7StandIn,
            40.0,
            10.0,
            "public nominal class figures (long-range rifle); verified: false",
        ),
        profile(
            "generic_heavy_machine_gun",
            12.7,
            45.0,
            880.0,
            6000.0,
            DragModelRef::G1StandIn,
            60.0,
            15.0,
            "public nominal class figures (HMG); verified: false",
        ),
        profile(
            "generic_shotgun_slug",
            18.4,
            28.0,
            450.0,
            1500.0,
            DragModelRef::G1StandIn,
            30.0,
            0.0, // often unrifled; spin 0
            "public nominal class figures (slug); verified: false",
        ),
        profile(
            "generic_subsonic",
            9.0,
            10.0,
            300.0,
            2500.0,
            DragModelRef::Analytic,
            20.0,
            10.0,
            "public nominal class figures (subsonic); verified: false",
        ),
    ]
}

/// Look up a bundled profile by exact name.
pub fn bundled_by_name(name: &str) -> Option<AmmoProfile> {
    bundled_profiles().into_iter().find(|p| p.name == name)
}
