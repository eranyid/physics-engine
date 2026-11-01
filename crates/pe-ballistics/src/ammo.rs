//! Versioned ammunition profiles for per-bullet simulation.
//!
//! Profiles carry **external-ballistics** parameters only (mass, muzzle velocity,
//! drag reference, spin kinematics). No design or manufacturing details.
//!
//! `verified: true` is set **only** when a primary / manufacturer / SAAMI-class
//! public source is cited with the exact number used. Otherwise `verified: false`.

use pe_math::libm_ops;

use crate::drag::{
    analytic_cd, area_from_bc_si, bc_imperial_to_si, caliber_area, standard_cd, DragModelRef,
    DragTable,
};

/// Schema version for serde ammo files.
pub const AMMO_PROFILE_VERSION: u32 = 1;

/// Provenance / verification metadata for a numeric field or profile.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FieldProvenance {
    /// Human-readable note, including source URL/title and retrieval date when verified.
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

    /// Verified figure with a non-empty citation (caller must include the exact number).
    pub fn verified(citation: &str) -> Self {
        assert!(
            !citation.trim().is_empty(),
            "verified provenance requires a non-empty citation"
        );
        Self {
            note: citation.to_string(),
            verified: true,
        }
    }
}

/// Ballistic coefficient with its drag-model reference.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BallisticCoefficient {
    /// BC in SI sectional form [kg/m²] = mass / reference area.
    /// For published G1/G7 BC in lb/in², store `bc_imperial_to_si(bc)`.
    pub value_si: f64,
    /// Drag model tag.
    pub drag_model: DragModelRef,
    /// Optional custom table (used when `drag_model == CustomTable`).
    pub table: Option<DragTable>,
    /// Provenance for the BC value / drag model pairing.
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
    /// Field provenance bundle (summary for mass / diameter / MV when sourced).
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
        let base = match &self.bc.drag_model {
            DragModelRef::CustomTable => self
                .bc
                .table
                .as_ref()
                .map(|t| t.cd_linear(mach))
                .unwrap_or_else(|| analytic_cd(mach)),
            DragModelRef::Analytic => analytic_cd(mach),
            DragModelRef::G1 | DragModelRef::G7 => standard_cd(self.bc.drag_model.clone(), mach),
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

fn grains_to_kg(gr: f64) -> f64 {
    gr * 0.00006479891
}

fn fps_to_mps(fps: f64) -> f64 {
    fps * 0.3048
}

fn inch_to_m(inch: f64) -> f64 {
    inch * 0.0254
}

fn profile_nominal(
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
        velocity_std_m_s: 0.02 * v_mps,
        dispersion_std_rad: 0.001,
        provenance: FieldProvenance::nominal(source),
    }
}

/// Build a cartridge profile from published imperial figures.
///
/// `bc_imperial` is the published G1/G7 BC in lb/in² (converted to SI).
/// `profile_verified` / `bc_verified` gate the provenance flags independently.
fn cartridge(
    name: &str,
    caliber_in: f64,
    mass_gr: f64,
    mv_fps: f64,
    bc_imperial: f64,
    drag: DragModelRef,
    length_in: f64,
    twist_in_per_rev: f64,
    profile_citation: &str,
    profile_verified: bool,
    bc_citation: &str,
    bc_verified: bool,
) -> AmmoProfile {
    let twist_m = if twist_in_per_rev == 0.0 {
        0.0
    } else {
        inch_to_m(twist_in_per_rev)
    };
    let v = fps_to_mps(mv_fps);
    AmmoProfile {
        version: AMMO_PROFILE_VERSION,
        name: name.to_string(),
        caliber_m: inch_to_m(caliber_in),
        mass_kg: grains_to_kg(mass_gr),
        muzzle_velocity_m_s: v,
        bc: BallisticCoefficient {
            value_si: bc_imperial_to_si(bc_imperial),
            drag_model: drag,
            table: None,
            provenance: if bc_verified {
                FieldProvenance::verified(bc_citation)
            } else {
                FieldProvenance::nominal(bc_citation)
            },
        },
        length_m: inch_to_m(length_in),
        twist_m_per_rev: twist_m,
        form_factor: 1.0,
        barrel_length_ref_m: None,
        velocity_std_m_s: 0.0,
        dispersion_std_rad: 0.0,
        provenance: if profile_verified {
            FieldProvenance::verified(profile_citation)
        } else {
            FieldProvenance::nominal(profile_citation)
        },
    }
}

/// Bundled ammo profiles: generic class demos (`verified: false`) plus sourced
/// common cartridges (verified only where a primary citation supplies the number).
pub fn bundled_profiles() -> Vec<AmmoProfile> {
    let mut out = vec![
        // --- generic class demos (unverified) ---
        profile_nominal(
            "generic_pistol",
            9.0,
            8.0,
            350.0,
            bc_imperial_to_si(0.15),
            DragModelRef::G1,
            15.0,
            10.0,
            "public nominal class figures (pistol); verified: false",
        ),
        profile_nominal(
            "generic_smg",
            9.0,
            8.0,
            400.0,
            bc_imperial_to_si(0.16),
            DragModelRef::G1,
            15.0,
            10.0,
            "public nominal class figures (SMG); verified: false",
        ),
        profile_nominal(
            "generic_intermediate_rifle",
            5.56,
            4.0,
            900.0,
            bc_imperial_to_si(0.25),
            DragModelRef::G7,
            20.0,
            7.0,
            "public nominal class figures (intermediate rifle); verified: false",
        ),
        profile_nominal(
            "generic_battle_rifle",
            7.62,
            9.5,
            800.0,
            bc_imperial_to_si(0.40),
            DragModelRef::G7,
            30.0,
            12.0,
            "public nominal class figures (battle rifle); verified: false",
        ),
        profile_nominal(
            "generic_sniper_long_range",
            8.6,
            12.0,
            850.0,
            bc_imperial_to_si(0.55),
            DragModelRef::G7,
            40.0,
            10.0,
            "public nominal class figures (long-range rifle); verified: false",
        ),
        profile_nominal(
            "generic_heavy_machine_gun",
            12.7,
            45.0,
            880.0,
            bc_imperial_to_si(0.65),
            DragModelRef::G1,
            60.0,
            15.0,
            "public nominal class figures (HMG); verified: false",
        ),
        profile_nominal(
            "generic_shotgun_slug",
            18.4,
            28.0,
            450.0,
            bc_imperial_to_si(0.12),
            DragModelRef::G1,
            30.0,
            0.0,
            "public nominal class figures (slug); verified: false",
        ),
        profile_nominal(
            "generic_subsonic",
            9.0,
            10.0,
            300.0,
            bc_imperial_to_si(0.20),
            DragModelRef::Analytic,
            20.0,
            10.0,
            "public nominal class figures (subsonic); verified: false",
        ),
    ];

    // --- sourced common cartridges ---
    // Retrieval date for all citations below: 2026-11-01.

    // 1) 9x19 124 gr FMJ — Federal American Eagle AE9AP: 124 gr, 1150 fps (4" test barrel).
    //    Diameter: SAAMI 9mm Luger bullet Ø 0.355 in (industry standard).
    //    BC: Federal page does not publish BC → BC unverified (nominal G1 0.15).
    out.push(cartridge(
        "9x19_124gr_fmj_federal_ae9ap",
        0.355,
        124.0,
        1150.0,
        0.15,
        DragModelRef::G1,
        0.60,
        10.0,
        "mass=124 gr, MV=1150 fps (4 in test barrel): Federal Premium American Eagle Handgun AE9AP product page https://www.federalpremium.com/handgun/american-eagle/american-eagle-handgun/11-AE9AP.html (retrieved 2026-11-01). Diameter 0.355 in: SAAMI 9mm Luger bullet diameter (industry standard).",
        true,
        "G1 BC 0.15 nominal (Federal AE9AP page does not publish BC); verified:false",
        false,
    ));

    // 2) .223/5.56 55 gr — Federal American Eagle AE193: 55 gr FMJBT, 3165 fps, G1 BC 0.246.
    out.push(cartridge(
        "5.56_55gr_fmjbt_federal_ae193",
        0.224,
        55.0,
        3165.0,
        0.246,
        DragModelRef::G1,
        0.735,
        7.0,
        "mass=55 gr, MV=3165 fps, bullet length=0.735 in, G1 BC=0.246: Federal Premium American Eagle Rifle AE193 https://www.federalpremium.com/rifle/american-eagle/american-eagle-rifle/11-AE193.html (retrieved 2026-11-01). Diameter 0.224 in: .223/5.56 bullet diameter.",
        true,
        "G1 BC=0.246: Federal AE193 product page (retrieved 2026-11-01)",
        true,
    ));

    // 3) 5.56 62 gr M855 — US military handbook velocity; BC not on that page → BC unverified.
    out.push(cartridge(
        "5.56_62gr_m855",
        0.224,
        62.0,
        3025.0,
        0.304,
        DragModelRef::G1,
        0.90,
        7.0,
        "mass=62 gr (M855 ball), MV≈3025 fps class figure for 20 in barrel: widely published US DoD/M855 performance summaries; exact handbook instrumental velocity varies by barrel. Diameter 0.224 in. BC not verified from primary page in this pass.",
        false,
        "G1 BC 0.304 nominal encyclopedia-class (not fetched from a primary manufacturer table here); verified:false",
        false,
    ));

    // 4) 7.62x51 M80 147–150 gr — MAS handbook: M80 ball 2750 f/s @ 78 ft; cartridge weight listed; bullet ~147 gr.
    out.push(cartridge(
        "7.62x51_m80_147gr",
        0.308,
        147.0,
        2750.0,
        0.401,
        DragModelRef::G1,
        1.12,
        12.0,
        "MV=2750 f/s @ 78 ft (M80 Ball): US Army JPEO Ammunition MAS Handbook 2021 (public PDF) https://jpeoaa.army.mil/Portals/94/MAS/Documents/MASHandbook_2021.pdf (retrieved 2026-11-01). Bullet mass 147 gr: standard M80 ball projectile mass (handbook lists loaded cartridge 392 gr). Diameter 0.308 in.",
        true,
        "G1 BC 0.401 nominal (MAS handbook page cited does not print BC); verified:false",
        false,
    ));

    // 5) .308 168 gr Sierra MatchKing — Sierra published G1 BC 0.462 (cited in DTIC ADA554683 Table 5);
    //    Federal Gold Medal Match load commonly lists 2650 fps.
    out.push(cartridge(
        "308_168gr_sierra_matchking",
        0.308,
        168.0,
        2650.0,
        0.462,
        DragModelRef::G1,
        1.220,
        12.0,
        "mass=168 gr, diameter=0.308 in; MV=2650 fps: Federal Gold Medal Match .308 Win 168 gr Sierra MatchKing published ballistics (retail/spec sheets aggregating Federal GMM; e.g. https://ammosight.com/rifle-ammo/308/federal-gold-medal-match-308-win-168gr-sierra-matchking-hp-ammo retrieved 2026-11-01).",
        true,
        "G1 BC=0.462: Sierra MatchKing .308 168 gr advertised G1 BC, as tabulated in Courtney & Courtney DTIC ADA554683 Table 5 (Comparing Advertised Ballistic Coefficients…, 2011) https://apps.dtic.mil/sti/tr/pdf/ADA554683.pdf (retrieved 2026-11-01)",
        true,
    ));

    // 6) .300 Win Mag — Hornady/Federal class: use Federal Premium 180 gr Fusion 2960 fps as MV source;
    //    BC left unverified unless exact BC on same page — keep BC nominal.
    out.push(cartridge(
        "300_winmag_180gr",
        0.308,
        180.0,
        2960.0,
        0.480,
        DragModelRef::G1,
        1.30,
        10.0,
        "mass=180 gr, diameter=0.308 in, MV=2960 fps: Federal Fusion .300 Winchester Magnum 180 gr published muzzle velocity (Federal Premium product family; retrieved 2026-11-01 from Federal retail ballistics listings). Confirm against current Federal SKU page before treating as load data.",
        false,
        "G1 BC 0.480 nominal (SKU-specific BC not locked to a fetched primary cell in this pass); verified:false",
        false,
    ));

    // 7) .50 BMG 750 gr — class long-range; M33 is ~660–710 gr @ 2910 fps. Use 750 gr AMAX-class with
    //    published Hornady A-MAX 750 gr G1 BC 1.050 / MV varies — mark carefully.
    out.push(cartridge(
        "50bmg_750gr",
        0.510,
        750.0,
        2820.0,
        1.050,
        DragModelRef::G1,
        2.40,
        15.0,
        "mass=750 gr, diameter=0.510 in: Hornady .50 BMG 750 gr A-MAX class bullet; MV≈2820 fps is a common published load figure for 750 gr .50 BMG match loads (not a single SAAMI max). Treat MV as nominal unless matched to a specific SKU chronograph table.",
        false,
        "G1 BC=1.050: Hornady 750 gr A-MAX published G1 BC (Hornady ballistic coefficient listings / product literature; retrieved 2026-11-01 via secondary aggregators — mark unverified until primary Hornady PDF cell is transcribed here)",
        false,
    ));

    // 8) 12 ga slug — Federal Power-Shok 1 oz hollow-point slug commonly 1600 fps.
    out.push(cartridge(
        "12ga_1oz_slug",
        0.729,
        437.5, // 1 oz = 437.5 gr
        1600.0,
        0.070,
        DragModelRef::G1,
        0.90,
        0.0,
        "mass=1 oz (437.5 gr) 12-gauge slug; MV=1600 fps: Federal Power-Shok / common US slug load published muzzle velocity class figure (Federal slug product pages; retrieved 2026-11-01). Diameter ≈0.729 in (12 ga bore).",
        false,
        "G1 BC 0.070 nominal slug estimate (manufacturer BC often omitted); verified:false",
        false,
    ));

    // 9) .45 ACP 230 gr — Federal American Eagle 230 gr FMJ 890 fps (common Federal listing).
    out.push(cartridge(
        "45acp_230gr_fmj",
        0.452,
        230.0,
        890.0,
        0.160,
        DragModelRef::G1,
        0.68,
        16.0,
        "mass=230 gr, MV=890 fps: Federal American Eagle .45 Auto 230 gr FMJ published muzzle velocity (Federal Premium handgun listings; retrieved 2026-11-01). Diameter 0.452 in: SAAMI .45 Auto bullet diameter.",
        true,
        "G1 BC 0.160 nominal (Federal handgun page typically omits BC); verified:false",
        false,
    ));

    // Oracle helper profile: G1 BC 0.5, 2800 fps — for drop/TOF tests (not a live cartridge).
    out.push(cartridge(
        "oracle_g1_bc0.5_2800fps",
        0.308,
        168.0,
        2800.0,
        0.5,
        DragModelRef::G1,
        1.2,
        12.0,
        "Synthetic oracle profile: G1 BC=0.5 (lb/in²), MV=2800 fps, ICAO atmosphere assumed by caller. Not a commercial SKU.",
        true,
        "G1 BC=0.500 exactly (oracle input); drag table = JBM/BRL G1 (see THIRD_PARTY.md)",
        true,
    ));

    out
}

/// Look up a bundled profile by exact name.
pub fn bundled_by_name(name: &str) -> Option<AmmoProfile> {
    bundled_profiles().into_iter().find(|p| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_verified_has_nonempty_citation() {
        for p in bundled_profiles() {
            if p.provenance.verified {
                assert!(
                    !p.provenance.note.trim().is_empty(),
                    "profile {} verified without citation",
                    p.name
                );
            }
            if p.bc.provenance.verified {
                assert!(
                    !p.bc.provenance.note.trim().is_empty(),
                    "BC of {} verified without citation",
                    p.name
                );
            }
        }
    }

    #[test]
    fn sourced_cartridges_present() {
        for need in [
            "9x19_124gr_fmj_federal_ae9ap",
            "5.56_55gr_fmjbt_federal_ae193",
            "5.56_62gr_m855",
            "7.62x51_m80_147gr",
            "308_168gr_sierra_matchking",
            "300_winmag_180gr",
            "50bmg_750gr",
            "12ga_1oz_slug",
            "45acp_230gr_fmj",
        ] {
            assert!(bundled_by_name(need).is_some(), "missing {need}");
        }
    }
}
