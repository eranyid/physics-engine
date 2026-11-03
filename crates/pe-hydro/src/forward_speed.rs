//! Forward-speed seakeeping corrections (encounter frequency + STF strip terms).
//!
//! # Encounter frequency
//!
//! For a ship advancing at speed `U` into regular waves of absolute frequency
//! `ω` and wave number `k`, with heading angle `χ` (`χ = 0` head seas,
//! `χ = π` following):
//!
//! ```text
//! ω_e = ω − U k cos χ
//! ```
//!
//! (deep-water `k = ω²/g` when depth → ∞). Sign convention: positive `U`
//! forward; head-sea `cos χ = +1` raises encounter frequency.
//!
//! # Salvesen–Tuck–Faltinsen (1970) forward-speed strip corrections
//!
//! At forward speed the frequency-domain radiation operators gain
//! speed-dependent terms from the convective derivative along the strip
//! (STF Trans. SNAME 1970). For heave–pitch of a slender ship the leading
//! corrections used here (clean-room from the published strip integrals) are:
//!
//! ```text
//! Aᵢⱼ(U) ≈ Aᵢⱼ(0)          (added mass kept at encounter-ω strip value)
//! B₃₃(U) ≈ B₃₃(0) − U (a₃₃ˢ − a₃₃ᵗ)
//! B₃₅(U) ≈ B₃₅(0) − U (xˢ a₃₃ˢ − xᵗ a₃₃ᵗ) − U ∫ a₃₃ dx
//! B₅₃(U) ≈ B₅₃(0) + U (xˢ a₃₃ˢ − xᵗ a₃₃ᵗ) + U ∫ a₃₃ dx
//! B₅₅(U) ≈ B₅₅(0) − U (xˢ² a₃₃ˢ − xᵗ² a₃₃ᵗ)
//! ```
//!
//! with superscripts `s`/`t` = stern / stem end stations (after STF §4). The
//! zero-speed `A,B` come from [`crate::seakeeping::assemble_strip_matrices`]
//! evaluated at **encounter** frequency `ω_e`.
//!
//! These are engineering strip approximations — **not** a 3-D panel forward-speed
//! Green function. End-station contributions use the outermost strips.
//!
//! # References
//! - Salvesen, N., Tuck, E. O. & Faltinsen, O. (1970). Ship motions and sea
//!   loads. *Trans. SNAME* **78**.

use pe_math::libm_ops;

use crate::seakeeping::{assemble_strip_matrices, sectional_a33, StripMatrices, StripStation};
use crate::waves::solve_wave_number;

/// Encounter frequency [rad/s]: `ω_e = ω − U k cos χ`.
///
/// `heading_rad`: wave direction relative to ship heading (`0` = head seas).
pub fn encounter_frequency(omega: f64, speed_u: f64, k: f64, heading_rad: f64) -> f64 {
    omega - speed_u * k * libm_ops::cos(heading_rad)
}

/// Encounter frequency with dispersion-consistent `k(ω, depth)`.
pub fn encounter_frequency_dispersive(
    omega: f64,
    speed_u: f64,
    depth: f64,
    heading_rad: f64,
    g: f64,
) -> f64 {
    let k = solve_wave_number(omega, depth, g);
    encounter_frequency(omega, speed_u, k, heading_rad)
}

/// Deep-water shortcut: `k = ω²/g`, `ω_e = ω − U (ω²/g) cos χ`.
pub fn encounter_frequency_deep(omega: f64, speed_u: f64, heading_rad: f64, g: f64) -> f64 {
    if g <= 0.0 {
        return omega;
    }
    let k = omega * omega / g;
    encounter_frequency(omega, speed_u, k, heading_rad)
}

/// STF forward-speed heave–pitch radiation matrices at encounter frequency.
///
/// Returns strip `A,B,C` with `B` corrected by the STF end-term / integral
/// contributions proportional to `U`. `A` and `C` are the zero-speed strip
/// values at `ω_e` (STF keeps restoring speed-independent; added mass at
/// encounter frequency).
///
/// `speed_u` [m/s] ≥ 0 forward. For `U = 0` this matches
/// [`assemble_strip_matrices`] bit-for-bit on `A,B,C`.
pub fn assemble_strip_matrices_forward_speed(
    stations: &[StripStation],
    rho: f64,
    omega_e: f64,
    speed_u: f64,
    g: f64,
) -> StripMatrices {
    let mut mats = assemble_strip_matrices(stations, rho, omega_e.max(0.0), g);
    if speed_u.abs() < 1e-15 || stations.is_empty() {
        return mats;
    }
    // End stations: lowest x = stern, highest x = stem (bow), sorted by x.
    let (stern, stem) = end_stations(stations);
    let a_s = sectional_a33(rho, omega_e, stern.beam, stern.draft, stern.area_coeff, g);
    let a_t = sectional_a33(rho, omega_e, stem.beam, stem.draft, stem.area_coeff, g);
    let xs = stern.x;
    let xt = stem.x;

    // ∫ a₃₃ dx already in mats.a[0][0] at this ω_e.
    let a33_int = mats.a[0][0];

    // B33 += −U (a33^s − a33^t)
    mats.b[0][0] += -speed_u * (a_s - a_t);
    // B35 += −U (x^s a33^s − x^t a33^t) − U ∫ a33 dx
    mats.b[0][1] += -speed_u * (xs * a_s - xt * a_t) - speed_u * a33_int;
    // B53 += +U (x^s a33^s − x^t a33^t) + U ∫ a33 dx
    mats.b[1][0] += speed_u * (xs * a_s - xt * a_t) + speed_u * a33_int;
    // B55 += −U (x^s² a33^s − x^t² a33^t)
    mats.b[1][1] += -speed_u * (xs * xs * a_s - xt * xt * a_t);

    mats
}

fn end_stations(stations: &[StripStation]) -> (&StripStation, &StripStation) {
    let mut i_stern = 0usize;
    let mut i_stem = 0usize;
    for (i, s) in stations.iter().enumerate() {
        if s.x < stations[i_stern].x {
            i_stern = i;
        }
        if s.x > stations[i_stem].x {
            i_stem = i;
        }
    }
    (&stations[i_stern], &stations[i_stem])
}

/// Froude number based on ship length (forward-speed context).
pub fn ship_froude(speed_u: f64, length: f64, g: f64) -> f64 {
    crate::resistance::froude_number(speed_u, length, g)
}
