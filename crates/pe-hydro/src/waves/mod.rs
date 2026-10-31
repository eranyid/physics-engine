//! Surface gravity waves (Airy, Stokes II, Gerstner, spectral ocean).
//!
//! # Sign / axis conventions (SI)
//!
//! - World frame: right-handed, **+z up**. Mean water level (MWL) is the plane
//!   `z = mwl` (default 0). Seabed for finite depth sits at `z = mwl − h` with
//!   still-water depth `h > 0` [m].
//! - Free-surface elevation **η(x,y,t)** [m] is the vertical displacement of the
//!   free surface **above** the MWL (positive upward). Reported by [`Wave::height`].
//! - Fluid particle velocity [`Wave::velocity`] returns `(u, v, w)` [m/s] in the
//!   world frame: `u` along +x, `v` along +y, `w` positive **upward**. Below the
//!   free surface and above the seabed; callers should not evaluate deep below
//!   the bed.
//! - Progressive waves travel in direction `θ` [rad] measured from +x toward +y
//!   (CCW when viewed from above). Wave number vector
//!   `k⃗ = k (cos θ, sin θ)` [1/m]; phase `χ = k_x x + k_y y − ω t + φ`.
//! - Angular frequency `ω > 0` [rad/s]; gravity `g` [m/s²] (default `g₀`).
//!
//! Clean-room from textbooks / papers — see module-level citations and
//! `THIRD_PARTY.md`.
//!
//! # References
//! - Airy, G. B. (1845); Dean & Dalrymple, *Water Wave Mechanics*;
//!   dispersion `ω² = g k tanh(k h)`.
//! - Stokes, G. G. (1847). On the theory of oscillatory waves (2nd order).
//! - Gerstner, F. (1802); Kinsman, *Wind Waves* (trochoidal / Lagrangian).
//! - Pierson & Moskowitz (1964); Hasselmann et al. JONSWAP (1973);
//!   directional spreading `cos^{2s}` (Longuet-Higgins et al.).

pub mod airy;
pub mod gerstner;
pub mod spectrum;
pub mod stokes;

pub use airy::{
    deep_water_wavelength, dispersion_omega, group_speed, phase_speed, solve_wave_number, AiryWave,
};
pub use gerstner::{GerstnerError, GerstnerWave};
pub use spectrum::{
    jonswap_spectrum, pierson_moskowitz_hs_approx, pierson_moskowitz_spectrum, SpectralOcean,
    SpectrumKind, SpreadingParams, WaveComponent,
};
pub use stokes::StokesSecondOrder;

use pe_math::Vec3;

/// Surface gravity wave field: elevation and kinematics.
///
/// See module docs for sign/axis/unit conventions.
pub trait Wave: Send + Sync {
    /// Free-surface elevation η(x, y, t) [m] above the mean water level.
    fn height(&self, x: f64, y: f64, t: f64) -> f64;

    /// Fluid velocity `(u,v,w)` [m/s] at `(x,y,z,t)`. `w` positive up.
    fn velocity(&self, x: f64, y: f64, z: f64, t: f64) -> Vec3;

    /// Mean water level [m] (world z of the still-water plane).
    fn mean_water_level(&self) -> f64 {
        0.0
    }
}

/// Owned wave field for force generators / PyO3 (enum avoids trait objects).
#[derive(Clone, Debug)]
pub enum WaveField {
    /// Linear Airy wave.
    Airy(AiryWave),
    /// Stokes second-order wave.
    Stokes(StokesSecondOrder),
    /// Gerstner / trochoidal wave.
    Gerstner(GerstnerWave),
    /// Finite-component spectral ocean.
    Spectral(SpectralOcean),
}

impl Wave for WaveField {
    fn height(&self, x: f64, y: f64, t: f64) -> f64 {
        match self {
            WaveField::Airy(w) => w.height(x, y, t),
            WaveField::Stokes(w) => w.height(x, y, t),
            WaveField::Gerstner(w) => w.height(x, y, t),
            WaveField::Spectral(w) => w.height(x, y, t),
        }
    }

    fn velocity(&self, x: f64, y: f64, z: f64, t: f64) -> Vec3 {
        match self {
            WaveField::Airy(w) => w.velocity(x, y, z, t),
            WaveField::Stokes(w) => w.velocity(x, y, z, t),
            WaveField::Gerstner(w) => w.velocity(x, y, z, t),
            WaveField::Spectral(w) => w.velocity(x, y, z, t),
        }
    }

    fn mean_water_level(&self) -> f64 {
        match self {
            WaveField::Airy(w) => w.mean_water_level(),
            WaveField::Stokes(w) => w.mean_water_level(),
            WaveField::Gerstner(w) => w.mean_water_level(),
            WaveField::Spectral(w) => w.mean_water_level(),
        }
    }
}
