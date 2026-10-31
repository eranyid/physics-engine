//! International Standard Atmosphere (ISA / ICAO Doc 7488) through 20 km.
//!
//! Geopotential altitude `h` [m]. Returns SI: `T` [K], `p` [Pa], `ρ` [kg/m³], `a` [m/s].
//! Layer model: troposphere 0..11 km (lapse 6.5 K/km), isothermal 11..20 km.
//!
//! Geometric vs geopotential altitude is not distinguished. The model stops at 20 km.
//! Compressibility / Mach corrections are out of scope (no Prandtl–Glauert; Mach≈1
//! singularities are not modelled).

use pe_math::libm_ops;
use pe_math::STANDARD_GRAVITY;

/// Sea-level static temperature [K] (ISA definition).
pub const T0: f64 = 288.15;
/// Sea-level static pressure [Pa] (ISA definition).
pub const P0: f64 = 101_325.0;
/// Sea-level density [kg/m³] (ISA definition).
pub const RHO0: f64 = 1.225;
/// Tropospheric lapse-rate magnitude [K/m] (T decreases with altitude).
pub const LAPSE: f64 = 0.0065;
/// Tropopause geopotential altitude [m].
pub const H_TROPOPAUSE: f64 = 11_000.0;
/// Top of the modelled lower stratosphere [m].
pub const H_STRATO_TOP: f64 = 20_000.0;
/// Ratio of specific heats for dry air (isentropic).
pub const GAMMA_AIR: f64 = 1.4;
/// Specific gas constant [J/(kg K)] implied by sea-level state: `R = p₀/(ρ₀ T₀)`.
pub const R_AIR: f64 = P0 / (RHO0 * T0);
/// Hydrostatic exponent `g₀/(R L)` for the troposphere power law [dimensionless].
const EXP: f64 = STANDARD_GRAVITY / (R_AIR * LAPSE);

/// Error when altitude exceeds the ISA model limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IsaAltitudeError {
    /// Requested altitude [m].
    pub h: i64,
}

impl core::fmt::Display for IsaAltitudeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ISA model limited to h <= {} m; got {}",
            H_STRATO_TOP as i64, self.h
        )
    }
}

impl std::error::Error for IsaAltitudeError {}

/// Static temperature [K] at geopotential altitude `h` [m].
pub fn temperature(h: f64) -> Result<f64, IsaAltitudeError> {
    if h < H_TROPOPAUSE {
        Ok(T0 - LAPSE * h)
    } else if h <= H_STRATO_TOP {
        Ok(T0 - LAPSE * H_TROPOPAUSE) // 216.65 K
    } else {
        Err(IsaAltitudeError { h: h as i64 })
    }
}

/// Static pressure [Pa] at geopotential altitude `h` [m].
pub fn pressure(h: f64) -> Result<f64, IsaAltitudeError> {
    let t = temperature(h)?;
    if h < H_TROPOPAUSE {
        Ok(P0 * libm_ops::pow(t / T0, EXP))
    } else {
        let t11 = temperature(H_TROPOPAUSE)?;
        let p11 = P0 * libm_ops::pow(t11 / T0, EXP);
        Ok(p11 * libm_ops::exp(-STANDARD_GRAVITY * (h - H_TROPOPAUSE) / (R_AIR * t11)))
    }
}

/// Mass density [kg/m³] via ideal gas law `ρ = p / (R T)`.
pub fn density(h: f64) -> Result<f64, IsaAltitudeError> {
    Ok(pressure(h)? / (R_AIR * temperature(h)?))
}

/// Isentropic speed of sound `a = √(γ R T)` [m/s].
pub fn speed_of_sound(h: f64) -> Result<f64, IsaAltitudeError> {
    Ok(libm_ops::sqrt(GAMMA_AIR * R_AIR * temperature(h)?))
}

/// Convenience bundle of atmospheric state at `h` [m].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmState {
    /// Static temperature [K].
    pub temperature: f64,
    /// Static pressure [Pa].
    pub pressure: f64,
    /// Mass density [kg/m³].
    pub density: f64,
    /// Speed of sound [m/s].
    pub speed_of_sound: f64,
}

/// Temperature, pressure, density, and speed of sound at `h` [m].
pub fn state(h: f64) -> Result<AtmState, IsaAltitudeError> {
    Ok(AtmState {
        temperature: temperature(h)?,
        pressure: pressure(h)?,
        density: density(h)?,
        speed_of_sound: speed_of_sound(h)?,
    })
}

/// Density model for force generators: full ISA table or a constant [kg/m³].
#[derive(Clone, Debug)]
pub enum DensityModel {
    /// Look up `density(h)` from the ISA model (`h` = world `z` [m]).
    Isa,
    /// Constant density [kg/m³] (ignores altitude).
    Constant(f64),
}

impl DensityModel {
    /// Density [kg/m³] at world height `h` [m]. Falls back to sea-level if ISA rejects `h`.
    pub fn density_at(&self, h: f64) -> f64 {
        match self {
            DensityModel::Isa => density(h).unwrap_or(RHO0),
            DensityModel::Constant(rho) => *rho,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    #[test]
    fn published_isa_sea_level() {
        assert_eq!(temperature(0.0).unwrap(), 288.15);
        assert_eq!(pressure(0.0).unwrap(), 101_325.0);
        assert_eq!(density(0.0).unwrap(), 1.225);
        assert_abs_diff_eq!(speed_of_sound(0.0).unwrap(), 340.294, epsilon = 5e-4);
    }

    #[test]
    fn published_isa_11_km() {
        assert_abs_diff_eq!(temperature(11_000.0).unwrap(), 216.65, epsilon = 1e-9);
        assert_abs_diff_eq!(pressure(11_000.0).unwrap(), 22_632.1, epsilon = 0.1);
        assert_abs_diff_eq!(density(11_000.0).unwrap(), 0.36392, epsilon = 1e-5);
    }

    #[test]
    fn published_isa_20_km() {
        assert_abs_diff_eq!(pressure(20_000.0).unwrap(), 5474.89, epsilon = 0.02);
        assert_abs_diff_eq!(density(20_000.0).unwrap(), 0.088035, epsilon = 1e-6);
    }

    #[test]
    fn tropopause_continuity() {
        let h = H_TROPOPAUSE;
        assert_relative_eq!(
            temperature(h - 1e-6).unwrap(),
            temperature(h).unwrap(),
            max_relative = 1e-9
        );
        assert_relative_eq!(
            pressure(h - 1e-3).unwrap(),
            pressure(h).unwrap(),
            max_relative = 1e-6
        );
    }

    #[test]
    fn rejects_above_model() {
        assert!(temperature(20_001.0).is_err());
    }
}
