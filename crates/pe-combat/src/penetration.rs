//! Terminal interaction: plate perforation, media penetration, ricochet, spall.
//!
//! # Models
//!
//! - **Lambert–Jonas** ballistic limit (Lambert, BRL-R-2036, 1978):
//!   `V50 = k (L/d)^0.15 √(f(z) d³/m)`, `f(z) = z + e^{−z} − 1`,
//!   `z = (t/d) sec^n θ` (`n = 0.75` classically).
//! - **Residual speed** (Recht & Ipson 1963 / Lambert): `v_r = a (v^p − V50^p)^{1/p}`
//!   via [`pe_ballistics::recht_ipson_residual`].
//! - **Poncelet** media: `m dv/dx = −A (a + b v²)` ⇒
//!   `v(x)² = (a/b + v₀²) e^{−2Abx/m} − a/b`; stops at
//!   [`pe_ballistics::poncelet_depth`].
//! - **Ricochet**: logistic probability in graze angle `β` around `β_c`;
//!   rebound `v' = e_t v_t − e_n v_n`.
//! - **Spall**: a share of the energy dissipated in a perforated layer is
//!   given to Mott-distributed fragments ([`pe_blast::mott_sample_mass`]) in a
//!   cone around the exit direction.
//!
//! Every interaction closes an **energy ledger**:
//! `KE_in = KE_out + E_deposited + E_spall` (to round-off).

use core::f64::consts::PI;

use pe_ballistics::ammo::AmmoProfile;
use pe_ballistics::hitscan::{shape_bounding_radius, shape_exit, ShapeRef};
use pe_ballistics::impact::{kinetic_energy, poncelet_depth, recht_ipson_residual};
use pe_ballistics::projectile::{HitEvent, HitOutcome, HitResolver, Projectile, ShotId};
use pe_blast::mott_sample_mass;
use pe_math::libm_ops;
use pe_math::{Vec3, Xoshiro256StarStar};

use crate::armor::{ArmorMaterial, ArmorStack, PenetrationModel, RicochetParams};
use crate::damage::{DamageEvent, Target, TargetTable};

/// Largest obliquity used in `sec θ` [rad] (89°) — avoids the singularity.
const MAX_OBLIQUITY_RAD: f64 = 89.0 * PI / 180.0;

/// Small offset [m] used to place a continuing projectile just off a surface.
const SURFACE_EPS_M: f64 = 1e-6;

/// Lambert `z = (t/d) sec^n θ` (θ from the surface normal, clamped to 89°).
pub fn lambert_z(thickness_m: f64, diameter_m: f64, obliquity_rad: f64, n: f64) -> f64 {
    let th = obliquity_rad.clamp(0.0, MAX_OBLIQUITY_RAD);
    let sec = 1.0 / libm_ops::cos(th);
    (thickness_m / diameter_m) * libm_ops::pow(sec, n)
}

/// Lambert–Jonas ballistic limit `V50` [m/s].
///
/// - `k` — SI Lambert constant (see [`PenetrationModel::Plate`])
/// - `mass_kg`, `diameter_m`, `length_m` — projectile (`L/d` floored at 1)
/// - `thickness_m` — normal plate thickness; `obliquity_rad` from the normal
/// - `n` — obliquity exponent
pub fn lambert_jonas_v50(
    k: f64,
    mass_kg: f64,
    diameter_m: f64,
    length_m: f64,
    thickness_m: f64,
    obliquity_rad: f64,
    n: f64,
) -> f64 {
    if !(mass_kg > 0.0 && diameter_m > 0.0 && thickness_m > 0.0) {
        return 0.0;
    }
    let z = lambert_z(thickness_m, diameter_m, obliquity_rad, n);
    let f = z + libm_ops::exp(-z) - 1.0;
    let ld = (length_m / diameter_m).max(1.0);
    k * libm_ops::pow(ld, 0.15) * libm_ops::sqrt(f * diameter_m * diameter_m * diameter_m / mass_kg)
}

/// Speed [m/s] after travelling `path_m` through a Poncelet medium, or `None`
/// if the projectile stops inside.
pub fn poncelet_exit_speed(
    mass_kg: f64,
    v0: f64,
    area_m2: f64,
    a_pa: f64,
    b_kg_m3: f64,
    path_m: f64,
) -> Option<f64> {
    if !(mass_kg > 0.0 && area_m2 > 0.0 && a_pa > 0.0 && b_kg_m3 > 0.0) {
        return None;
    }
    let r = a_pa / b_kg_m3;
    let v2 =
        (r + v0 * v0) * libm_ops::exp(-2.0 * area_m2 * b_kg_m3 * path_m.max(0.0) / mass_kg) - r;
    if v2 > 0.0 {
        Some(libm_ops::sqrt(v2))
    } else {
        None
    }
}

/// Ricochet probability for graze angle `graze_rad` (from the surface plane).
pub fn ricochet_probability(graze_rad: f64, p: &RicochetParams) -> f64 {
    if p.critical_graze_rad <= 0.0 {
        return 0.0;
    }
    if p.transition_width_rad <= 0.0 {
        return if graze_rad < p.critical_graze_rad {
            1.0
        } else {
            0.0
        };
    }
    1.0 / (1.0 + libm_ops::exp((graze_rad - p.critical_graze_rad) / p.transition_width_rad))
}

/// Rebound velocity: `e_t v_t − e_n v_n` (`normal` is the outward unit normal).
pub fn ricochet_velocity(v: Vec3, normal: Vec3, e_n: f64, e_t: f64) -> Vec3 {
    let n = normal.normalize();
    let vn = n * v.dot(n);
    let vt = v - vn;
    vt * e_t - vn * e_n
}

/// Outcome class of one projectile–surface interaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImpactKind {
    /// Defeated / embedded.
    Stopped,
    /// Perforated every layer and exited.
    Perforated,
    /// Glanced off the outer surface.
    Ricochet,
}

/// A spall / debris fragment created behind a perforated layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpallFragment {
    /// Start position [m] (exit point).
    pub position: Vec3,
    /// Velocity [m/s].
    pub velocity: Vec3,
    /// Mass [kg].
    pub mass_kg: f64,
}

/// Projectile state entering an armor stack.
#[derive(Clone, Copy, Debug)]
pub struct StackInput {
    /// Incident velocity [m/s].
    pub velocity: Vec3,
    /// Projectile mass [kg].
    pub mass_kg: f64,
    /// Projectile diameter [m].
    pub diameter_m: f64,
    /// Projectile length [m] (`0` if unknown).
    pub length_m: f64,
    /// Outward unit surface normal at the entry point.
    pub normal: Vec3,
    /// Geometric path length through the shape along the incident direction
    /// [m] (`0` if unknown).
    pub chord_m: f64,
}

/// Result of [`resolve_stack`].
#[derive(Clone, Debug)]
pub struct StackResult {
    /// Outcome class.
    pub kind: ImpactKind,
    /// Velocity afterwards [m/s] (`0` when stopped).
    pub velocity_out: Vec3,
    /// Mass afterwards [kg].
    pub mass_out_kg: f64,
    /// Multiplicative change of the projectile drag multiplier (≥ 1).
    pub drag_factor: f64,
    /// Depth inside the stopping layer [m] (media only).
    pub stop_depth_m: Option<f64>,
    /// Number of layers fully perforated.
    pub layers_perforated: usize,
    /// Incident kinetic energy [J].
    pub ke_in_j: f64,
    /// Outgoing kinetic energy [J].
    pub ke_out_j: f64,
    /// Energy deposited in the target [J].
    pub e_deposited_j: f64,
    /// Kinetic energy carried by spall [J].
    pub e_spall_j: f64,
    /// Spall fragments (positions are zero; the caller places them).
    pub spall: Vec<SpallFragment>,
    /// Energy deposited in each layer [J] (spall excluded), outermost first.
    /// Only layers the projectile reached are listed. Ledger:
    /// `Σ layer_dissipated_j + e_spall_j + ke_out_j = ke_in_j`.
    pub layer_dissipated_j: Vec<f64>,
    /// Layer index the projectile stopped in (`None` if it exited or ricocheted).
    pub stopped_in_layer: Option<usize>,
}

fn orthonormal_basis(d: Vec3) -> (Vec3, Vec3) {
    let helper = if libm_ops::fabs(d.z) < 0.9 {
        Vec3::unit_z()
    } else {
        Vec3::unit_x()
    };
    let t1 = d.cross(helper).normalize();
    let t2 = d.cross(t1);
    (t1, t2)
}

/// Small Gaussian angular deflection of unit direction `d`.
fn deflect(d: Vec3, std_rad: f64, rng: &mut Xoshiro256StarStar) -> Vec3 {
    if std_rad <= 0.0 {
        return d;
    }
    let (t1, t2) = orthonormal_basis(d);
    let a = rng.normal() * std_rad;
    let b = rng.normal() * std_rad;
    (d + t1 * a + t2 * b).normalize()
}

/// Uniform direction inside a cone of half-angle `half_rad` around unit `d`.
fn cone_direction(d: Vec3, half_rad: f64, rng: &mut Xoshiro256StarStar) -> Vec3 {
    let (t1, t2) = orthonormal_basis(d);
    let cos_max = libm_ops::cos(half_rad.clamp(0.0, PI));
    let cos_a = 1.0 - rng.next_f64() * (1.0 - cos_max);
    let sin_a = libm_ops::sqrt((1.0 - cos_a * cos_a).max(0.0));
    let phi = 2.0 * PI * rng.next_f64();
    (d * cos_a + t1 * (sin_a * libm_ops::cos(phi)) + t2 * (sin_a * libm_ops::sin(phi))).normalize()
}

fn spall_from_layer(
    mat: &ArmorMaterial,
    e_dissipated: f64,
    plug_volume_m3: f64,
    exit_dir: Vec3,
    rng: &mut Xoshiro256StarStar,
) -> (Vec<SpallFragment>, f64) {
    let s = &mat.spall;
    if !(s.energy_fraction > 0.0 && e_dissipated > 0.0 && plug_volume_m3 > 0.0) {
        return (Vec::new(), 0.0);
    }
    let target = s.mass_factor * mat.density_kg_m3 * plug_volume_m3;
    if !(target > 0.0) {
        return (Vec::new(), 0.0);
    }
    let mut masses = Vec::new();
    let mut total = 0.0;
    while total < target && masses.len() < s.max_fragments {
        let m = mott_sample_mass(rng, s.mott_mu_kg).min(target - total);
        if m <= 0.0 {
            break;
        }
        masses.push(m);
        total += m;
    }
    if masses.is_empty() {
        return (Vec::new(), 0.0);
    }
    let e_sp = s.energy_fraction * e_dissipated;
    let speed = libm_ops::sqrt(2.0 * e_sp / total);
    let frags: Vec<SpallFragment> = masses
        .into_iter()
        .map(|m| SpallFragment {
            position: Vec3::zero(),
            velocity: cone_direction(exit_dir, s.cone_half_angle_rad, rng) * speed,
            mass_kg: m,
        })
        .collect();
    let e_actual: f64 = frags
        .iter()
        .map(|f| kinetic_energy(f.mass_kg, f.velocity))
        .sum();
    (frags, e_actual)
}

/// Result for a projectile that stops inside layer `layer` with `ke_layer_in`
/// kinetic energy left (all of it is deposited there).
fn stopped(
    input: &StackInput,
    ke_in: f64,
    e_spall: f64,
    spall: Vec<SpallFragment>,
    mut layer_dissipated_j: Vec<f64>,
    layer: Option<(usize, f64)>,
) -> StackResult {
    let mut layers_perforated = 0;
    let mut stopped_in_layer = None;
    if let Some((li, ke_layer_in)) = layer {
        layer_dissipated_j.push(ke_layer_in);
        layers_perforated = li;
        stopped_in_layer = Some(li);
    }
    StackResult {
        kind: ImpactKind::Stopped,
        velocity_out: Vec3::zero(),
        mass_out_kg: input.mass_kg,
        drag_factor: 1.0,
        stop_depth_m: None,
        layers_perforated,
        ke_in_j: ke_in,
        ke_out_j: 0.0,
        e_deposited_j: (ke_in - e_spall).max(0.0),
        e_spall_j: e_spall,
        spall,
        layer_dissipated_j,
        stopped_in_layer,
    }
}

/// Run a projectile through an armor stack (outermost layer first).
///
/// `materials` is indexed by [`crate::ArmorLayer::material`]. Stochastic
/// parts (ricochet roll, deflection, spall) draw from `rng`.
pub fn resolve_stack(
    materials: &[ArmorMaterial],
    stack: &ArmorStack,
    input: StackInput,
    rng: &mut Xoshiro256StarStar,
) -> StackResult {
    let ke_in = kinetic_energy(input.mass_kg, input.velocity);
    let speed_in = input.velocity.norm();
    if stack.layers.is_empty() || speed_in <= 0.0 {
        let mut r = stopped(&input, ke_in, 0.0, Vec::new(), Vec::new(), None);
        if stack.layers.is_empty() && speed_in > 0.0 {
            // No armor: nothing to stop it — pass through unchanged.
            r.kind = ImpactKind::Perforated;
            r.velocity_out = input.velocity;
            r.ke_out_j = ke_in;
            r.e_deposited_j = 0.0;
        } else if !stack.layers.is_empty() {
            // Zero speed: nothing to deposit, but report the outer layer.
            r.layer_dissipated_j.push(0.0);
            r.stopped_in_layer = Some(0);
        }
        return r;
    }
    let n = input.normal.normalize();
    let dir_in = input.velocity / speed_in;
    let cos_th = (-n.dot(dir_in)).clamp(0.0, 1.0);
    let theta = libm_ops::acos(cos_th);
    let graze = 0.5 * PI - theta;

    let mut v = input.velocity;
    let mut mass = input.mass_kg;
    let mut drag_factor = 1.0;
    let mut e_spall = 0.0;
    let mut spall = Vec::new();
    let mut layer_dep: Vec<f64> = Vec::with_capacity(stack.layers.len());
    let area = 0.25 * PI * input.diameter_m * input.diameter_m;

    for (li, layer) in stack.layers.iter().enumerate() {
        let mat = &materials[layer.material];
        let speed = v.norm();
        let dir = v / speed;

        // Ricochet is decided on the outer surface only.
        if li == 0 {
            let p_rico = ricochet_probability(graze, &mat.ricochet);
            if p_rico > 0.0 && rng.next_f64() < p_rico {
                let r = &mat.ricochet;
                let mut vr = ricochet_velocity(v, n, r.normal_restitution, r.tangential_retention);
                let sp = vr.norm();
                if sp > 0.0 {
                    let mut d = deflect(vr / sp, r.deflection_std_rad, rng);
                    if d.dot(n) < 0.0 {
                        d = (d - n * (2.0 * d.dot(n))).normalize();
                    }
                    vr = d * sp;
                }
                let ke_out = kinetic_energy(mass, vr);
                return StackResult {
                    kind: ImpactKind::Ricochet,
                    velocity_out: vr,
                    mass_out_kg: mass,
                    drag_factor: 1.0 + mat.tumble_drag_gain * (1.0 - ke_out / ke_in).max(0.0),
                    stop_depth_m: None,
                    layers_perforated: 0,
                    ke_in_j: ke_in,
                    ke_out_j: ke_out,
                    e_deposited_j: (ke_in - ke_out).max(0.0),
                    e_spall_j: 0.0,
                    spall: Vec::new(),
                    layer_dissipated_j: vec![(ke_in - ke_out).max(0.0)],
                    stopped_in_layer: None,
                };
            }
        }

        let ke_layer_in = kinetic_energy(mass, v);
        let (v_out, mass_out, plug_volume) = match mat.model {
            PenetrationModel::Plate {
                lambert_k,
                obliquity_exponent,
                residual_a,
                residual_p,
                erosion_per_z,
            } => {
                let t = if layer.thickness_m > 0.0 {
                    layer.thickness_m
                } else {
                    input.chord_m * cos_th
                };
                if !(t > 0.0) {
                    return stopped(
                        &input,
                        ke_in,
                        e_spall,
                        spall,
                        layer_dep,
                        Some((li, ke_layer_in)),
                    );
                }
                let v50 = lambert_jonas_v50(
                    lambert_k,
                    mass,
                    input.diameter_m,
                    input.length_m,
                    t,
                    theta,
                    obliquity_exponent,
                );
                if speed <= v50 {
                    return stopped(
                        &input,
                        ke_in,
                        e_spall,
                        spall,
                        layer_dep,
                        Some((li, ke_layer_in)),
                    );
                }
                let vr = recht_ipson_residual(speed, v50, residual_a, residual_p);
                let z = lambert_z(t, input.diameter_m, theta, obliquity_exponent);
                let m_out = (mass * (1.0 - erosion_per_z * z)).clamp(0.2 * mass, mass);
                let los = t / cos_th.max(libm_ops::cos(MAX_OBLIQUITY_RAD));
                (vr, m_out, area * los)
            }
            PenetrationModel::Medium {
                poncelet_a_pa,
                poncelet_b_kg_m3,
            } => {
                let path = if layer.thickness_m > 0.0 {
                    layer.thickness_m / cos_th.max(libm_ops::cos(MAX_OBLIQUITY_RAD))
                } else {
                    input.chord_m
                };
                match poncelet_exit_speed(mass, speed, area, poncelet_a_pa, poncelet_b_kg_m3, path)
                {
                    Some(vr) if path > 0.0 => (vr, mass, area * path),
                    _ => {
                        let mut r = stopped(
                            &input,
                            ke_in,
                            e_spall,
                            spall,
                            layer_dep,
                            Some((li, ke_layer_in)),
                        );
                        r.stop_depth_m =
                            poncelet_depth(mass, speed, area, poncelet_a_pa, poncelet_b_kg_m3)
                                .map(|d| if path > 0.0 { d.min(path) } else { d });
                        return r;
                    }
                }
            }
        };

        let ke_layer_out = kinetic_energy(mass_out, dir * v_out);
        let loss = if ke_layer_in > 0.0 {
            (1.0 - ke_layer_out / ke_layer_in).clamp(0.0, 1.0)
        } else {
            1.0
        };
        drag_factor *= 1.0 + mat.tumble_drag_gain * loss;
        let new_dir = deflect(dir, mat.exit_deflection_std_rad * loss, rng);
        v = new_dir * v_out;
        mass = mass_out;
        let (frags, e_sp) =
            spall_from_layer(mat, ke_layer_in - ke_layer_out, plug_volume, new_dir, rng);
        e_spall += e_sp;
        spall.extend(frags);
        layer_dep.push((ke_layer_in - kinetic_energy(mass, v) - e_sp).max(0.0));
    }

    let ke_out = kinetic_energy(mass, v);
    StackResult {
        kind: ImpactKind::Perforated,
        velocity_out: v,
        mass_out_kg: mass,
        drag_factor,
        stop_depth_m: None,
        layers_perforated: stack.layers.len(),
        ke_in_j: ke_in,
        ke_out_j: ke_out,
        e_deposited_j: (ke_in - ke_out - e_spall).max(0.0),
        e_spall_j: e_spall,
        spall,
        layer_dissipated_j: layer_dep,
        stopped_in_layer: None,
    }
}

/// Full record of one interaction, for damage, effects and audio.
#[derive(Clone, Debug)]
pub struct ImpactRecord {
    /// Shot id.
    pub shot_id: ShotId,
    /// Shape index hit.
    pub shape_index: usize,
    /// Body id of the shape, if any.
    pub body_id: Option<usize>,
    /// Simulation time [s].
    pub time_s: f64,
    /// Outcome class.
    pub kind: ImpactKind,
    /// Entry point [m].
    pub entry: Vec3,
    /// Exit point [m] (perforation only).
    pub exit: Option<Vec3>,
    /// Outward normal at entry.
    pub normal: Vec3,
    /// Incident velocity [m/s].
    pub velocity_in: Vec3,
    /// Outgoing velocity [m/s].
    pub velocity_out: Vec3,
    /// Incident mass [kg].
    pub mass_in_kg: f64,
    /// Obliquity from the normal [rad].
    pub incidence_rad: f64,
    /// Incident KE [J].
    pub ke_in_j: f64,
    /// Outgoing KE [J].
    pub ke_out_j: f64,
    /// Energy deposited in the target [J].
    pub e_deposited_j: f64,
    /// Energy carried away by spall [J].
    pub e_spall_j: f64,
    /// Stop depth inside a medium [m].
    pub stop_depth_m: Option<f64>,
    /// Spall fragments (world positions).
    pub spall: Vec<SpallFragment>,
}

/// [`HitResolver`] that runs every hit through a per-shape [`Target`].
///
/// - [`Target::Inert`] (or a shape index beyond the table) stops the projectile,
///   like [`pe_ballistics::StopOnHit`].
/// - [`Target::Armor`] resolves an [`ArmorStack`] (walls, plates, vehicles).
/// - [`Target::Zone`] resolves the zone's armor plus its tissue and appends a
///   [`DamageEvent`] to [`damage_events`](Self::damage_events). Spall from armor
///   worn on the body goes into the tissue instead of flying away.
///
/// Every interaction is appended to [`records`](Self::records); drain both
/// logs between steps.
#[derive(Clone, Debug, Default)]
pub struct CombatResolver {
    /// Material table.
    pub materials: Vec<ArmorMaterial>,
    /// What each shape index is (same indexing as the `shapes` slice).
    pub targets: TargetTable,
    /// Interaction log.
    pub records: Vec<ImpactRecord>,
    /// Damage to character zones.
    pub damage_events: Vec<DamageEvent>,
}

impl CombatResolver {
    /// New resolver from materials and per-shape armor (`None` = inert).
    pub fn new(materials: Vec<ArmorMaterial>, shape_armor: Vec<Option<ArmorStack>>) -> Self {
        Self::with_targets(materials, TargetTable::from_armor(shape_armor))
    }

    /// New resolver from materials and a full target table.
    pub fn with_targets(materials: Vec<ArmorMaterial>, targets: TargetTable) -> Self {
        Self {
            materials,
            targets,
            records: Vec::new(),
            damage_events: Vec::new(),
        }
    }

    /// Add a material; returns its index.
    pub fn add_material(&mut self, m: ArmorMaterial) -> usize {
        self.materials.push(m);
        self.materials.len() - 1
    }
}

impl HitResolver for CombatResolver {
    fn resolve(
        &mut self,
        hit: &HitEvent,
        shape: &ShapeRef,
        profile: &AmmoProfile,
        projectile: &mut Projectile,
    ) -> HitOutcome {
        let v = hit.velocity;
        let speed = v.norm();
        let mut record = ImpactRecord {
            shot_id: hit.shot_id,
            shape_index: hit.shape_index,
            body_id: hit.body_id,
            time_s: hit.time_s,
            kind: ImpactKind::Stopped,
            entry: hit.position,
            exit: None,
            normal: hit.normal,
            velocity_in: v,
            velocity_out: Vec3::zero(),
            mass_in_kg: projectile.mass,
            incidence_rad: hit.incidence_rad,
            ke_in_j: hit.energy_j,
            ke_out_j: 0.0,
            e_deposited_j: hit.energy_j,
            e_spall_j: 0.0,
            stop_depth_m: None,
            spall: Vec::new(),
        };
        let (stack, zone) = match self.targets.get(hit.shape_index) {
            Some(Target::Armor(s)) if speed > 0.0 => (s, None),
            Some(Target::Zone {
                character,
                zone,
                stack,
            }) if speed > 0.0 => (stack, Some((*character, *zone, stack.layers.len() - 1))),
            _ => {
                self.records.push(record);
                return HitOutcome::Stop;
            }
        };
        let dir = v / speed;
        let max_len = 2.0 * shape_bounding_radius(shape) + 1.0;
        let (exit_pt, chord) = match shape_exit(hit.position, dir, shape, max_len) {
            Some((p, _n, c)) => (p, c),
            None => (hit.position, 0.0),
        };
        let mut res = resolve_stack(
            &self.materials,
            stack,
            StackInput {
                velocity: v,
                mass_kg: projectile.mass,
                diameter_m: profile.caliber_m,
                length_m: profile.length_m,
                normal: hit.normal,
                chord_m: chord,
            },
            &mut projectile.rng,
        );
        if let Some((character, zone, tissue_layer)) = zone {
            // Spall from worn armor ends up in the wearer once the armor is
            // breached; otherwise it stays in the armor. Ledger:
            // e_armor + e_tissue = KE_in − KE_out.
            let reached_tissue = res.layer_dissipated_j.len() > tissue_layer;
            let spall_e = res.e_spall_j;
            let tissue_spall = if reached_tissue { spall_e } else { 0.0 };
            let e_tissue = res
                .layer_dissipated_j
                .get(tissue_layer)
                .copied()
                .unwrap_or(0.0)
                + tissue_spall;
            let e_armor = res
                .layer_dissipated_j
                .iter()
                .take(tissue_layer)
                .sum::<f64>()
                + (spall_e - tissue_spall);
            res.e_deposited_j += spall_e;
            res.e_spall_j = 0.0;
            res.spall.clear();
            let wound_track_m = if !reached_tissue {
                0.0
            } else if res.kind == ImpactKind::Perforated {
                chord
            } else {
                res.stop_depth_m.unwrap_or(0.0)
            };
            self.damage_events.push(DamageEvent {
                character,
                zone,
                shot_id: hit.shot_id,
                time_s: hit.time_s,
                position: hit.position,
                direction: dir,
                ke_in_j: res.ke_in_j,
                e_tissue_j: e_tissue,
                e_armor_j: e_armor,
                penetrated_armor: reached_tissue && tissue_layer > 0,
                through_and_through: res.kind == ImpactKind::Perforated,
                ricochet: res.kind == ImpactKind::Ricochet,
                wound_track_m,
                momentum_in: v * projectile.mass,
                momentum_out: res.velocity_out * res.mass_out_kg,
            });
        }
        record.kind = res.kind;
        record.velocity_out = res.velocity_out;
        record.ke_out_j = res.ke_out_j;
        record.e_deposited_j = res.e_deposited_j;
        record.e_spall_j = res.e_spall_j;
        record.stop_depth_m = res.stop_depth_m;
        let outcome = match res.kind {
            ImpactKind::Stopped => HitOutcome::Stop,
            ImpactKind::Ricochet => HitOutcome::Continue {
                position: hit.position + hit.normal * SURFACE_EPS_M,
                velocity: res.velocity_out,
                mass_kg: res.mass_out_kg,
                drag_multiplier: projectile.drag_multiplier * res.drag_factor,
            },
            ImpactKind::Perforated => {
                record.exit = Some(exit_pt);
                let out_dir = if res.velocity_out.norm_squared() > 0.0 {
                    res.velocity_out.normalize()
                } else {
                    dir
                };
                HitOutcome::Continue {
                    position: exit_pt + out_dir * SURFACE_EPS_M,
                    velocity: res.velocity_out,
                    mass_kg: res.mass_out_kg,
                    drag_multiplier: projectile.drag_multiplier * res.drag_factor,
                }
            }
        };
        record.spall = res
            .spall
            .into_iter()
            .map(|mut f| {
                f.position = exit_pt;
                f
            })
            .collect();
        self.records.push(record);
        outcome
    }
}
