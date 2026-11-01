//! Individual projectile body and SoA projectile system.
//!
//! Each bullet has its own state and a dedicated RNG stream seeded from
//! `(shot_id)` so trajectories are reproducible regardless of how many other
//! bullets are in flight. Stepping is deterministic in ascending id order.

use pe_aero::atmosphere::{density as isa_density, speed_of_sound};
use pe_aero::turbulence::WindField;
use pe_core::World;
use pe_math::libm_ops;
use pe_math::{Vec3, Xoshiro256StarStar, STANDARD_GRAVITY};

use crate::ammo::AmmoProfile;
use crate::drag::drag_force_magnitude;
use crate::earth::LocalFrame;
use crate::hitscan::{segment_vs_shapes, segment_vs_shapes_excluding, Hit, ShapeRef};
use crate::impact::{apply_impact_impulse, kinetic_energy};
use crate::spin::{magnus_accel, spin_drift_accel, DEFAULT_MAGNUS_K, DEFAULT_SPIN_DRIFT_K};

/// Stable projectile identifier (also seeds the per-shot RNG).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShotId(pub u64);

/// Hit event reported by the projectile system.
#[derive(Clone, Debug)]
pub struct HitEvent {
    /// Shot id.
    pub shot_id: ShotId,
    /// World hit position [m].
    pub position: Vec3,
    /// Outward normal.
    pub normal: Vec3,
    /// Impact velocity [m/s].
    pub velocity: Vec3,
    /// Kinetic energy [J].
    pub energy_j: f64,
    /// Angle of incidence [rad] from surface normal (0 = head-on).
    pub incidence_rad: f64,
    /// Hit body / shape index (caller-defined; `None` if environment shape only).
    pub body_id: Option<usize>,
    /// Time of hit [s].
    pub time_s: f64,
    /// Index of the hit shape in the `shapes` slice passed to `step`.
    pub shape_index: usize,
}

/// What happens to a projectile after a surface interaction.
///
/// Returned by a [`HitResolver`]. [`HitOutcome::Stop`] reproduces the classic
/// behaviour of [`ProjectileSystem::step`] (bullet dies at the surface).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HitOutcome {
    /// Projectile stops (embedded / defeated). It is placed at the hit point.
    Stop,
    /// Projectile continues — perforation (exit point beyond the surface) or
    /// ricochet (position on the surface, reflected velocity).
    Continue {
        /// New position [m] (exit point or ricochet point).
        position: Vec3,
        /// New velocity [m/s].
        velocity: Vec3,
        /// New mass [kg] (erosion / fragmentation; ≤ incident mass).
        mass_kg: f64,
        /// Drag multiplier from now on (≥ 1 for yawing / tumbling).
        drag_multiplier: f64,
    },
}

/// Decides the outcome of every projectile–surface interaction.
///
/// Called in deterministic order (ascending [`ShotId`], then along each
/// bullet's path). The resolver may draw from `projectile.rng` — that stream
/// is per-shot, so outcomes stay reproducible regardless of other bullets.
pub trait HitResolver {
    /// Resolve one hit. `shape` is the shape that was hit, `profile` the ammo.
    fn resolve(
        &mut self,
        hit: &HitEvent,
        shape: &ShapeRef,
        profile: &AmmoProfile,
        projectile: &mut Projectile,
    ) -> HitOutcome;
}

/// Classic resolver: every hit stops the projectile.
#[derive(Clone, Copy, Debug, Default)]
pub struct StopOnHit;

impl HitResolver for StopOnHit {
    fn resolve(
        &mut self,
        _hit: &HitEvent,
        _shape: &ShapeRef,
        _profile: &AmmoProfile,
        _projectile: &mut Projectile,
    ) -> HitOutcome {
        HitOutcome::Stop
    }
}

/// Maximum surface interactions resolved per projectile sub-step.
pub const MAX_INTERACTIONS_PER_SUBSTEP: usize = 8;

/// Single in-flight projectile state.
#[derive(Clone, Debug)]
pub struct Projectile {
    /// Shot id.
    pub id: ShotId,
    /// Ammo profile index into the system table.
    pub profile_index: usize,
    /// Position [m].
    pub x: Vec3,
    /// Velocity [m/s].
    pub v: Vec3,
    /// Mass [kg].
    pub mass: f64,
    /// Spin rate [rad/s] about velocity axis.
    pub spin_rad_s: f64,
    /// Alive flag.
    pub alive: bool,
    /// Accumulated drag work [J] (dissipated).
    pub drag_work_j: f64,
    /// Drag multiplier (1 = nominal; > 1 after yaw / tumble from a hit).
    pub drag_multiplier: f64,
    /// Per-shot RNG (for any stochastic follow-ups; dispersion applied at spawn).
    pub rng: Xoshiro256StarStar,
}

/// Muzzle spawn pose.
#[derive(Clone, Copy, Debug)]
pub struct MuzzlePose {
    /// Muzzle position [m].
    pub position: Vec3,
    /// Aim unit direction (world).
    pub direction: Vec3,
    /// Optional barrel length [m] for velocity scaling.
    pub barrel_length_m: Option<f64>,
}

/// Recoil impulse returned to the shooter [N·s] (world), opposite muzzle velocity.
#[derive(Clone, Copy, Debug)]
pub struct RecoilImpulse {
    /// Impulse [N·s].
    pub impulse: Vec3,
    /// Application point [m].
    pub point: Vec3,
}

/// SoA projectile system — thousands of bullets, stepped in id order.
#[derive(Clone, Debug)]
pub struct ProjectileSystem {
    /// Ammo profiles (indexed by projectile.profile_index).
    pub profiles: Vec<AmmoProfile>,
    /// Projectiles (kept sorted by id after each spawn batch).
    pub projectiles: Vec<Projectile>,
    /// Local Earth frame (Coriolis / gravity).
    pub frame: LocalFrame,
    /// Wind (steady + optional Dryden).
    pub wind: WindField,
    /// Enable spin drift / Magnus.
    pub enable_spin_effects: bool,
    /// Simulation time [s].
    pub t: f64,
    /// Hit events from the last `step`.
    pub hits: Vec<HitEvent>,
}

impl ProjectileSystem {
    /// Empty system at latitude `lat_rad` [rad], calm wind.
    pub fn new(lat_rad: f64) -> Self {
        Self {
            profiles: Vec::new(),
            projectiles: Vec::new(),
            frame: LocalFrame::new(lat_rad),
            wind: WindField::steady(Vec3::zero()),
            enable_spin_effects: true,
            t: 0.0,
            hits: Vec::new(),
        }
    }

    /// Register a profile; returns index.
    pub fn add_profile(&mut self, p: AmmoProfile) -> usize {
        self.profiles.push(p);
        self.profiles.len() - 1
    }

    /// Spawn a projectile. Deterministic given `shot_id` and profile.
    ///
    /// Returns recoil impulse for the shooter body (caller applies to World).
    pub fn spawn(
        &mut self,
        shot_id: ShotId,
        profile_index: usize,
        muzzle: MuzzlePose,
    ) -> RecoilImpulse {
        let profile = &self.profiles[profile_index];
        let mut rng = Xoshiro256StarStar::seed(shot_id.0 ^ 0x9E37_79B9_7F4A_7C15);
        let mut v0 = profile.muzzle_velocity_for_barrel(muzzle.barrel_length_m.unwrap_or(0.0));
        if muzzle.barrel_length_m.is_none() {
            v0 = profile.muzzle_velocity_m_s;
        }
        // Velocity variance
        v0 += rng.normal() * profile.velocity_std_m_s;
        v0 = v0.max(0.0);
        // Dispersion cone
        let dir = disperse_direction(muzzle.direction, profile.dispersion_std_rad, &mut rng);
        let velocity = dir * v0;
        let spin = profile.spin_rad_per_s(v0);
        let p = Projectile {
            id: shot_id,
            profile_index,
            x: muzzle.position,
            v: velocity,
            mass: profile.mass_kg,
            spin_rad_s: spin,
            alive: true,
            drag_work_j: 0.0,
            drag_multiplier: 1.0,
            rng,
        };
        let recoil = RecoilImpulse {
            impulse: velocity * (-profile.mass_kg), // −p_bullet
            point: muzzle.position,
        };
        self.projectiles.push(p);
        self.projectiles.sort_by_key(|p| p.id);
        recoil
    }

    /// Apply recoil to a shooter body in `world`.
    pub fn apply_recoil(world: &mut World, body: usize, recoil: RecoilImpulse) {
        world.apply_impulse(body, recoil.impulse, recoil.point);
    }

    /// Count alive projectiles.
    pub fn alive_count(&self) -> usize {
        self.projectiles.iter().filter(|p| p.alive).count()
    }

    /// Step all alive projectiles by `dt` [s] with optional environment shapes.
    ///
    /// Uses RK4 on the point-mass ODEs; sub-steps when a swept segment would
    /// cross a target (binary refine) so fast bullets do not tunnel.
    pub fn step(&mut self, dt: f64, shapes: &[ShapeRef], shape_body_ids: &[Option<usize>]) {
        self.step_with_resolver(dt, shapes, shape_body_ids, &mut StopOnHit);
    }

    /// Like [`step`](Self::step) but every hit is passed to `resolver`, which
    /// may stop, perforate or ricochet the projectile. After a non-stop
    /// outcome the bullet continues along a straight segment for the rest of
    /// the sub-step (the shape just hit is excluded for that segment), so
    /// several walls / plates can be crossed in one step without tunneling.
    /// All interactions are recorded in [`hits`](Self::hits).
    pub fn step_with_resolver<R: HitResolver + ?Sized>(
        &mut self,
        dt: f64,
        shapes: &[ShapeRef],
        shape_body_ids: &[Option<usize>],
        resolver: &mut R,
    ) {
        self.hits.clear();
        // Advance wind once per system step (shared atmosphere).
        let _ = self.wind.step(100.0, dt);
        // Clone ids order — already sorted.
        let n = self.projectiles.len();
        for i in 0..n {
            if !self.projectiles[i].alive {
                continue;
            }
            self.step_one(i, dt, shapes, shape_body_ids, resolver);
        }
        self.t += dt;
    }

    fn step_one<R: HitResolver + ?Sized>(
        &mut self,
        index: usize,
        dt: f64,
        shapes: &[ShapeRef],
        shape_body_ids: &[Option<usize>],
        resolver: &mut R,
    ) {
        let x0 = self.projectiles[index].x;
        // Adaptive sub-step by Mach: finer near transonic / high speed.
        let speed = self.projectiles[index].v.norm();
        let alt = self.projectiles[index].x.z;
        let a_sound = speed_of_sound(alt).unwrap_or(340.0);
        let mach = speed / a_sound.max(1.0);
        let n_sub = if mach > 0.9 {
            4
        } else if speed > 500.0 {
            2
        } else {
            1
        };
        let h = dt / n_sub as f64;
        for _ in 0..n_sub {
            if !self.projectiles[index].alive {
                break;
            }
            let x_before = self.projectiles[index].x;
            self.rk4_point_mass(index, h);
            let x_after = self.projectiles[index].x;
            let mut seg_a = x_before;
            let mut seg_b = x_after;
            let mut seg_dt = h;
            let mut exclude: Option<usize> = None;
            for _ in 0..MAX_INTERACTIONS_PER_SUBSTEP {
                let found = match exclude {
                    None => segment_vs_shapes(seg_a, seg_b, shapes),
                    Some(ex) => segment_vs_shapes_excluding(seg_a, seg_b, shapes, ex),
                };
                let Some((si, hit)) = found else {
                    break;
                };
                let v = self.projectiles[index].v;
                let mass = self.projectiles[index].mass;
                let id = self.projectiles[index].id;
                let inc = libm_ops::acos((-hit.normal.dot(v.normalize())).clamp(-1.0, 1.0));
                let body = shape_body_ids.get(si).copied().flatten();
                let event = HitEvent {
                    shot_id: id,
                    position: hit.point,
                    normal: hit.normal,
                    velocity: v,
                    energy_j: kinetic_energy(mass, v),
                    incidence_rad: inc,
                    body_id: body,
                    time_s: self.t,
                    shape_index: si,
                };
                let pidx = self.projectiles[index].profile_index;
                let outcome = resolver.resolve(
                    &event,
                    &shapes[si],
                    &self.profiles[pidx],
                    &mut self.projectiles[index],
                );
                self.hits.push(event);
                match outcome {
                    HitOutcome::Stop => {
                        self.projectiles[index].x = hit.point;
                        self.projectiles[index].v = v;
                        self.projectiles[index].alive = false;
                        break;
                    }
                    HitOutcome::Continue {
                        position,
                        velocity,
                        mass_kg,
                        drag_multiplier,
                    } => {
                        let p = &mut self.projectiles[index];
                        p.mass = mass_kg.clamp(0.0, mass);
                        p.drag_multiplier = drag_multiplier.max(1e-6);
                        p.v = velocity;
                        if p.mass <= 0.0 || p.mass.is_nan() || velocity.norm_squared() < 1e-12 {
                            p.x = position;
                            p.alive = false;
                            break;
                        }
                        // Remaining fraction of this sub-step along a straight line.
                        let rem = seg_dt * (1.0 - hit.t).clamp(0.0, 1.0);
                        seg_dt = rem;
                        seg_a = position;
                        seg_b = position + velocity * rem;
                        p.x = seg_b;
                        exclude = Some(si);
                    }
                }
            }
            if !self.projectiles[index].alive {
                break;
            }
        }
        let _ = x0;
    }

    fn rk4_point_mass(&mut self, index: usize, dt: f64) {
        let pidx = self.projectiles[index].profile_index;
        let spin = self.projectiles[index].spin_rad_s;
        let x = self.projectiles[index].x;
        let v = self.projectiles[index].v;
        let mass = self.projectiles[index].mass;
        let drag_mult = self.projectiles[index].drag_multiplier;
        let accel = |sys: &ProjectileSystem, x: Vec3, v: Vec3, pidx: usize, spin: f64| {
            sys.accel_at(x, v, pidx, spin, mass, drag_mult)
        };
        let a1 = accel(self, x, v, pidx, spin);
        let a2 = accel(self, x + v * (0.5 * dt), v + a1 * (0.5 * dt), pidx, spin);
        let a3 = accel(
            self,
            x + (v + a1 * (0.5 * dt)) * (0.5 * dt),
            v + a2 * (0.5 * dt),
            pidx,
            spin,
        );
        let a4 = accel(
            self,
            x + v * dt + a1 * (0.5 * dt * dt),
            v + a3 * dt,
            pidx,
            spin,
        );
        // Simpler classical RK4 for (x,v):
        let k1v = a1;
        let k1x = v;
        let k2v = a2;
        let k2x = v + k1v * (0.5 * dt);
        let k3v = a3;
        let k3x = v + k2v * (0.5 * dt);
        let k4v = a4;
        let k4x = v + k3v * dt;
        let v_new = v + (k1v + k2v * 2.0 + k3v * 2.0 + k4v) * (dt / 6.0);
        let x_new = x + (k1x + k2x * 2.0 + k3x * 2.0 + k4x) * (dt / 6.0);
        // Drag work estimate
        let profile = &self.profiles[pidx];
        let wind = self.wind.current();
        let v_rel = ((v + v_new) * 0.5) - wind;
        let speed = v_rel.norm();
        let rho = isa_density(x.z).unwrap_or(1.225);
        let a_sound = speed_of_sound(x.z).unwrap_or(340.0);
        let cd = profile.cd_at_mach(speed / a_sound.max(1.0));
        let f_drag = drag_force_magnitude(rho, speed, profile.reference_area_m2(), cd) * drag_mult;
        self.projectiles[index].drag_work_j += f_drag * speed * dt;
        self.projectiles[index].x = x_new;
        self.projectiles[index].v = v_new;
    }

    fn accel_at(
        &self,
        x: Vec3,
        v: Vec3,
        profile_index: usize,
        spin_rad_s: f64,
        mass_kg: f64,
        drag_mult: f64,
    ) -> Vec3 {
        let profile = &self.profiles[profile_index];
        let wind = self.wind.current();
        let v_rel = v - wind;
        let speed = v_rel.norm();
        let rho = isa_density(x.z).unwrap_or(1.225);
        let a_sound = speed_of_sound(x.z).unwrap_or(340.0);
        let mach = speed / a_sound.max(1.0);
        let cd = profile.cd_at_mach(mach);
        let area = profile.reference_area_m2();
        let mut a = self.frame.gravity_accel(x.z, v);
        a += self.frame.coriolis_accel(v);
        if speed > 1e-9 {
            let f = drag_force_magnitude(rho, speed, area, cd) * drag_mult;
            a += v_rel.normalize() * (-f / mass_kg);
        }
        if self.enable_spin_effects && spin_rad_s.abs() > 1e-9 {
            a += spin_drift_accel(v, spin_rad_s, DEFAULT_SPIN_DRIFT_K);
            let axis = if speed > 1e-9 {
                v.normalize()
            } else {
                Vec3::unit_x()
            };
            a += magnus_accel(v, axis, spin_rad_s, rho, DEFAULT_MAGNUS_K);
        }
        let _ = STANDARD_GRAVITY;
        a
    }
}

fn disperse_direction(dir: Vec3, std_rad: f64, rng: &mut Xoshiro256StarStar) -> Vec3 {
    let d = dir.normalize();
    if std_rad <= 0.0 {
        return d;
    }
    // Build orthonormal basis
    let helper = if libm_ops::fabs(d.z) < 0.9 {
        Vec3::unit_z()
    } else {
        Vec3::unit_x()
    };
    let t1 = d.cross(helper).normalize();
    let t2 = d.cross(t1);
    let a = rng.normal() * std_rad;
    let b = rng.normal() * std_rad;
    (d + t1 * a + t2 * b).normalize()
}

/// Convenience: apply hit impulses to a world for all events with `body_id`.
pub fn apply_hit_impulses(world: &mut World, hits: &[HitEvent], mass_kg: f64, e_transfer: f64) {
    for h in hits {
        if let Some(body) = h.body_id {
            apply_impact_impulse(world, body, h.position, mass_kg, h.velocity, e_transfer);
        }
    }
}

/// Export hit for tests.
pub type SegmentHit = Hit;
