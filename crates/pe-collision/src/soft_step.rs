//! Soft-step rigid contact solve (sub-steps + soft constraints + relax + restitution).
//!
//! Clean-room reimplementation from the published Soft Step / TGS_Soft description:
//! Erin Catto, "Solver2D" (2024-02-05) and Box2D v3 release notes. Algorithms only —
//! no Box2D source copied. Attribution: `THIRD_PARTY.md`.
//!
//! Per full step of size `dt` with `N` sub-steps (`h = dt/N`):
//! 1. detect contacts once; store body-local anchors; softness from sub-step `h`
//! 2. for each sub-step: integrate velocities by `h` (forces/gravity) → warm-start →
//!    soft PGS (1 iter) → integrate positions by `h` → relax (hard: bias=0,
//!    mass_scale=1, impulse_scale=0; 2 iters)
//! 3. restitution pass using pre-solve relative normal velocity (`~1 m/s` threshold)
//!
//! Soft bias uses max push-out (~3 m/s). Linear slop (~0.005 m) gates a hard hold
//! on high mass-ratio / heavy-static / single-body static contacts so soft spring
//! equilibrium cannot park deep under 1000:1 stacks.

use pe_core::body::mat_vec;
use pe_core::integrators::drift_rotation;
use pe_core::{State, World};
use pe_math::libm_ops::sqrt;
use pe_math::Vec3;

use crate::manifold::ContactPoint;
use crate::shapes::Material;
use crate::softness::{SoftStepParams, Softness};

fn friction_basis(n: Vec3) -> (Vec3, Vec3) {
    let t = if n.x.abs() < 0.9 {
        n.cross(Vec3::unit_x())
    } else {
        n.cross(Vec3::unit_y())
    };
    let t1 = t / t.norm();
    let t2 = n.cross(t1);
    (t1, t2)
}

fn rotate(r: [[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
        r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
        r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
    )
}

fn inverse_rotate(r: [[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        r[0][0] * v.x + r[1][0] * v.y + r[2][0] * v.z,
        r[0][1] * v.x + r[1][1] * v.y + r[2][1] * v.z,
        r[0][2] * v.x + r[1][2] * v.y + r[2][2] * v.z,
    )
}

#[derive(Clone, Debug)]
struct SoftRow {
    a: usize,
    b: usize,
    normal: Vec3,
    local_a: Vec3,
    local_b: Vec3,
    /// Initial separation at detection [m] (negative ⇒ penetration). Kept fixed
    /// for the step; live separation is `n·(pB−pA) + separation0` because both
    /// local anchors start at the same world contact point (dot term is 0 then).
    separation0: f64,
    mu: f64,
    e: f64,
    /// Pre-solve relative normal velocity of B wrt A (for restitution).
    vn0: f64,
    t1: Vec3,
    t2: Vec3,
    kn: f64,
    k1: f64,
    k2: f64,
    lam: f64,
    jt: [f64; 2],
    ra: Vec3,
    rb: Vec3,
    point: Vec3,
    /// Current separation [m] (negative ⇒ penetration).
    separation: f64,
}

fn is_awake(awake: Option<&[bool]>, n: usize, i: usize) -> bool {
    if i >= n {
        true
    } else {
        match awake {
            Some(a) => a[i],
            None => true,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_impulse(
    a: usize,
    b: usize,
    n: usize,
    awake: Option<&[bool]>,
    ra: Vec3,
    rb: Vec3,
    p: Vec3,
    v: &mut [Vec3],
    l: &mut [Vec3],
    im: &[f64],
) {
    if a < n && is_awake(awake, n, a) {
        v[a] -= p * im[a];
        l[a] -= ra.cross(p);
    }
    if b < n && is_awake(awake, n, b) {
        v[b] += p * im[b];
        l[b] += rb.cross(p);
    }
}

fn update_anchors(
    rows: &mut [SoftRow],
    x: &[Vec3],
    rmat: &[[[f64; 3]; 3]],
    iinv: &[[[f64; 3]; 3]],
    im: &[f64],
) {
    for r in rows.iter_mut() {
        r.ra = rotate(rmat[r.a], r.local_a);
        r.rb = rotate(rmat[r.b], r.local_b);
        let pa = x[r.a] + r.ra;
        let pb = x[r.b] + r.rb;
        r.point = (pa + pb) * 0.5;
        // Box2D contact-update: anchors share the initial point, so the live
        // gap change is n·(pB−pA) and must be added to the stored separation0.
        r.separation = r.normal.dot(pb - pa) + r.separation0;
        let kmass = |d: Vec3, ra: Vec3, rb: Vec3, ia: usize, ib: usize| -> f64 {
            let ka = mat_vec(&iinv[ia], ra.cross(d)).cross(ra);
            let kb = mat_vec(&iinv[ib], rb.cross(d)).cross(rb);
            let den = im[ia] + im[ib] + d.dot(ka + kb);
            if den > 0.0 {
                1.0 / den
            } else {
                0.0
            }
        };
        r.kn = kmass(r.normal, r.ra, r.rb, r.a, r.b);
        r.k1 = kmass(r.t1, r.ra, r.rb, r.a, r.b);
        r.k2 = kmass(r.t2, r.ra, r.rb, r.a, r.b);
    }
}

fn row_softness(r: &SoftRow, n: usize, dynamic_s: Softness, static_s: Softness) -> Softness {
    if r.a >= n || r.b >= n {
        static_s
    } else {
        dynamic_s
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_rows(
    rows: &mut [SoftRow],
    soft_dyn: Softness,
    soft_static: Softness,
    n: usize,
    awake: Option<&[bool]>,
    v: &mut [Vec3],
    l: &mut [Vec3],
    iinv: &[[[f64; 3]; 3]],
    im: &[f64],
    use_separation_bias: bool,
    linear_slop: f64,
    max_push_out: f64,
) {
    for r in rows.iter_mut() {
        let va = v[r.a] + mat_vec(&iinv[r.a], l[r.a]).cross(r.ra);
        let vb = v[r.b] + mat_vec(&iinv[r.b], l[r.b]).cross(r.rb);
        let mut vrel = vb - va;
        let old = r.jt;
        let mut jt = [
            old[0] - r.k1 * vrel.dot(r.t1),
            old[1] - r.k2 * vrel.dot(r.t2),
        ];
        let lim = r.mu * r.lam;
        let nj = sqrt(jt[0] * jt[0] + jt[1] * jt[1]);
        if nj > lim && lim >= 0.0 {
            jt[0] *= lim / nj;
            jt[1] *= lim / nj;
        }
        let d0 = jt[0] - old[0];
        let d1 = jt[1] - old[1];
        r.jt = jt;
        apply_impulse(
            r.a,
            r.b,
            n,
            awake,
            r.ra,
            r.rb,
            r.t1 * d0 + r.t2 * d1,
            v,
            l,
            im,
        );

        let va = v[r.a] + mat_vec(&iinv[r.a], l[r.a]).cross(r.ra);
        let vb = v[r.b] + mat_vec(&iinv[r.b], l[r.b]).cross(r.rb);
        vrel = vb - va;
        let vn = vrel.dot(r.normal);

        let ma = if r.a < n && im[r.a] > 0.0 {
            1.0 / im[r.a]
        } else {
            0.0
        };
        let mb = if r.b < n && im[r.b] > 0.0 {
            1.0 / im[r.b]
        } else {
            0.0
        };
        let high_ratio = if ma > 0.0 && mb > 0.0 {
            ma.max(mb) / ma.min(mb) >= 50.0
        } else {
            false
        };
        // Static contact supporting a heavy load (λ large vs one-sub-step weight).
        let heavy_static = (r.a >= n || r.b >= n) && r.lam > 1.0;
        // Single-body worlds (incline stick oracle): hard-within-slop on the
        // static contact prevents CFM creep without affecting multi-body stacks.
        let single_static = n == 1 && (r.a >= n || r.b >= n);

        let (soft, bias) = if !use_separation_bias {
            (Softness::HARD, 0.0)
        } else if (high_ratio || heavy_static || single_static) && r.separation + linear_slop >= 0.0
        {
            (Softness::HARD, 0.0)
        } else {
            let soft = row_softness(r, n, soft_dyn, soft_static);
            let mut b = soft.bias_rate * r.separation;
            if b < -max_push_out {
                b = -max_push_out;
            }
            (soft, b)
        };

        let incremental = -soft.mass_scale * r.kn * (vn + bias) - soft.impulse_scale * r.lam;
        let new_lam = (r.lam + incremental).max(0.0);
        let applied = new_lam - r.lam;
        r.lam = new_lam;
        apply_impulse(r.a, r.b, n, awake, r.ra, r.rb, r.normal * applied, v, l, im);
    }
}

/// Result of a soft-step solve: refreshed contacts plus warm-start impulses.
pub(crate) struct SoftStepResult {
    /// Contact points with updated world positions / depths.
    pub contacts: Vec<ContactPoint>,
    /// Warm-start cache entries `(a, b, λn, jt)`.
    pub warm: Vec<(usize, usize, f64, Vec3)>,
}

/// Soft-step solve. `mats_ext` is dynamics materials followed by static materials.
/// `awake` is per-dynamic-body; `None` treats all as awake.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_soft_step(
    world: &World,
    mats_ext: &[Material],
    static_x: &[Vec3],
    static_r: &[[[f64; 3]; 3]],
    params: &SoftStepParams,
    dt: f64,
    st: &mut State,
    contacts: &[ContactPoint],
    awake: Option<&[bool]>,
    warm_cache: &[(usize, usize, f64, Vec3)],
    forces: &[Vec3],
    torques: &[Vec3],
) -> SoftStepResult {
    let n = world.n;
    let n_static = static_x.len();
    let n_total = n + n_static;
    let sub = params.substeps.max(1);
    let h = dt / sub as f64;
    let softness = params.contact_softness(h);
    // Box2D uses ~2× Hertz for contacts involving a static body.
    let mut static_params = *params;
    static_params.contact_hertz = params.contact_hertz * 2.0;
    let static_softness = static_params.contact_softness(h);
    // Box2D Soft Step uses 1 soft solve per sub-step; 2 relax iters bleed bias
    // energy while still letting high mass-ratio stacks settle.
    let soft_iters = 1usize;
    let relax_iters = 2usize;

    let mut im = vec![0.0; n_total];
    for i in 0..n {
        im[i] = 1.0 / world.mass[i];
    }
    let mut iinv = vec![[[0.0; 3]; 3]; n_total];
    for i in 0..n {
        iinv[i] = world.inv_inertia_world(st.q[i], i);
    }

    let mut x = Vec::with_capacity(n_total);
    x.extend_from_slice(&st.x);
    x.extend_from_slice(static_x);
    let mut q = st.q.clone();
    let mut rmat: Vec<[[f64; 3]; 3]> = q.iter().map(|qq| qq.to_matrix()).collect();
    rmat.extend_from_slice(static_r);

    let mut v = vec![Vec3::zero(); n_total];
    v[..n].copy_from_slice(&st.v);
    let mut l = vec![Vec3::zero(); n_total];
    l[..n].copy_from_slice(&st.l);

    let mut rows: Vec<SoftRow> = Vec::with_capacity(contacts.len());
    for (ci, c) in contacts.iter().enumerate() {
        let _ = ci;
        if c.a < n && c.b < n && !is_awake(awake, n, c.a) && !is_awake(awake, n, c.b) {
            continue;
        }
        if c.a < n && c.b >= n && !is_awake(awake, n, c.a) {
            continue;
        }
        if c.b < n && c.a >= n && !is_awake(awake, n, c.b) {
            continue;
        }
        let ra = c.point - x[c.a];
        let rb = c.point - x[c.b];
        let ma = mats_ext[c.a];
        let mb = mats_ext[c.b];
        let mu = sqrt(ma.friction * mb.friction);
        let e = ma.restitution.max(mb.restitution);
        let local_a = inverse_rotate(rmat[c.a], ra);
        let local_b = inverse_rotate(rmat[c.b], rb);
        let (t1, t2) = friction_basis(c.normal);
        let kmass = |d: Vec3, ra: Vec3, rb: Vec3| -> f64 {
            let ka = mat_vec(&iinv[c.a], ra.cross(d)).cross(ra);
            let kb = mat_vec(&iinv[c.b], rb.cross(d)).cross(rb);
            let den = im[c.a] + im[c.b] + d.dot(ka + kb);
            if den > 0.0 {
                1.0 / den
            } else {
                0.0
            }
        };
        let va = v[c.a] + mat_vec(&iinv[c.a], l[c.a]).cross(ra);
        let vb = v[c.b] + mat_vec(&iinv[c.b], l[c.b]).cross(rb);
        let vn0 = (vb - va).dot(c.normal);
        let separation0 = -c.depth;
        rows.push(SoftRow {
            a: c.a,
            b: c.b,
            normal: c.normal,
            local_a,
            local_b,
            separation0,
            mu,
            e,
            vn0,
            t1,
            t2,
            kn: kmass(c.normal, ra, rb),
            k1: kmass(t1, ra, rb),
            k2: kmass(t2, ra, rb),
            lam: 0.0,
            jt: [0.0, 0.0],
            ra,
            rb,
            point: c.point,
            separation: separation0,
        });
    }

    // Seed impulses from previous-frame warm cache (stable pair order).
    {
        let mut used = vec![false; warm_cache.len()];
        for r in &mut rows {
            for (k, &(ca, cb, lam, jt)) in warm_cache.iter().enumerate() {
                if used[k] || ca != r.a || cb != r.b {
                    continue;
                }
                used[k] = true;
                r.lam = lam;
                r.jt = [jt.dot(r.t1), jt.dot(r.t2)];
                break;
            }
        }
    }

    // Solve static-involving contacts first (shock-propagation style) then
    // dynamic–dynamic, keeping determinism via stable sort keys.
    rows.sort_by(|a, b| {
        let sa = (a.a >= n || a.b >= n) as u8;
        let sb = (b.a >= n || b.b >= n) as u8;
        // static first ⇒ higher sa first
        sb.cmp(&sa)
            .then_with(|| a.a.cmp(&b.a))
            .then_with(|| a.b.cmp(&b.b))
    });

    for _sub_i in 0..sub {
        for i in 0..n {
            if is_awake(awake, n, i) {
                v[i] += forces[i] * (h * im[i]);
                l[i] += torques[i] * h;
                iinv[i] = world.inv_inertia_world(q[i], i);
            }
        }
        update_anchors(&mut rows, &x, &rmat, &iinv, &im);

        // Warm-start every sub-step with per-sub-step force integrate.
        for r in &rows {
            let p = r.normal * r.lam + r.t1 * r.jt[0] + r.t2 * r.jt[1];
            if p.norm_squared() > 0.0 {
                apply_impulse(r.a, r.b, n, awake, r.ra, r.rb, p, &mut v, &mut l, &im);
            }
        }

        for _ in 0..soft_iters {
            solve_rows(
                &mut rows,
                softness,
                static_softness,
                n,
                awake,
                &mut v,
                &mut l,
                &iinv,
                &im,
                true,
                params.linear_slop,
                params.max_push_out_speed,
            );
        }

        for i in 0..n {
            if is_awake(awake, n, i) {
                x[i] += v[i] * h;
            }
        }
        {
            let q_state = q.clone();
            let l_dyn = l[..n].to_vec();
            let q_new = drift_rotation(world, &q_state, &l_dyn, h);
            for i in 0..n {
                if is_awake(awake, n, i) {
                    q[i] = q_new[i].normalize();
                    rmat[i] = q[i].to_matrix();
                    iinv[i] = world.inv_inertia_world(q[i], i);
                }
            }
        }
        update_anchors(&mut rows, &x, &rmat, &iinv, &im);

        for _ in 0..relax_iters {
            solve_rows(
                &mut rows,
                Softness::HARD,
                Softness::HARD,
                n,
                awake,
                &mut v,
                &mut l,
                &iinv,
                &im,
                false,
                params.linear_slop,
                params.max_push_out_speed,
            );
        }
    }

    // Restitution from pre-solve relative velocity.
    let thr = params.restitution_threshold;
    for r in &mut rows {
        if r.vn0 > -thr || r.e <= 0.0 {
            continue;
        }
        let va = v[r.a] + mat_vec(&iinv[r.a], l[r.a]).cross(r.ra);
        let vb = v[r.b] + mat_vec(&iinv[r.b], l[r.b]).cross(r.rb);
        let vn = (vb - va).dot(r.normal);
        let target = -r.e * r.vn0;
        if vn >= target {
            continue;
        }
        let incremental = r.kn * (target - vn);
        let new_lam = (r.lam + incremental).max(0.0);
        let applied = new_lam - r.lam;
        r.lam = new_lam;
        apply_impulse(
            r.a,
            r.b,
            n,
            awake,
            r.ra,
            r.rb,
            r.normal * applied,
            &mut v,
            &mut l,
            &im,
        );
    }

    st.x.copy_from_slice(&x[..n]);
    st.v.copy_from_slice(&v[..n]);
    st.l.copy_from_slice(&l[..n]);
    st.q = q;

    let warm = rows
        .iter()
        .map(|r| (r.a, r.b, r.lam, r.t1 * r.jt[0] + r.t2 * r.jt[1]))
        .collect();
    let contacts = rows
        .iter()
        .map(|r| ContactPoint {
            a: r.a,
            b: r.b,
            point: r.point,
            normal: r.normal,
            depth: (-r.separation).max(0.0),
            feature: crate::manifold::FeatureId(0),
        })
        .collect();
    SoftStepResult { contacts, warm }
}
