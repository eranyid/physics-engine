//! Offline probe: DFSPH dam-break height-field front vs Ritter time curve.
//! `cargo run -p pe-fluid --example probe_dam_ritter --release`

use pe_fluid::{
    dam_break_front_curve_dfsph, dam_break_ritter_geometry, dam_break_times_from_tau,
    dfsph_dam_break_front_and_height_at, DAM_BREAK_TAU_SAMPLES,
};

fn main() {
    let h0 = 0.4;
    let (l_res, channel, g) = dam_break_ritter_geometry(h0);
    let times = dam_break_times_from_tau(h0, g, &DAM_BREAK_TAU_SAMPLES);
    println!("L_res={l_res:.3} channel={channel:.3}");
    for &dx in &[0.05f64, 0.045, 0.04] {
        let curve = dam_break_front_curve_dfsph(dx, h0, l_res, channel, &times);
        println!("dx={dx}");
        for (tau, (t, xf, rel)) in DAM_BREAK_TAU_SAMPLES.iter().zip(curve.iter()) {
            println!("  tau={tau:.1} t={t:.4} xf={xf:.4} rel={rel:.4}");
        }
        let (_xf, frel, hrel, n, _) =
            dfsph_dam_break_front_and_height_at(dx, h0, l_res, times[1], channel);
        println!("  h@tau2: frel={frel:.4} hrel={hrel:.4} n={n}");
    }
}
