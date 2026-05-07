// utils/rust_engine/src/kernels.rs
// ═══════════════════════════════════════════════════════════════════════════════
// Pure-math kernels — no state, no side effects, #[inline] for maximum speed.
// ═══════════════════════════════════════════════════════════════════════════════

/// Normalized min-jerk velocity: v(τ) = 30·τ²·(1-τ)²
/// ∫₀¹ v(τ)dτ = 1.0, peak = 1.875 at τ = 0.5
#[inline]
pub fn minjerk_vel(tau: f64) -> f64 {
    let t = tau.clamp(0.0, 1.0);
    30.0 * t * t * (1.0 - t) * (1.0 - t)
}

/// Min-jerk position: s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
/// s(0)=0, s(1)=1
#[inline]
pub fn minjerk_pos(tau: f64) -> f64 {
    let t = tau.clamp(0.0, 1.0);
    ((6.0 * t - 15.0) * t + 10.0) * t * t * t
}

/// Adaptive impedance velocity: v_des = (K·pos_gain·e - B·(v - v_ref)) · dg · power
///
/// Soft deadzone: pos_gain ramps quadratically 0→1 over [dz_r, 3·dz_r]
#[inline]
pub fn impedance_vel(
    e_x: f64, e_y: f64,
    vx: f64, vy: f64,
    vref_x: f64, vref_y: f64,
    k: f64, b: f64,
    v_max: f64,
    dz_r: f64,
    dg: f64,
    power: f64,
) -> (f64, f64) {
    let dist = (e_x * e_x + e_y * e_y).sqrt() + 1e-9;

    let pos_gain = if dist < dz_r {
        0.0
    } else if dist < dz_r * 3.0 {
        let t = (dist - dz_r) / (dz_r * 2.0);
        t * t
    } else {
        1.0
    };

    let vxr = vx - vref_x;
    let vyr = vy - vref_y;

    let mut vcx = (k * e_x * pos_gain - b * vxr) * dg * power;
    let mut vcy = (k * e_y * pos_gain - b * vyr) * dg * power;

    let spd = (vcx * vcx + vcy * vcy).sqrt();
    if spd > v_max {
        let sc = v_max / spd;
        vcx *= sc;
        vcy *= sc;
    }

    (vcx, vcy)
}
