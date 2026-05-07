// utils/rust_engine/src/lib.rs
// ── CIPHER v1.0 Rust-Native Engine ──
//
// Exposes CipherEngine as a Python class via PyO3.
// All hot-path kernels are #[inline] in kernels.rs for maximum speed.

use pyo3::prelude::*;

mod kernels;

/// Python-callable wrapper around the Rust math kernels.
#[pyclass(name = "CipherEngine")]
pub struct PyCipherEngine;

#[pymethods]
impl PyCipherEngine {
    /// Normalized min-jerk velocity: v(τ) = 30·τ²·(1-τ)²
    #[staticmethod]
    fn minjerk_vel(tau: f64) -> f64 {
        kernels::minjerk_vel(tau)
    }

    /// Min-jerk position: s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
    #[staticmethod]
    fn minjerk_pos(tau: f64) -> f64 {
        kernels::minjerk_pos(tau)
    }

    /// Adaptive impedance velocity command (returns (vx, vy))
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    fn impedance_vel(
        e_x: f64, e_y: f64,
        vx: f64, vy: f64,
        vref_x: f64, vref_y: f64,
        k: f64, b: f64,
        v_max: f64,
        dz_r: f64,
        dg: f64,
        power: f64,
    ) -> (f64, f64) {
        kernels::impedance_vel(e_x, e_y, vx, vy, vref_x, vref_y, k, b, v_max, dz_r, dg, power)
    }
}

/// Python module entry point.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCipherEngine>()?;
    Ok(())
}
