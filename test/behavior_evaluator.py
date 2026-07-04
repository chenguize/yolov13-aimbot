"""Kinematic evaluation utilities for simulated and recorded cursor traces."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class BehaviorMetrics:
    reaction_time_s: float
    movement_time_s: float
    peak_speed: float
    peak_acceleration: float
    dimensionless_jerk: float
    submovement_count: int
    median_submovement_interval_s: float
    psd_centroid_hz: float
    high_frequency_power_ratio: float
    speed_kl_divergence: Optional[float] = None
    jerk_kl_divergence: Optional[float] = None


def _sanitize_trace(times, positions):
    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(positions, dtype=np.float64)
    if t.ndim != 1 or p.ndim != 2 or p.shape != (len(t), 2):
        raise ValueError("positions must have shape (len(times), 2)")
    if len(t) < 8 or not np.all(np.diff(t) > 0.0):
        raise ValueError("trace needs at least 8 strictly increasing samples")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)):
        raise ValueError("trace contains non-finite values")
    return t, p


def _derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.gradient(values, times, axis=0, edge_order=2)


def _peak_indices(speed: np.ndarray, times: np.ndarray) -> np.ndarray:
    threshold = max(1e-9, float(np.max(speed)) * 0.12)
    candidates = np.flatnonzero(
        (speed[1:-1] > speed[:-2])
        & (speed[1:-1] >= speed[2:])
        & (speed[1:-1] >= threshold)
    ) + 1
    if not len(candidates):
        return np.array([int(np.argmax(speed))], dtype=np.int64)

    selected = [int(candidates[0])]
    for index in candidates[1:]:
        if times[index] - times[selected[-1]] >= 0.040:
            selected.append(int(index))
        elif speed[index] > speed[selected[-1]]:
            selected[-1] = int(index)
    return np.asarray(selected, dtype=np.int64)


def _distribution_kl(sample, reference, bins=48) -> float:
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    combined = np.concatenate((sample[np.isfinite(sample)], reference[np.isfinite(reference)]))
    if len(combined) < 4 or float(np.ptp(combined)) <= 1e-12:
        return 0.0
    edges = np.linspace(float(np.min(combined)), float(np.max(combined)), bins + 1)
    p, _ = np.histogram(sample, bins=edges, density=False)
    q, _ = np.histogram(reference, bins=edges, density=False)
    eps = 1e-9
    p = (p + eps) / (np.sum(p) + eps * bins)
    q = (q + eps) / (np.sum(q) + eps * bins)
    return float(np.sum(p * np.log(p / q)))


def evaluate_trajectory(
        times,
        positions,
        reference_times=None,
        reference_positions=None,
        high_frequency_hz: float = 10.0,
) -> BehaviorMetrics:
    """Evaluate one cursor trajectory without assuming a human reference."""
    t, p = _sanitize_trace(times, positions)
    velocity = _derivative(p, t)
    acceleration = _derivative(velocity, t)
    jerk = _derivative(acceleration, t)
    speed = np.linalg.norm(velocity, axis=1)
    accel_norm = np.linalg.norm(acceleration, axis=1)
    jerk_norm = np.linalg.norm(jerk, axis=1)

    peak_speed = float(np.max(speed))
    onset_threshold = max(1e-6, peak_speed * 0.01)
    moving = speed >= onset_threshold
    onset = 0
    for index in range(len(speed) - 2):
        if np.all(moving[index:index + 3]):
            onset = index
            break

    peaks = _peak_indices(speed[onset:], t[onset:]) + onset
    intervals = np.diff(t[peaks])
    distance = float(np.linalg.norm(p[-1] - p[onset]))
    duration = max(float(t[-1] - t[onset]), 1e-6)
    jerk_integral = float(np.trapz(jerk_norm * jerk_norm, t))
    dimensionless_jerk = (
        duration ** 5 * jerk_integral / max(distance * distance, 1e-9)
    )

    dt = float(np.median(np.diff(t)))
    centered_speed = speed - float(np.mean(speed))
    frequencies = np.fft.rfftfreq(len(centered_speed), d=dt)
    power = np.abs(np.fft.rfft(centered_speed)) ** 2
    power_sum = float(np.sum(power[1:]))
    if power_sum <= 1e-12:
        centroid = 0.0
        hf_ratio = 0.0
    else:
        centroid = float(np.sum(frequencies[1:] * power[1:]) / power_sum)
        hf_ratio = float(np.sum(power[frequencies >= high_frequency_hz]) / power_sum)

    speed_kl = jerk_kl = None
    if reference_times is not None and reference_positions is not None:
        rt, rp = _sanitize_trace(reference_times, reference_positions)
        ref_velocity = _derivative(rp, rt)
        ref_accel = _derivative(ref_velocity, rt)
        ref_jerk = _derivative(ref_accel, rt)
        ref_speed = np.linalg.norm(ref_velocity, axis=1)
        speed_kl = _distribution_kl(
            speed / max(peak_speed, 1e-9),
            ref_speed / max(float(np.max(ref_speed)), 1e-9),
        )
        jerk_kl = _distribution_kl(jerk_norm, np.linalg.norm(ref_jerk, axis=1))

    return BehaviorMetrics(
        reaction_time_s=float(t[onset] - t[0]),
        movement_time_s=duration,
        peak_speed=peak_speed,
        peak_acceleration=float(np.max(accel_norm)),
        dimensionless_jerk=dimensionless_jerk,
        submovement_count=int(len(peaks)),
        median_submovement_interval_s=(
            float(np.median(intervals)) if len(intervals) else 0.0
        ),
        psd_centroid_hz=centroid,
        high_frequency_power_ratio=hf_ratio,
        speed_kl_divergence=speed_kl,
        jerk_kl_divergence=jerk_kl,
    )
