import unittest

import numpy as np

from test.behavior_evaluator import evaluate_trajectory


class BehaviorEvaluatorTests(unittest.TestCase):
    def test_minimum_jerk_trace_has_one_primary_submovement(self):
        times = np.linspace(0.0, 0.4, 201)
        tau = np.clip((times - 0.1) / 0.25, 0.0, 1.0)
        position = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        points = np.column_stack((240.0 * position, np.zeros_like(position)))

        metrics = evaluate_trajectory(times, points)

        self.assertAlmostEqual(metrics.reaction_time_s, 0.1, delta=0.015)
        self.assertEqual(metrics.submovement_count, 1)
        self.assertTrue(np.isfinite(metrics.dimensionless_jerk))
        self.assertLess(metrics.high_frequency_power_ratio, 0.05)

    def test_reference_kl_is_zero_for_identical_trace(self):
        times = np.linspace(0.0, 0.3, 151)
        points = np.column_stack((times ** 2, np.sin(times)))

        metrics = evaluate_trajectory(times, points, times, points)

        self.assertAlmostEqual(metrics.speed_kl_divergence, 0.0, places=8)
        self.assertAlmostEqual(metrics.jerk_kl_divergence, 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
