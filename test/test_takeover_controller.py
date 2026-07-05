import unittest

import numpy as np

from controllers.pro_controller import PROController


class TakeoverControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = PROController()
        self.controller._rng = np.random.RandomState(7)
        self.controller._ic_reaction_mean = 0.006
        self.controller._ic_reaction_sd = 0.0
        self.controller._ic_reaction_min = 0.006
        self.controller._ic_reaction_max = 0.006

    def test_visibility_edge_starts_one_reaction_and_loss_disarms(self):
        c = self.controller
        c.observe_target(True, np.array([120.0, -40.0]), 0.9)
        generation = c._handoff_generation

        self.assertEqual(c.takeover_state, 'REACTION')
        self.assertTrue(c._target_visible)
        c.observe_target(True, np.array([130.0, -30.0]), 0.8)
        self.assertEqual(c._handoff_generation, generation)

        c.observe_target(False)
        self.assertEqual(c.takeover_state, 'NO_TARGET')
        self.assertFalse(c._target_visible)

    def test_handoff_is_idempotent_and_keeps_only_useful_human_velocity(self):
        c = self.controller
        c.observe_target(True, np.zeros(2), 0.9)
        human_velocity = np.array([420.0, -160.0])
        target_velocity = np.array([-900.0, 0.0])
        error = np.array([300.0, 20.0])
        c.begin_handoff(
            human_velocity=human_velocity,
            target_velocity=target_velocity,
            error=error,
        )
        generation = c._handoff_generation

        radial = error / np.linalg.norm(error)
        target_radial = np.dot(target_velocity, radial)
        reaction_left = max(
            c._ic_reaction_delay - c._ic_reaction_elapsed,
            0.050,
        )
        far_weight = np.clip(
            (np.linalg.norm(error) / c._px_to_ct - 60.0) / 60.0,
            0.0, 1.0,
        )
        scheduled_horizon = c._handoff_open_loop_horizon * (
            1.0 - far_weight * (1.0 - c._handoff_open_loop_far_ratio)
        )
        open_loop_max = (
            c._handoff_open_loop_max
            + far_weight * c._handoff_open_loop_far_max_boost
        )
        open_loop_fraction = (
            c._handoff_open_loop_fraction
            + far_weight * c._handoff_open_loop_far_fraction_boost
        )
        expected_closing = min(
            max(
                0.0,
                np.dot(human_velocity, radial) - target_radial,
                np.linalg.norm(error) / scheduled_horizon,
            ),
            open_loop_max,
            np.linalg.norm(error) * open_loop_fraction / reaction_left,
        )
        expected = (
            target_velocity * c._handoff_target_coast_gain
            + radial * expected_closing
        )
        np.testing.assert_allclose(c._arm_vel, expected)
        c.begin_handoff(
            human_velocity=np.array([1000.0, 1000.0]),
            target_velocity=np.zeros(2),
            error=error,
        )
        self.assertEqual(c._handoff_generation, generation)
        np.testing.assert_allclose(c._arm_vel, expected)

        c._takeover_state = 'ACTIVE_LOCK'
        c.begin_handoff(
            human_velocity=np.array([-2000.0, 500.0]),
            target_velocity=np.zeros(2),
            error=error,
        )
        self.assertEqual(c._handoff_generation, generation)
        np.testing.assert_allclose(c._arm_vel, expected)

    def test_reaction_coasts_selected_handoff_primitive(self):
        c = self.controller
        c._ic_reaction_mean = 0.020
        c._ic_reaction_min = 0.020
        c._ic_reaction_max = 0.020
        c.begin_handoff(
            human_velocity=np.array([400.0, 120.0]),
            target_velocity=np.array([100.0, 50.0]),
            error=np.array([300.0, 0.0]),
        )
        expected = c._handoff_residual_velocity.copy()

        for _ in range(5):
            c.compute(
                300.0, 0.0, 0.002,
                v_real=np.array([100.0, 50.0]),
                a_real=np.zeros(2),
                power_factor=1.0,
                bbox_w=60.0,
            )

        np.testing.assert_allclose(c._arm_vel, expected, atol=1e-9)
        self.assertEqual(c.takeover_state, 'REACTION')

    def test_reaction_revises_velocity_feedforward_with_slew_limit(self):
        c = self.controller
        c._ic_reaction_mean = 0.020
        c._ic_reaction_min = 0.020
        c._ic_reaction_max = 0.020
        c.begin_handoff(
            human_velocity=np.array([500.0, 0.0]),
            target_velocity=np.array([400.0, 0.0]),
            error=np.array([300.0, 0.0]),
        )
        before = c._arm_vel.copy()

        c.compute(
            300.0, 0.0, 0.002,
            v_real=np.array([-400.0, 0.0]),
            a_real=np.zeros(2),
            power_factor=1.0,
            bbox_w=60.0,
        )

        self.assertLess(c._arm_vel[0], before[0])
        self.assertLessEqual(
            np.linalg.norm(c._arm_vel - before) / 0.002,
            c._handoff_max_accel + 1e-6,
        )

    def test_reaction_rotates_frozen_path_after_error_reversal(self):
        c = self.controller
        c._ic_reaction_mean = 0.020
        c._ic_reaction_min = 0.020
        c._ic_reaction_max = 0.020
        c.begin_handoff(
            human_velocity=np.array([500.0, 0.0]),
            target_velocity=np.zeros(2),
            error=np.array([300.0, 0.0]),
        )
        before = c._arm_vel.copy()

        c.compute(
            -300.0, 0.0, 0.002,
            v_real=np.zeros(2),
            a_real=np.zeros(2),
            power_factor=1.0,
            bbox_w=60.0,
        )

        self.assertLess(c._arm_vel[0], before[0])
        self.assertLessEqual(
            np.linalg.norm(c._arm_vel - before) / 0.002,
            c._handoff_max_accel + 1e-6,
        )

    def test_acceleration_guard_survives_active_lock_transition(self):
        c = self.controller
        c._handoff_reason = 'human_release'
        c._takeover_state = 'PRIMING'
        c._handoff_motor_elapsed = 0.0
        c._handoff_residual_velocity[:] = (300.0, 0.0)
        c._handoff_last_output[:] = (300.0, 0.0)

        dt = 0.002
        first = c._apply_handoff_envelope(np.array([1000.0, 0.0]), dt)
        self.assertLessEqual(
            np.linalg.norm(first - np.array([300.0, 0.0])) / dt,
            c._handoff_max_accel + 1e-6,
        )

        c._takeover_state = 'ACTIVE_LOCK'
        second = c._apply_handoff_envelope(np.array([-1000.0, 500.0]), dt)
        self.assertLessEqual(
            np.linalg.norm(second - first) / dt,
            c._handoff_max_accel + 1e-6,
        )

    def test_reaction_commits_latest_observation_to_priming(self):
        c = self.controller
        c.observe_target(True, np.array([200.0, 0.0]), 0.9)
        c._ic_reaction_delay = 0.006

        for _ in range(4):
            c.compute(
                500.0, 40.0, 0.002,
                v_real=np.array([200.0, 0.0]),
                a_real=np.zeros(2),
                power_factor=1.0,
                bbox_w=60.0,
            )

        self.assertIn(c.takeover_state, ('PRIMING', 'ACTIVE_LOCK'))
        self.assertGreater(c._prog_T, 0.0)
        self.assertAlmostEqual(c._prog_init_dx, 500.0)
        self.assertAlmostEqual(c._prog_init_dy, 40.0)

    def test_transverse_velocity_spike_does_not_create_checkmark_path(self):
        c = self.controller
        c.set_pixel_to_count_scale(2.0, 2.0)
        c._ic_reaction_delay = 0.120
        c._ic_reaction_elapsed = 0.0
        c._ou_sigma_ball = 0.0
        c._ou_sigma_track = 0.0
        c._drift_sigma = 0.0
        c._noise_hand[:] = 0.0
        c._noise_fatigue[:] = 0.0

        error_px = np.array([120.0, 0.0])
        position_px = np.zeros(2)
        false_target_velocity = np.array([0.0, 600.0])
        c.begin_handoff(
            human_velocity=np.array([500.0, 300.0]),
            target_velocity=false_target_velocity,
            error=error_px * 2.0,
            reason="human_release",
        )
        c.set_mouse_emit(True)

        path = []
        for frame in range(60):
            target_velocity = (
                false_target_velocity if frame < 12 else np.zeros(2)
            )
            c.compute(
                *(error_px * 2.0), 0.010,
                v_real=target_velocity,
                a_real=np.zeros(2),
                bbox_w=20.0,
            )
            for _ in range(10):
                dx, dy = c.tick_mouse(dt_override=0.001)
                movement_px = np.array([dx, dy], dtype=np.float64) / 2.0
                position_px += movement_px
                error_px -= movement_px
            path.append(position_px.copy())

        path = np.asarray(path)
        self.assertLessEqual(np.max(np.abs(path[:, 1])), 8.0)
        self.assertGreater(path[30, 0], 110.0)


if __name__ == '__main__':
    unittest.main()
