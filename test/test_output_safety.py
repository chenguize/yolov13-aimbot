import time
import unittest

from perception.ring_buffer import RingBuffer
from utils.output_safety import OutputSafetyGate


class _Clock:
    def __init__(self, value=1.0):
        self.value = value

    def __call__(self):
        return self.value


class _ActivityBuffer:
    def __init__(self):
        self.activity = 0.0

    def get_intent_activity(self, _start, _end):
        return self.activity


class OutputSafetyGateTests(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.buffer = _ActivityBuffer()

    def _gate(self):
        gate = OutputSafetyGate(self.buffer, clock=self.clock)
        gate.set_runtime_enabled(True)
        return gate

    def test_enabled_agent_opens_gate_without_window_checks(self):
        decision = self._gate().evaluate()
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "allowed")

    def test_human_activity_latches_override_without_direction_check(self):
        gate = self._gate()
        self.buffer.activity = 2.0
        self.assertEqual(gate.evaluate().reason, "human_override")

        self.buffer.activity = 0.0
        self.clock.value += 0.10
        self.assertEqual(gate.evaluate().reason, "human_override")
        self.clock.value += 0.20
        self.assertTrue(gate.evaluate().allowed)

    def test_disabled_agent_cannot_open_output(self):
        gate = self._gate()
        gate.set_runtime_enabled(False)
        self.assertEqual(gate.evaluate().reason, "agent_disabled")


class RingBufferActivityTests(unittest.TestCase):
    def test_pynput_activity_subtracts_ai_echo_by_path_length(self):
        buffer = RingBuffer()
        buffer._sub_ai_checked = True
        buffer._sub_ai_enabled = True
        buffer.add_event(6, 0, is_ai=True)
        buffer.add_event(10, 0, is_ai=False)
        now = time.perf_counter()
        self.assertAlmostEqual(buffer.get_intent_activity(now - 0.1, now), 4.0)

    def test_reconciled_ai_echo_is_not_human_or_double_counted(self):
        buffer = RingBuffer()
        buffer.enable_observed_echo_reconciliation(True)
        buffer.add_event(7, -3, is_ai=True)
        buffer.add_observed_cursor_event(7, -3)
        now = time.perf_counter()

        self.assertEqual(buffer.get_intent_delta(now - 0.1, now), (0, 0))
        self.assertEqual(buffer.get_total_delta_sum(now - 0.1, now), (7, -3))
        self.assertEqual(buffer.get_intent_activity(now - 0.1, now), 0.0)

    def test_reconciled_event_preserves_physical_residual(self):
        buffer = RingBuffer()
        buffer.enable_observed_echo_reconciliation(True)
        buffer.add_event(5, 0, is_ai=True)
        buffer.add_observed_cursor_event(8, -2)
        now = time.perf_counter()

        self.assertEqual(buffer.get_intent_delta(now - 0.1, now), (3, -2))
        self.assertEqual(buffer.get_total_delta_sum(now - 0.1, now), (8, -2))

    def test_reconciliation_handles_coalesced_ai_callbacks(self):
        buffer = RingBuffer()
        buffer.enable_observed_echo_reconciliation(True)
        buffer.add_event(2, 1, is_ai=True)
        buffer.add_event(3, 2, is_ai=True)
        buffer.add_observed_cursor_event(5, 3)
        now = time.perf_counter()

        self.assertEqual(buffer.get_intent_delta(now - 0.1, now), (0, 0))
        self.assertEqual(buffer.get_total_delta_sum(now - 0.1, now), (5, 3))


if __name__ == "__main__":
    unittest.main()
