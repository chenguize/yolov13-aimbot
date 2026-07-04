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

    def _gate(self, title_reader):
        gate = OutputSafetyGate(
            self.buffer,
            foreground_reader=title_reader,
            clock=self.clock,
        )
        gate.set_runtime_enabled(True)
        return gate

    def test_desktop_is_blocked_even_when_agent_is_enabled(self):
        decision = self._gate(lambda: "Program Manager").evaluate()
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "foreground_blocked")

    def test_allowed_game_window_opens_gate(self):
        decision = self._gate(lambda: "VALORANT  ").evaluate()
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "allowed")

    def test_foreground_reader_failure_fails_closed(self):
        def broken_reader():
            raise OSError("window API unavailable")

        decision = self._gate(broken_reader).evaluate()
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "foreground_blocked")

    def test_human_activity_latches_override_without_direction_check(self):
        gate = self._gate(lambda: "Aim Lab")
        self.buffer.activity = 2.0
        self.assertEqual(gate.evaluate().reason, "human_override")

        self.buffer.activity = 0.0
        self.clock.value += 0.10
        self.assertEqual(gate.evaluate().reason, "human_override")
        self.clock.value += 0.20
        self.assertTrue(gate.evaluate().allowed)

    def test_disabled_agent_cannot_open_output(self):
        gate = self._gate(lambda: "VALORANT")
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


if __name__ == "__main__":
    unittest.main()
