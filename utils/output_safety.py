"""Fail-closed safety gate for all synthetic mouse output."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from config import config


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str
    human_activity: float = 0.0


class OutputSafetyGate:
    """Own the final decision to permit synthetic movement and clicks.

    The gate requires a running/enabled agent and no recent physical mouse
    activity. It deliberately does not inspect windows or processes.
    """

    def __init__(
        self,
        ring_buffer,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._ring_buffer = ring_buffer
        self._clock = clock
        self._lock = threading.Lock()

        self._activity_window_s = max(
            0.005, config.getfloat("Safety", "human_override_window_ms", 30.0) / 1000.0
        )
        self._activity_settle_s = max(
            0.0, config.getfloat("Safety", "human_override_settle_ms", 4.0) / 1000.0
        )
        self._activity_threshold = max(
            0.1, config.getfloat("Safety", "human_override_threshold", 1.5)
        )
        self._override_hold_s = max(
            0.0, config.getfloat("Safety", "human_override_hold_ms", 250.0) / 1000.0
        )

        self._runtime_enabled = False
        self._override_until = 0.0

    def set_runtime_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._runtime_enabled = bool(enabled)

    def evaluate(self, now: Optional[float] = None) -> SafetyDecision:
        now = self._clock() if now is None else float(now)
        with self._lock:
            activity = 0.0
            if self._ring_buffer is not None:
                settled_end = now - self._activity_settle_s
                settled_start = settled_end - self._activity_window_s
                activity = self._ring_buffer.get_intent_activity(
                    settled_start, settled_end
                )
                if activity >= self._activity_threshold:
                    self._override_until = max(
                        self._override_until, now + self._override_hold_s
                    )

            if not self._runtime_enabled:
                return SafetyDecision(False, "agent_disabled", activity)
            if now < self._override_until:
                return SafetyDecision(False, "human_override", activity)
            return SafetyDecision(True, "allowed", activity)
