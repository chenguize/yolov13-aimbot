"""Fail-closed safety gate for all synthetic mouse output."""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from config import config


def get_foreground_window_title() -> str:
    """Return the active window title, or an empty string on any Win32 failure."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        if user32.GetWindowTextW(hwnd, buffer, len(buffer)) <= 0:
            return ""
        return buffer.value.strip()
    except Exception:
        return ""


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str
    foreground_title: str = ""
    human_activity: float = 0.0


class OutputSafetyGate:
    """Own the final decision to permit synthetic movement and clicks.

    The gate fails closed. It requires an explicitly allowed foreground window,
    a running/enabled agent, and no recent physical mouse activity.
    """

    def __init__(
        self,
        ring_buffer,
        *,
        foreground_reader: Optional[Callable[[], str]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._ring_buffer = ring_buffer
        self._foreground_reader = foreground_reader or get_foreground_window_title
        self._clock = clock
        self._lock = threading.Lock()

        self._foreground_guard = config.getbool("Safety", "foreground_guard_enable", True)
        self._allowed_titles = tuple(
            token.casefold()
            for token in config.getlist(
                "Safety", "foreground_allowed_titles", ["VALORANT", "Aim Lab"]
            )
            if token
        )
        self._foreground_poll_s = max(
            0.01, config.getfloat("Safety", "foreground_poll_ms", 40.0) / 1000.0
        )
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
        self._foreground_title = ""
        self._foreground_allowed = not self._foreground_guard
        self._next_foreground_poll = 0.0
        self._override_until = 0.0

    def set_runtime_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._runtime_enabled = bool(enabled)

    def _poll_foreground(self, now: float) -> None:
        if not self._foreground_guard:
            self._foreground_allowed = True
            return
        if now < self._next_foreground_poll:
            return
        self._next_foreground_poll = now + self._foreground_poll_s
        try:
            title = (self._foreground_reader() or "").strip()
        except Exception:
            title = ""
        folded = title.casefold()
        self._foreground_title = title
        self._foreground_allowed = bool(
            folded and self._allowed_titles
            and any(token in folded for token in self._allowed_titles)
        )

    def evaluate(self, now: Optional[float] = None) -> SafetyDecision:
        now = self._clock() if now is None else float(now)
        with self._lock:
            self._poll_foreground(now)

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
                return SafetyDecision(False, "agent_disabled", self._foreground_title, activity)
            if not self._foreground_allowed:
                return SafetyDecision(False, "foreground_blocked", self._foreground_title, activity)
            if now < self._override_until:
                return SafetyDecision(False, "human_override", self._foreground_title, activity)
            return SafetyDecision(True, "allowed", self._foreground_title, activity)
