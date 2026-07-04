# perception/ring_buffer.py
"""
鼠标事件环形缓冲区。

职责：
- 收集人手 (HumanMouseListener, is_ai=False) 和 AI (gHub SendInput, is_ai=True) 的位移事件
- 提供时间窗口内的位移汇总查询
- pynput 逐事件核销 SendInput 回显；inputs 保持独立物理轴
- 为 WorldModel 提供只计一次的人+AI 总位移
"""
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple


@dataclass
class InputEvent:
    timestamp: float
    dx: int
    dy: int
    is_ai: bool


class RingBuffer:
    def __init__(self, max_duration: float = 2.0):
        self._buffer: deque[InputEvent] = deque()
        self._pending_ai_echo: deque[InputEvent] = deque()
        self._lock = threading.Lock()
        self.max_duration = max_duration
        self._sub_ai_enabled: bool = False  # 由 use_subtract_ai() 延迟初始化
        self._observed_events_reconciled = False
        self._echo_match_window = 0.050

    # ── 写入 ──────────────────────────────────────────────────────────────

    def add_event(self, dx: int, dy: int, is_ai: bool):
        if dx == 0 and dy == 0:
            return
        now = time.perf_counter()
        ev = InputEvent(timestamp=now, dx=dx, dy=dy, is_ai=is_ai)
        with self._lock:
            self._buffer.append(ev)
            if is_ai and self._observed_events_reconciled:
                self._pending_ai_echo.append(
                    InputEvent(timestamp=now, dx=dx, dy=dy, is_ai=True)
                )
            while self._buffer and (now - self._buffer[0].timestamp > self.max_duration):
                self._buffer.popleft()

    def enable_observed_echo_reconciliation(self, enabled: bool = True) -> None:
        """Mark cursor-hook events as a mixed physical + injected channel."""
        with self._lock:
            self._observed_events_reconciled = bool(enabled)
            self._pending_ai_echo.clear()

    @staticmethod
    def _consume_axis(observed: int, injected: int) -> Tuple[int, int]:
        if observed == 0 or injected == 0 or (observed > 0) != (injected > 0):
            return observed, injected
        amount = min(abs(observed), abs(injected))
        sign = 1 if observed > 0 else -1
        return observed - sign * amount, injected - sign * amount

    def add_observed_cursor_event(self, dx: int, dy: int) -> None:
        """Record a pynput event after removing matching SendInput echo.

        SendInput commands are registered before dispatch. Cursor-hook events
        can be split or coalesced, so matching consumes each axis across all
        recent pending commands and records only the unexplained residual as
        physical human input.
        """
        if dx == 0 and dy == 0:
            return
        now = time.perf_counter()
        residual_x, residual_y = int(dx), int(dy)
        with self._lock:
            cutoff = now - self._echo_match_window
            while self._pending_ai_echo and self._pending_ai_echo[0].timestamp < cutoff:
                self._pending_ai_echo.popleft()

            for echo in self._pending_ai_echo:
                residual_x, echo.dx = self._consume_axis(residual_x, echo.dx)
                residual_y, echo.dy = self._consume_axis(residual_y, echo.dy)
                if residual_x == 0 and residual_y == 0:
                    break
            self._pending_ai_echo = deque(
                echo for echo in self._pending_ai_echo if echo.dx or echo.dy
            )

            if residual_x or residual_y:
                self._buffer.append(InputEvent(
                    timestamp=now,
                    dx=residual_x,
                    dy=residual_y,
                    is_ai=False,
                ))
            while self._buffer and (now - self._buffer[0].timestamp > self.max_duration):
                self._buffer.popleft()

    # ── 读数：基础通道 ────────────────────────────────────────────────────

    def get_cursor_delta_sum(self, t_start: float, t_end: float) -> Tuple[int, int]:
        """仅累加 is_ai=False：HumanMouseListener 写入的人手物理位移。"""
        sum_x, sum_y = 0, 0
        if t_start >= t_end:
            return 0, 0
        with self._lock:
            for event in reversed(self._buffer):
                if event.timestamp > t_end:
                    continue
                if event.timestamp < t_start:
                    break
                if not event.is_ai:
                    sum_x += event.dx
                    sum_y += event.dy
        return sum_x, sum_y

    def get_total_delta_sum(self, t_start: float, t_end: float) -> Tuple[int, int]:
        """汇总人+AI 的全量位移。WorldModel 用此追踪真实相机旋转。"""
        sum_x, sum_y = 0, 0
        if t_start >= t_end:
            return 0, 0
        with self._lock:
            for event in reversed(self._buffer):
                if event.timestamp > t_end:
                    continue
                if event.timestamp < t_start:
                    break
                sum_x += event.dx
                sum_y += event.dy
        return sum_x, sum_y

    def get_ai_delta_sum(self, t_start: float, t_end: float) -> Tuple[int, int]:
        """仅累加 is_ai=True：AI SendInput 的下发指令。"""
        sum_x, sum_y = 0, 0
        if t_start >= t_end:
            return 0, 0
        with self._lock:
            for event in reversed(self._buffer):
                if event.timestamp > t_end:
                    continue
                if event.timestamp < t_start:
                    break
                if event.is_ai:
                    sum_x += event.dx
                    sum_y += event.dy
        return sum_x, sum_y

    # ── 读数：意图通道（自动处理 pynput/inputs 后端差异）─────────────────

    def _check_sub_ai(self) -> bool:
        """懒加载：判断是否需要 subtract_injected_ai。"""
        if self._observed_events_reconciled:
            return False
        if not hasattr(self, '_sub_ai_checked'):
            from config import config
            b = (config.getstr("Input", "human_input_backend", "inputs") or "inputs").strip().lower()
            self._sub_ai_enabled = (
                b in ("pyn", "pynput", "hook")
                and config.getbool("Input", "human_intent_subtract_ai", True)
            )
            self._sub_ai_checked = True
        return self._sub_ai_enabled

    def get_intent_delta(self, t_start: float, t_end: float) -> Tuple[int, int]:
        """
        人手「意图」位移 —— 自动根据后端选择正确的剥离策略。

        - inputs 后端（REL 物理轴）：直接返回 get_cursor_delta_sum
        - pynput 后端（屏幕坐标差分）：表观人位移 − AI 记账位移，解除 SendInput 重影

        这是 agent / world_model / 离合器 等人机判定逻辑应使用的统一入口。
        """
        if not self._check_sub_ai():
            return self.get_cursor_delta_sum(t_start, t_end)
        hx, hy = self.get_cursor_delta_sum(t_start, t_end)
        ax, ay = self.get_ai_delta_sum(t_start, t_end)
        return (hx - ax, hy - ay)

    def get_intent_activity(self, t_start: float, t_end: float) -> float:
        """Return direction-independent physical mouse activity.

        With the pynput backend, injected movement is visible to the hook as if
        it were human input. Comparing path lengths (rather than net vectors)
        removes that echo while still detecting a human moving with, against,
        or across the controller's direction.
        """
        if t_start >= t_end:
            return 0.0
        human_path = 0.0
        ai_path = 0.0
        with self._lock:
            for event in reversed(self._buffer):
                if event.timestamp > t_end:
                    continue
                if event.timestamp < t_start:
                    break
                distance = math.hypot(event.dx, event.dy)
                if event.is_ai:
                    ai_path += distance
                else:
                    human_path += distance
        # pynput reconciled 模式：核销残余可能包含 AI 回显，用 path 减法兜底
        # inputs 模式：人手事件是纯物理轴，不需要减法
        if not self._check_sub_ai() and not self._observed_events_reconciled:
            return human_path
        return max(0.0, human_path - ai_path)

    def get_intent_delta_sum(self, t_start: float, t_end: float,
                             *, subtract_injected_ai: bool) -> Tuple[int, int]:
        """
        [兼容] 显式指定 subtract_injected_ai 的版本。新代码请用 get_intent_delta()。
        """
        if not subtract_injected_ai or self._observed_events_reconciled:
            return self.get_cursor_delta_sum(t_start, t_end)
        hx, hy = self.get_cursor_delta_sum(t_start, t_end)
        ax, ay = self.get_ai_delta_sum(t_start, t_end)
        return (hx - ax, hy - ay)
