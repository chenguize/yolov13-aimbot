# perception/ring_buffer.py
"""
鼠标事件环形缓冲区。

职责：
- 收集人手 (HumanMouseListener, is_ai=False) 和 AI (gHub SendInput, is_ai=True) 的位移事件
- 提供时间窗口内的位移汇总查询
- 内置 pynput/inputs 后端的人手意图剥离逻辑（subtract_injected_ai）
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
        self._lock = threading.Lock()
        self.max_duration = max_duration
        self._sub_ai_enabled: bool = False  # 由 use_subtract_ai() 延迟初始化

    # ── 写入 ──────────────────────────────────────────────────────────────

    def add_event(self, dx: int, dy: int, is_ai: bool):
        if dx == 0 and dy == 0:
            return
        now = time.perf_counter()
        ev = InputEvent(timestamp=now, dx=dx, dy=dy, is_ai=is_ai)
        with self._lock:
            self._buffer.append(ev)
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
        if not self._check_sub_ai():
            return human_path
        return max(0.0, human_path - ai_path)

    def get_intent_delta_sum(self, t_start: float, t_end: float,
                             *, subtract_injected_ai: bool) -> Tuple[int, int]:
        """
        [兼容] 显式指定 subtract_injected_ai 的版本。新代码请用 get_intent_delta()。
        """
        if not subtract_injected_ai:
            return self.get_cursor_delta_sum(t_start, t_end)
        hx, hy = self.get_cursor_delta_sum(t_start, t_end)
        ax, ay = self.get_ai_delta_sum(t_start, t_end)
        return (hx - ax, hy - ay)
