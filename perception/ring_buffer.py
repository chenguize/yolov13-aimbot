# perception/ring_buffer.py
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class InputEvent:
    timestamp: float
    dx: int
    dy: int
    is_ai: bool


class RingBuffer:
    """
    Phase 3 核心组件：因果账本

    职责：
    1. 记录过去 N 秒内所有的鼠标输入 (Human + AI)。
    2. 提供时间切片查询，用于 WorldModel 进行位移对冲 (Causal Hedging)。
    3. 线程安全。
    """

    def __init__(self, max_duration: float = 2.0):
        self._buffer: deque[InputEvent] = deque()
        self._lock = threading.Lock()
        self.max_duration = max_duration  # 只保留最近 2 秒的数据

    def add_event(self, dx: int, dy: int, is_ai: bool):
        """
        写入输入事件 (Human 或 AI)
        """
        if dx == 0 and dy == 0:
            return

        now = time.perf_counter()
        ev = InputEvent(timestamp=now, dx=dx, dy=dy, is_ai=is_ai)

        with self._lock:
            self._buffer.append(ev)
            # 懒惰清理：只有写入时才顺便清理过期数据
            while self._buffer and (now - self._buffer[0].timestamp > self.max_duration):
                self._buffer.popleft()

    def get_cursor_delta_sum(self, t_start: float, t_end: float) -> Tuple[int, int]:
        """
        [极其关键的修正]
        查询时间区间内的【屏幕真实总位移】。
        由于 Windows GetCursorPos (标记为 is_ai=False) 已经包含了人与 AI 的混合物理位移，
        这里绝对不能把 is_ai=True 的数据再加进去，否则会引发致命的双重计算 (Double Counting)！
        """
        sum_x, sum_y = 0, 0
        if t_start >= t_end:
            return 0, 0

        with self._lock:
            for event in reversed(self._buffer):
                if event.timestamp > t_end: continue
                if event.timestamp < t_start: break

                # ⚠️ 只累加外部捕获的真实物理位移
                if not event.is_ai:
                    sum_x += event.dx
                    sum_y += event.dy

        return sum_x, sum_y

    def get_ai_delta_sum(self, t_start: float, t_end: float) -> Tuple[int, int]:
        """
        统计 AI (gHub) 明确下发的已知位移指令。
        """
        sum_x, sum_y = 0, 0
        if t_start >= t_end:
            return 0, 0

        with self._lock:
            for event in reversed(self._buffer):
                if event.timestamp > t_end: continue
                if event.timestamp < t_start: break

                if event.is_ai:
                    sum_x += event.dx
                    sum_y += event.dy

        return sum_x, sum_y

    def get_pure_human_delta_sum(self, t_start: float, t_end: float) -> Tuple[int, int]:
        """
        [实战核心接口] 剥离出真正的人类手腕发力物理量。
        纯净人类位移 = 屏幕观测总位移 - AI 已知下发位移
        """
        total_dx, total_dy = self.get_cursor_delta_sum(t_start, t_end)
        ai_dx, ai_dy = self.get_ai_delta_sum(t_start, t_end)

        return total_dx - ai_dx, total_dy - ai_dy