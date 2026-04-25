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
    def __init__(self, max_duration: float = 2.0):
        self._buffer: deque[InputEvent] = deque()
        self._lock = threading.Lock()
        self.max_duration = max_duration

    def add_event(self, dx: int, dy: int, is_ai: bool):
        if dx == 0 and dy == 0:
            return
        now = time.perf_counter()
        ev = InputEvent(timestamp=now, dx=dx, dy=dy, is_ai=is_ai)
        with self._lock:
            self._buffer.append(ev)
            while self._buffer and (now - self._buffer[0].timestamp > self.max_duration):
                self._buffer.popleft()

    def get_cursor_delta_sum(self, t_start: float, t_end: float) -> Tuple[int, int]:
        """
        【修复核心】：获取游戏画面的绝对总偏移。
        因为 Windows RawInput 会捕捉所有物理和软件输入的混合，
        所以 is_ai=False 就是传输给游戏的真实总位移！
        千万不能再累加 is_ai=True，否则会造成 2 倍放大的震荡！
        """
        sum_x, sum_y = 0, 0
        if t_start >= t_end:
            return 0, 0

        with self._lock:
            for event in reversed(self._buffer):
                if event.timestamp > t_end: continue
                if event.timestamp < t_start: break

                # 仅提取混合了全部指令的底层真实输入
                if not event.is_ai:
                    sum_x += event.dx
                    sum_y += event.dy

        return sum_x, sum_y

    def get_ai_delta_sum(self, t_start: float, t_end: float) -> Tuple[int, int]:
        """统计 AI 明确下发的已知位移指令。"""
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
        【人机离合器核心】：纯净人类位移 = 底层总位移 - AI下发位移
        """
        total_dx, total_dy = self.get_cursor_delta_sum(t_start, t_end)
        ai_dx, ai_dy = self.get_ai_delta_sum(t_start, t_end)

        return total_dx - ai_dx, total_dy - ai_dy