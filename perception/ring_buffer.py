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
        查询时间区间 [t_start, t_end] 内的累积位移。
        WorldModel 用此接口来计算：
        "从截图那一刻(t_start) 到现在(t_end)，准星自己动了多少？"
        """
        sum_x, sum_y = 0, 0

        # 快速路径
        if t_start >= t_end:
            return 0, 0

        with self._lock:
            # 倒序遍历（因为大部分查询都是查最近的数据）
            for event in reversed(self._buffer):
                if event.timestamp > t_end:
                    continue
                if event.timestamp < t_start:
                    break  # 已超出时间窗口，停止遍历

                # 累加区间内的位移
                sum_x += event.dx
                sum_y += event.dy

        return sum_x, sum_y

    def get_ai_confidence(self, t_lookback: float = 0.5) -> float:
        """
        (可选) 分析最近 0.5s 内 AI 介入的程度。
        用于判断"现在是不是 AI 在主导控制"。
        """
        now = time.perf_counter()
        ai_moves = 0
        total_moves = 0

        with self._lock:
            for event in reversed(self._buffer):
                if now - event.timestamp > t_lookback:
                    break
                total_moves += abs(event.dx) + abs(event.dy)
                if event.is_ai:
                    ai_moves += abs(event.dx) + abs(event.dy)

        if total_moves == 0:
            return 0.0
        return ai_moves / total_moves

    def get_human_delta_sum(self, t_start: float, t_end: float) -> Tuple[int, int]:
        """
        专门为自调参设计的接口：只返回人类手动操作的累积位移。
        """
        sum_x, sum_y = 0, 0
        if t_start >= t_end:
            return 0, 0

        with self._lock:
            for event in reversed(self._buffer):
                if event.timestamp > t_end:
                    continue
                if event.timestamp < t_start:
                    break

                # 关键：只统计非 AI 事件，避免 AI 的位移污染校准基准
                if not event.is_ai:
                    sum_x += event.dx
                    sum_y += event.dy
        return sum_x, sum_y