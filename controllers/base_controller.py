from abc import ABC, abstractmethod
from typing import Tuple, Deque
from collections import deque
import threading
import time
import numpy as np

from config import config


class BaseController(ABC):

    def __init__(self):
        """
        Phase 6:
        统一运动学运行态基类
        - 所有 Controller 自动继承：
            pos / vel
            backlog
            _lock
            max_backlog_age
        """

        self.config = config

        # -----------------------------
        # screen geometry
        # -----------------------------
        w = self.config.getint("General", "screen_width", 1920)
        h = self.config.getint("General", "screen_height", 1080)
        self.screen_center = (w / 2, h / 2)

        # -----------------------------
        # threading
        # -----------------------------
        self._lock = threading.Lock()

        # -----------------------------
        # kinematic state
        # -----------------------------
        self.pos = np.zeros(2, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)

        self.last_cmd = np.zeros(2, dtype=np.float64)

        # -----------------------------
        # backlog buffer (human / ai)
        # -----------------------------
        self.max_backlog_age = self.config.getfloat(
            "Controller", "max_backlog_age", 0.25
        )

        self.backlog: Deque = deque()

        # -----------------------------
        # timing
        # -----------------------------
        self.last_tick = time.perf_counter()

    # =====================================================
    # backlog helpers
    # =====================================================

    def _cleanup_backlog(self, now: float):

        while self.backlog and now - self.backlog[0][0] > self.max_backlog_age:
            self.backlog.popleft()

    def _get_backlog_delta(self, t0: float, t1: float):

        dx = dy = 0.0

        for t, mx, my in self.backlog:
            if t0 <= t <= t1:
                dx += mx
                dy += my

        return dx, dy

    # =====================================================
    # abstract API
    # =====================================================

    @abstractmethod
    def compute(
        self,
        intent_dx: float,
        intent_dy: float,
        dt: float,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Args:
            intent_dx/dy: aim strategy / world model 输出
            dt: timestep
        Returns:
            vx, vy (counts / second)
        """
        pass

    @abstractmethod
    def tick_mouse(self) -> Tuple[int, int]:
        """
        1000Hz polling interface
        使用 self.vel / self.pos
        """
        pass

    def get_move_emit_diag(self):
        """
        可选：PROController 在 compute 与 tick_mouse 之间统计「指令速度 vs 实际下发」。
        非 PRO 或尚未实现时返回 None。
        """
        return None
