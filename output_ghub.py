# output_ghub.py
from ctypes import *
from os import path
import time
import threading
from queue import Queue
from typing import Tuple
from config import config
from world_model import WorldModel


# ================================================
# 第一部分：你原有的 GhubMouse 封装（完全不动！）
# ================================================

class GhubMouse:
    def __init__(self):
        self.basedir = path.dirname(path.abspath(__file__))
        self.dlldir = path.join(self.basedir, 'ghub_mouse.dll')
        self.gm = CDLL(self.dlldir)
        self.gmok = self.gm.mouse_open()

    @staticmethod
    def _ghub_SendInput(*inputs):
        nInputs = len(inputs)
        LPINPUT = INPUT * nInputs
        pInputs = LPINPUT(*inputs)
        cbSize = c_int(sizeof(INPUT))
        return windll.user32.SendInput(nInputs, pInputs, cbSize)

    @staticmethod
    def _ghub_Input(structure):
        return INPUT(0, _INPUTunion(mi=structure))

    @staticmethod
    def _ghub_MouseInput(flags, x, y, data):
        return MOUSEINPUT(x, y, data, flags, 0, None)

    @staticmethod
    def _ghub_Mouse(flags, x=0, y=0, data=0):
        return GhubMouse._ghub_Input(GhubMouse._ghub_MouseInput(flags, x, y, data))

    def mouse_xy(self, x, y):
        if self.gmok:
            return self.gm.moveR(x, y)
        return self._ghub_SendInput(self._ghub_Mouse(0x0001, x, y))

    def mouse_down(self, key=1):
        if self.gmok:
            return self.gm.press(key)
        if key == 1:
            return self._ghub_SendInput(self._ghub_Mouse(0x0002))
        elif key == 2:
            return self._ghub_SendInput(self._ghub_Mouse(0x0008))

    def mouse_up(self, key=1):
        if self.gmok:
            return self.gm.release()
        if key == 1:
            return self._ghub_SendInput(self._ghub_Mouse(0x0004))
        elif key == 2:
            return self._ghub_SendInput(self._ghub_Mouse(0x0010))

    def mouse_close(self):
        if self.gmok:
            return self.gm.mouse_close()


LONG = c_long
DWORD = c_ulong
ULONG_PTR = POINTER(DWORD)


class MOUSEINPUT(Structure):
    _fields_ = (('dx', LONG),
                ('dy', LONG),
                ('mouseData', DWORD),
                ('dwFlags', DWORD),
                ('time', DWORD),
                ('dwExtraInfo', ULONG_PTR))


class _INPUTunion(Union):
    _fields_ = (('mi', MOUSEINPUT),)


class INPUT(Structure):
    _fields_ = (('type', DWORD),
                ('union', _INPUTunion))


# 全局实例（完全保留你原来的用法）
gHub = GhubMouse()


# ================================================
# 第二部分：异步包装层（不改动任何 DLL 调用）
# ================================================

class GHUBOutput(threading.Thread):
    """
    异步 GHUB 输出层（安全包装）
    - 使用队列避免阻塞主循环
    - 实现微移动合并、死区、频率限制
    - 实际发送成功的 u_k 反馈给 world_model
    - 核心发送调用仍然走你原有的 gHub.xxx()
    """

    def __init__(self, world_model: WorldModel):
        super().__init__(name="GHUBOutput", daemon=True)
        self.world_model = world_model
        self.queue = Queue(maxsize=64)
        self.last_send_time = 0.0
        self.rate_limit = 1.0 / config.getint("Output", "output_rate_limit_hz", 250)
        self.accum_dx = 0.0
        self.accum_dy = 0.0
        self.micro_threshold = config.getfloat("Output", "micro_move_threshold", 0.8)
        self.deadzone = config.getfloat("Output", "deadzone_pixels", 1.2)

    def send_move(self, dx: float, dy: float):
        """外部调用接口：放入队列"""
        try:
            self.queue.put_nowait((dx, dy))
        except Queue.Full:
            pass  # 队列满就丢弃

    def run(self):
        print("[GHUB] 异步输出线程启动（微移动合并 + 反馈）")

        while True:
            try:
                dx, dy = self.queue.get(timeout=0.05)
            except:
                time.sleep(0.005)
                continue

            now = time.perf_counter()
            if now - self.last_send_time < self.rate_limit:
                time.sleep(self.rate_limit - (now - self.last_send_time))
                now = time.perf_counter()

            # 1. 微移动累积
            self.accum_dx += dx
            self.accum_dy += dy

            move_x = round(self.accum_dx)
            move_y = round(self.accum_dy)

            # 2. 死区 + 最小移动阈值
            if abs(move_x) <= self.deadzone or abs(self.accum_dx) < self.micro_threshold:
                continue
            if abs(move_y) <= self.deadzone or abs(self.accum_dy) < self.micro_threshold:
                continue

            # 3. 发送（使用你原有的 gHub 实例，一行不改！）
            gHub.mouse_xy(move_x, move_y)

            # 4. 关键：反馈实际发送成功的控制量 u_k
            self.world_model.receive_control_feedback(float(move_x), float(move_y))

            # 更新累积
            self.accum_dx -= move_x
            self.accum_dy -= move_y

            self.last_send_time = now

        print("[GHUB] 输出线程退出")