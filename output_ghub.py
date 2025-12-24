# output_ghub.py - GHUB 输出模块
#
# 核心职责：
# 1. GHUB DLL 封装（鼠标控制接口）
# 2. 异步输出队列（避免阻塞主循环）
# 3. 微移动合并（累积小移动，提高精度）
# 4. 死区处理（过滤过小的移动）
# 5. 频率限制（控制输出频率）
# 6. 控制反馈（将实际移动量反馈给世界模型）
#
# 架构特点：带队列 + 微移动合并 + 背压，可开关
#

from ctypes import *
from os import path
import time
import threading
from queue import Queue
from typing import Tuple
from config import config
from world_model import WorldModel


# ================================================
# 第一部分：GhubMouse 封装（完全保留原有实现）
# ================================================

class GhubMouse:
    """GHUB 鼠标控制封装类 - 提供底层鼠标控制接口"""
    
    def __init__(self):
        """初始化 GHUB DLL"""
        self.basedir = path.dirname(path.abspath(__file__))
        self.dlldir = path.join(self.basedir, 'ghub_mouse.dll')
        self.gm = CDLL(self.dlldir)
        self.gmok = self.gm.mouse_open()

    @staticmethod
    def _ghub_SendInput(*inputs):
        """使用 Windows API 发送输入事件"""
        nInputs = len(inputs)
        LPINPUT = INPUT * nInputs
        pInputs = LPINPUT(*inputs)
        cbSize = c_int(sizeof(INPUT))
        return windll.user32.SendInput(nInputs, pInputs, cbSize)

    @staticmethod
    def _ghub_Input(structure):
        """创建输入结构"""
        return INPUT(0, _INPUTunion(mi=structure))

    @staticmethod
    def _ghub_MouseInput(flags, x, y, data):
        """创建鼠标输入结构"""
        return MOUSEINPUT(x, y, data, flags, 0, None)

    @staticmethod
    def _ghub_Mouse(flags, x=0, y=0, data=0):
        """创建鼠标事件"""
        return GhubMouse._ghub_Input(GhubMouse._ghub_MouseInput(flags, x, y, data))

    def mouse_xy(self, x, y):
        """相对移动鼠标 x, y 像素"""
        if self.gmok:
            return self.gm.moveR(x, y)  # 使用 GHUB DLL
        return self._ghub_SendInput(self._ghub_Mouse(0x0001, x, y))  # 回退到 Windows API

    def mouse_down(self, key=1):
        """按下鼠标按键"""
        if self.gmok:
            return self.gm.press(key)  # 使用 GHUB DLL
        if key == 1:
            return self._ghub_SendInput(self._ghub_Mouse(0x0002))  # 左键
        elif key == 2:
            return self._ghub_SendInput(self._ghub_Mouse(0x0008))  # 右键

    def mouse_up(self, key=1):
        """释放鼠标按键"""
        if self.gmok:
            return self.gm.release()  # 使用 GHUB DLL
        if key == 1:
            return self._ghub_SendInput(self._ghub_Mouse(0x0004))  # 左键
        elif key == 2:
            return self._ghub_SendInput(self._ghub_Mouse(0x0010))  # 右键

    def mouse_close(self):
        """关闭鼠标控制"""
        if self.gmok:
            return self.gm.mouse_close()


# Windows API 结构定义
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
# 第二部分：异步输出包装层
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
        """初始化异步输出线程"""
        super().__init__(name="GHUBOutput", daemon=True)
        self.world_model = world_model  # 世界模型引用，用于反馈控制量
        self.queue = Queue(maxsize=64)  # 输出队列，限制队列大小
        self.last_send_time = 0.0  # 上次发送时间，用于频率限制
        self.rate_limit = 1.0 / config.getint("Output", "output_rate_limit_hz", 250)  # 输出频率限制
        self.accum_dx = 0.0  # X 轴累积移动量
        self.accum_dy = 0.0  # Y 轴累积移动量
        self.micro_threshold = config.getfloat("Output", "micro_move_threshold", 0.8)  # 微移动阈值
        self.deadzone = config.getfloat("Output", "deadzone_pixels", 1.2)  # 死区阈值

    def send_move(self, dx: float, dy: float):
        """外部调用接口：将移动量放入队列"""
        try:
            self.queue.put_nowait((dx, dy))  # 非阻塞放入队列
        except Queue.Full:
            pass  # 队列满就丢弃，避免阻塞

    def run(self):
        """异步输出线程主循环"""
        print("[GHUB] 异步输出线程启动（微移动合并 + 反馈）")

        while True:
            try:
                # 从队列获取移动量
                dx, dy = self.queue.get(timeout=0.05)
            except:
                # 队列空时短暂休眠
                time.sleep(0.005)
                continue

            # 频率限制：确保不超过最大输出频率
            now = time.perf_counter()
            if now - self.last_send_time < self.rate_limit:
                time.sleep(self.rate_limit - (now - self.last_send_time))
                now = time.perf_counter()

            # 1. 微移动累积：将当前移动量加入累积值
            self.accum_dx += dx
            self.accum_dy += dy

            move_x = round(self.accum_dx)  # 四舍五入为整数像素
            move_y = round(self.accum_dy)

            # 2. 死区 + 最小移动阈值检查
            # 如果移动量小于死区或累积值小于微移动阈值，则跳过
            if abs(move_x) <= self.deadzone or abs(self.accum_dx) < self.micro_threshold:
                continue
            if abs(move_y) <= self.deadzone or abs(self.accum_dy) < self.micro_threshold:
                continue

            # 3. 发送移动命令（使用原有的 gHub 实例，一行不改！）
            gHub.mouse_xy(move_x, move_y)

            # 4. 关键：反馈实际发送成功的控制量 u_k 给世界模型
            # 这用于自运动补偿，让世界模型知道实际的鼠标移动
            self.world_model.receive_control_feedback(float(move_x), float(move_y))

            # 更新累积值，减去已发送的部分
            self.accum_dx -= move_x
            self.accum_dy -= move_y

            self.last_send_time = now  # 更新上次发送时间

        print("[GHUB] 输出线程退出")