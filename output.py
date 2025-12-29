# output_system.py
# Phase 3: 纯 Win32 API 版 (SendInput)
# 核心特性：支持 dwExtraInfo 水印，用于 RingBuffer 的敌我识别

import ctypes
from ctypes import c_long, c_ulong, Structure, Union, POINTER, sizeof, byref
from typing import Optional

# 避免循环导入
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perception.ring_buffer import RingBuffer

# ============================================================================
# Win32 API 常量与结构定义
# ============================================================================
LONG = c_long
DWORD = c_ulong
ULONG_PTR = POINTER(DWORD)

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

# [关键] 魔法数字：用于标记这是 AI 发出的指令
# 你的 Input Listener 必须检查 dwExtraInfo 是否等于这个值
AI_SIGNATURE = 0xFFC0FFEE


class MOUSEINPUT(Structure):
    _fields_ = (
        ('dx', LONG),
        ('dy', LONG),
        ('mouseData', DWORD),
        ('dwFlags', DWORD),
        ('time', DWORD),
        ('dwExtraInfo', ULONG_PTR)
    )


class _INPUTunion(Union):
    _fields_ = (('mi', MOUSEINPUT),)


class INPUT(Structure):
    _fields_ = (('type', DWORD), ('union', _INPUTunion))


# ============================================================================
# SystemMouse 类实现
# ============================================================================

class SystemMouse:
    def __init__(self):
        self.ring_buffer: Optional['RingBuffer'] = None
        print("[SystemMouse] 初始化完成 (SendInput Mode)")
        print(f"[SystemMouse] AI 指令水印已配置: {hex(AI_SIGNATURE)}")

    def set_ring_buffer(self, ring_buffer: 'RingBuffer'):
        """
        依赖注入：传入全局唯一的 RingBuffer 实例，用于闭环反馈
        """
        self.ring_buffer = ring_buffer

    def _send_input(self, dx: int, dy: int, flags: int):
        """
        底层发送函数：封装 SendInput 并注入水印
        """
        mi = MOUSEINPUT()
        mi.dx = dx
        mi.dy = dy
        mi.mouseData = 0
        mi.dwFlags = flags
        mi.time = 0

        # [Phase 3 核心] 注入水印
        # 这让监听器知道：这条指令是自己人发的，别当成人类操作记录
        mi.dwExtraInfo = ctypes.cast(AI_SIGNATURE, ULONG_PTR)

        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi = mi

        # 执行发送
        ctypes.windll.user32.SendInput(1, byref(inp), sizeof(inp))

    def mouse_xy(self, x: int, y: int):
        """
        相对移动鼠标，并回写 RingBuffer
        """
        if x == 0 and y == 0:
            return

        ix, iy = int(x), int(y)

        # 1. 物理执行 (带水印)
        self._send_input(ix, iy, MOUSEEVENTF_MOVE)

        # 2. 逻辑回写 (闭环反馈)
        # 主动告诉 RingBuffer："我动了，这是 AI 行为"
        if self.ring_buffer:
            self.ring_buffer.add_event(ix, iy, is_ai=True)
        else:
            # 开发调试期容错
            pass

    def mouse_down(self, key=1):
        """
        按下鼠标 (目前默认左键)
        """
        if key == 1:
            self._send_input(0, 0, MOUSEEVENTF_LEFTDOWN)

    def mouse_up(self, key=1):
        """
        抬起鼠标
        """
        if key == 1:
            self._send_input(0, 0, MOUSEEVENTF_LEFTUP)

    def mouse_close(self):
        """
        资源清理 (API 模式无需清理)
        """
        pass


# 为了保持 main.py 兼容性，变量名依然叫 gHub
gHub = SystemMouse()