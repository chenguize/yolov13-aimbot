# output.py - 64-bit Compatible Fix
import ctypes
from ctypes import c_long, c_ulong, c_ulonglong, Structure, Union, sizeof, byref
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from perception.ring_buffer import RingBuffer

# ============================================================================
# Win32 API 常量与结构定义 (64-bit Safe)
# ============================================================================
LONG = c_long
DWORD = c_ulong
# [关键修复] 在64位Python中，ULONG_PTR 必须是64位整数，而不是指针对象
# 否则 SendInput 会因为结构体大小不对或数据截断而拒绝执行
ULONG_PTR = c_ulonglong

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

# [关键] 魔法数字：用于标记这是 AI 发出的指令
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
        print("[SystemMouse] 初始化完成 (SendInput Mode - 64bit Fix)")

    def set_ring_buffer(self, ring_buffer):
        """
        依赖注入：传入全局唯一的 RingBuffer 实例，用于闭环反馈
        """
        self.ring_buffer = ring_buffer

    def _send_input(self, dx: int, dy: int, flags: int):
        """
        底层发送函数：封装 SendInput 并注入水印
        """
        mi = MOUSEINPUT()
        mi.dx = int(dx)
        mi.dy = int(dy)
        mi.mouseData = 0
        mi.dwFlags = flags
        mi.time = 0
        # [关键] 注入水印 (现在可以直接赋值，因为 ULONG_PTR 是整数类型)
        mi.dwExtraInfo = AI_SIGNATURE

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

        # 1. 物理执行 (带水印)
        self._send_input(x, y, MOUSEEVENTF_MOVE)

        # 2. 逻辑回写 (闭环反馈)
        if self.ring_buffer:
            self.ring_buffer.add_event(int(x), int(y), is_ai=True)

    def mouse_down(self, key=1):
        if key == 1: self._send_input(0, 0, MOUSEEVENTF_LEFTDOWN)

    def mouse_up(self, key=1):
        if key == 1: self._send_input(0, 0, MOUSEEVENTF_LEFTUP)

# 单例导出
gHub = SystemMouse()