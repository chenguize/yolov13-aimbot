# output.py - Optimized
import ctypes
from ctypes import c_long, c_ulong, c_ulonglong, Structure, Union, sizeof, byref
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from perception.ring_buffer import RingBuffer

# Win32 API Definitions
LONG = c_long
DWORD = c_ulong
ULONG_PTR = c_ulonglong
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
AI_SIGNATURE = 0xFFC0FFEE


class MOUSEINPUT(Structure):
    _fields_ = (('dx', LONG), ('dy', LONG), ('mouseData', DWORD),
                ('dwFlags', DWORD), ('time', DWORD), ('dwExtraInfo', ULONG_PTR))


class _INPUTunion(Union):
    _fields_ = (('mi', MOUSEINPUT),)


class INPUT(Structure):
    _fields_ = (('type', DWORD), ('union', _INPUTunion))


class SystemMouse:
    def __init__(self):
        self.ring_buffer: Optional['RingBuffer'] = None

        # === 优化核心：预先分配内存 ===
        self._inp = INPUT()
        self._inp.type = INPUT_MOUSE
        self._inp.union.mi.dwExtraInfo = AI_SIGNATURE  # 预先填好水印
        self._inp.union.mi.time = 0
        self._inp.union.mi.mouseData = 0

        # 缓存 SendInput 函数指针，减少属性查找开销
        self._send_input_func = ctypes.windll.user32.SendInput
        self._sizeof_inp = sizeof(INPUT)

        print("[SystemMouse] 初始化完成 (Pre-allocated Mode)")

    def set_ring_buffer(self, ring_buffer):
        self.ring_buffer = ring_buffer

    def mouse_xy(self, x: int, y: int):
        if x == 0 and y == 0:
            return

        # === 极速发送 ===
        # 只修改变化的 dx, dy 和 flags
        self._inp.union.mi.dx = int(x)
        self._inp.union.mi.dy = int(y)
        self._inp.union.mi.dwFlags = MOUSEEVENTF_MOVE

        # 直接调用，无需创建新对象
        self._send_input_func(1, byref(self._inp), self._sizeof_inp)

        if self.ring_buffer:
            self.ring_buffer.add_event(int(x), int(y), is_ai=True)

    def mouse_down(self, key=1):
        if key == 1:
            self._inp.union.mi.dx = 0
            self._inp.union.mi.dy = 0
            self._inp.union.mi.dwFlags = MOUSEEVENTF_LEFTDOWN
            self._send_input_func(1, byref(self._inp), self._sizeof_inp)

    def mouse_up(self, key=1):
        if key == 1:
            self._inp.union.mi.dx = 0
            self._inp.union.mi.dy = 0
            self._inp.union.mi.dwFlags = MOUSEEVENTF_LEFTUP
            self._send_input_func(1, byref(self._inp), self._sizeof_inp)


gHub = SystemMouse()