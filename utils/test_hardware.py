import time
import ctypes
from ctypes import c_long, c_ulong, Structure, Union, POINTER, sizeof, byref

# --- 最小化 Win32 定义 ---
LONG = c_long
DWORD = c_ulong
ULONG_PTR = POINTER(DWORD)
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001


class MOUSEINPUT(Structure):
    _fields_ = (('dx', LONG), ('dy', LONG), ('mouseData', DWORD), ('dwFlags', DWORD),
                ('time', DWORD), ('dwExtraInfo', ULONG_PTR))


class _INPUTunion(Union):
    _fields_ = (('mi', MOUSEINPUT),)


class INPUT(Structure):
    _fields_ = (('type', DWORD), ('union', _INPUTunion))


def move_mouse_test(dx, dy):
    mi = MOUSEINPUT(dx=dx, dy=dy, mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=None)
    inp = INPUT(type=INPUT_MOUSE, union=_INPUTunion(mi=mi))
    # 发送指令
    ret = ctypes.windll.user32.SendInput(1, byref(inp), sizeof(inp))
    return ret


if __name__ == "__main__":
    print("=== 鼠标硬件层测试 ===")
    print("请在 3 秒内松开鼠标，观察光标是否移动...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print(">>> 发送移动指令 (右下 +100, +100)")
    result = move_mouse_test(100, 100)

    if result == 1:
        print("✅ Windows 接收了指令 (Return 1)")
        print("如果光标没动，说明被高权限窗口(如任务管理器/游戏)拦截了。")
        print("解决方案：请务必以【管理员身份】运行代码。")
    else:
        print(f"❌ 发送失败 (Return {result}) - 可能是结构体定义不兼容")