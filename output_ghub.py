# output_ghub.py - 只用 DLL 版（强制不回退 SendInput）

from ctypes import *
from os import path


class GhubMouse:
    def __init__(self):
        self.basedir = path.dirname(path.abspath(__file__))
        self.dlldir = path.join(self.basedir, 'ghub_mouse.dll')
        print(f"[GHUB] 加载 DLL: {self.dlldir}")

        try:
            self.gm = CDLL(self.dlldir)
            self.gmok = self.gm.mouse_open()
            print(f"[GHUB] mouse_open 返回: {self.gmok}")
            if not self.gmok:
                print("[GHUB] !!! mouse_open 返回 0 !!! DLL 可能无效，但继续使用")
        except Exception as e:
            print(f"[GHUB] DLL 加载失败: {e}")
            self.gmok = False
            raise RuntimeError("DLL 加载失败，无法继续")

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
        print(f"[GHUB] mouse_xy 调用: x={x}, y={y}")
        x, y = int(x), int(y)

        # 强制只用 DLL，不回退
        if self.gmok:
            ret = self.gm.moveR(x, y)
            print(f"[GHUB] DLL moveR 返回: {ret}")
            return ret
        else:
            raise RuntimeError("DLL 未初始化成功，无法移动鼠标（严格只用 DLL）")

    def mouse_down(self, key=1):
        if self.gmok:
            return self.gm.press(key)
        raise RuntimeError("DLL 未初始化成功，无法按下鼠标")

    def mouse_up(self, key=1):
        if self.gmok:
            return self.gm.release()
        raise RuntimeError("DLL 未初始化成功，无法释放鼠标")

    def mouse_close(self):
        if self.gmok:
            return self.gm.mouse_close()


# Windows API 结构（备用，但不使用）
LONG = c_long
DWORD = c_ulong
ULONG_PTR = POINTER(DWORD)


class MOUSEINPUT(Structure):
    _fields_ = (('dx', LONG), ('dy', LONG), ('mouseData', DWORD),
                ('dwFlags', DWORD), ('time', DWORD), ('dwExtraInfo', ULONG_PTR))


class _INPUTunion(Union):
    _fields_ = (('mi', MOUSEINPUT),)


class INPUT(Structure):
    _fields_ = (('type', DWORD), ('union', _INPUTunion))


gHub = GhubMouse()