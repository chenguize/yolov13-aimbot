import random
import time
import threading
import signal
import ctypes
from ctypes import windll, wintypes, byref
import keyboard

# --- 模块导入 ---
from config import config
from utils.types import InferenceContext
from perception.bus import FrameBus
from perception.capture import CaptureThread
from perception.ring_buffer import RingBuffer
from inference import InferenceThread
from world_model import WorldModel
from output import gHub as output_device, AI_SIGNATURE
from utils.recorder import TraceRecorder

# --- 全局控制 ---
shutdown_event = threading.Event()
paused = False
enable_aimbot = True  # 控制 AI 是否输出指令


# ==============================================================================
# [Internal Class] InputMonitor (保持原版带 Windows API 逻辑)
# ==============================================================================
class InputMonitor(threading.Thread):
    def __init__(self, ring_buffer: RingBuffer):
        super().__init__(name="InputMonitor", daemon=True)
        self.ring_buffer = ring_buffer
        self.hook = None
        self.WH_MOUSE_LL = 14
        self.WM_MOUSEMOVE = 0x0200
        try:
            self.user32 = ctypes.WinDLL('user32', use_last_error=True)
            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        except:
            self.user32, self.kernel32 = windll.user32, windll.kernel32
        self._CallNextHookEx = getattr(self.user32, 'CallNextHookEx', None)
        self._SetWindowsHookExA = getattr(self.user32, 'SetWindowsHookExA', None)
        self._UnhookWindowsHookEx = getattr(self.user32, 'UnhookWindowsHookEx', None)
        self._GetCursorPos = getattr(self.user32, 'GetCursorPos', None)
        self._PeekMessageW = getattr(self.user32, 'PeekMessageW', None)
        self._TranslateMessage = getattr(self.user32, 'TranslateMessage', None)
        self._DispatchMessageW = getattr(self.user32, 'DispatchMessageW', None)
        self._GetModuleHandleW = getattr(self.kernel32, 'GetModuleHandleW', None)
        self.HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

    def run(self):
        class MSLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [('pt', wintypes.POINT), ('mouseData', ctypes.c_ulong), ('flags', ctypes.c_ulong),
                        ('time', ctypes.c_ulong), ('dwExtraInfo', ctypes.c_ulong)]

        last_x, last_y = 0, 0
        first_run = True

        def hook_proc(nCode, wParam, lParam):
            nonlocal last_x, last_y, first_run
            if nCode >= 0 and wParam == self.WM_MOUSEMOVE:
                try:
                    struct = ctypes.cast(lParam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                    # 仅记录非 AI 生成的物理移动
                    if struct.dwExtraInfo != AI_SIGNATURE:
                        if first_run:
                            last_x, last_y = struct.pt.x, struct.pt.y
                            first_run = False
                        else:
                            dx, dy = struct.pt.x - last_x, struct.pt.y - last_y
                            last_x, last_y = struct.pt.x, struct.pt.y
                            if dx != 0 or dy != 0: self.ring_buffer.add_event(dx, dy, is_ai=False)
                except:
                    pass
            return self._CallNextHookEx(None, nCode, wParam, lParam)

        pt = wintypes.POINT()
        self._GetCursorPos(byref(pt))
        last_x, last_y = pt.x, pt.y
        self.pointer = self.HOOKPROC(hook_proc)
        self.hook = self._SetWindowsHookExA(self.WH_MOUSE_LL, self.pointer, self._GetModuleHandleW(None), 0)
        msg = wintypes.MSG()
        while not shutdown_event.is_set():
            if self._PeekMessageW(byref(msg), None, 0, 0, 1):
                self._TranslateMessage(byref(msg));
                self._DispatchMessageW(byref(msg))
            else:
                time.sleep(0.001)

    def stop_hook(self):
        if self.hook: self._UnhookWindowsHookEx(self.hook)


def signal_handler(sig, frame): shutdown_event.set()


def toggle_pause(): global paused; paused = not paused; print(f"PAUSE: {paused}")


def toggle_aimbot():
    global enable_aimbot
    enable_aimbot = not enable_aimbot
    print(f"[Aimbot] {'ENABLED' if enable_aimbot else 'DISABLED (Calibration Mode)'}")


def main():
    global paused, enable_aimbot
    signal.signal(signal.SIGINT, signal_handler)

    ring_buffer = RingBuffer(max_duration=2.0)
    frame_bus = FrameBus()
    recorder = TraceRecorder(save_path="debug_trace.pkl")
    if config.getbool("Debug", "enable_trace", False): recorder.enable()

    output_device.set_ring_buffer(ring_buffer)
    input_monitor = InputMonitor(ring_buffer)
    world_model = WorldModel()

    input_monitor.start()
    CaptureThread(frame_bus, shutdown_event).start()
    InferenceThread(frame_bus, world_model, shutdown_event).start()

    keyboard.add_hotkey('p', toggle_pause)
    keyboard.add_hotkey('alt+f1', toggle_aimbot)  # 绑定快捷键用于“离线校准”

    ctx = InferenceContext()
    last_processed_t_cap = 0.0
    loop_counter = 0
    enable_trigger = config.getbool("General", "enable_triggerbot", True)

    while not shutdown_event.is_set():
        if paused: time.sleep(0.1); continue

        loop_start = time.perf_counter()
        loop_counter += 1

        # 核心：WorldModel 始终运行 step 以进行延迟观测和灵敏度校准
        final_move = world_model.step(ctx, ring_buffer)

        if not final_move or ctx.t_cap == last_processed_t_cap:
            time.sleep(0.0005)
            continue

        last_processed_t_cap = ctx.t_cap
        final_x, final_y = final_move

        # 仅在 enable_aimbot 为 True 时执行 AI 移动
        if enable_aimbot:
            if abs(final_x) > 0.5 or abs(final_y) > 0.5:
                output_device.mouse_xy(final_x, final_y)

            if enable_trigger:
                trigger_target = {"screen_x": ctx.p_predict[0], "screen_y": ctx.p_predict[1], "conf": ctx.conf}
                if world_model.controller.should_trigger(trigger_target,
                                                         (world_model.crop_center, world_model.crop_center)):
                    output_device.mouse_down(1)
                    time.sleep(random.uniform(0.015, 0.03))
                    output_device.mouse_up(1)

        if loop_counter % 100 == 0:
            total_ms = (time.perf_counter() - loop_start) * 1000
            print(
                f"[Main] FPS: {1000 / total_ms:.1f} | DynLag: {ctx.dynamic_lag_ms:.1f}ms | K: {world_model.strategy.calib.k_x:.2f} | Aim:{enable_aimbot}")

        recorder.record_frame(ctx)

    recorder.save_to_disk()
    input_monitor.stop_hook()
    keyboard.unhook_all()


if __name__ == "__main__":
    main()