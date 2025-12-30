# main.py - Phase 4 System Coordinator (Anti-Flying Fix)
# 核心修复：
# 1. [防重复] 增加 last_processed_t_cap 检查，防止对同一帧重复瞄准
# 2. [防越界] 增加 capture_size 边界检查，过滤掉 (697, 697) 这种幽灵坐标

import random
import time
import threading
import signal
import sys
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
from aim_strategies.factory import create_aim_strategy
from controllers.controller_factory import get_controller
from output import gHub as output_device, AI_SIGNATURE
from utils.recorder import TraceRecorder

# --- 全局控制 ---
shutdown_event = threading.Event()
paused = False


# ==============================================================================
# [Internal Class] InputMonitor (全函数显式定义版)
# ==============================================================================
class InputMonitor(threading.Thread):
    def __init__(self, ring_buffer: RingBuffer):
        super().__init__(name="InputMonitor", daemon=True)
        self.ring_buffer = ring_buffer
        self.hook = None
        self.WH_MOUSE_LL = 14
        self.WM_MOUSEMOVE = 0x0200

        # 1. 安全加载 DLL
        try:
            self.user32 = ctypes.WinDLL('user32', use_last_error=True)
            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        except Exception as e:
            print(f"[Input] ⚠️ 标准加载失败 ({e})，尝试 windll 回退...")
            self.user32 = windll.user32
            self.kernel32 = windll.kernel32

        self._CallNextHookEx = getattr(self.user32, 'CallNextHookEx', None)
        self._SetWindowsHookExA = getattr(self.user32, 'SetWindowsHookExA', None)
        self._UnhookWindowsHookEx = getattr(self.user32, 'UnhookWindowsHookEx', None)
        self._GetCursorPos = getattr(self.user32, 'GetCursorPos', None)
        self._PeekMessageW = getattr(self.user32, 'PeekMessageW', None)
        self._TranslateMessage = getattr(self.user32, 'TranslateMessage', None)
        self._DispatchMessageW = getattr(self.user32, 'DispatchMessageW', None)
        self._GetModuleHandleW = getattr(self.kernel32, 'GetModuleHandleW', None)
        if not self._GetModuleHandleW:
            self._GetModuleHandleW = getattr(self.kernel32, 'GetModuleHandleA', None)

        if self._CallNextHookEx:
            self._CallNextHookEx.argtypes = (ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
            self._CallNextHookEx.restype = ctypes.c_void_p
        if self._GetModuleHandleW:
            self._GetModuleHandleW.argtypes = (ctypes.c_wchar_p,)
            self._GetModuleHandleW.restype = ctypes.c_void_p
        if self._SetWindowsHookExA:
            self._SetWindowsHookExA.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
            self._SetWindowsHookExA.restype = ctypes.c_void_p
        if self._GetCursorPos:
            self._GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
            self._GetCursorPos.restype = ctypes.c_int

        self.HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

    def run(self):
        print("[Input] 鼠标监听线程启动")

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
                    if struct.dwExtraInfo == AI_SIGNATURE:
                        pass
                    else:
                        if first_run:
                            last_x, last_y = struct.pt.x, struct.pt.y; first_run = False
                        else:
                            dx = struct.pt.x - last_x;
                            dy = struct.pt.y - last_y;
                            last_x, last_y = struct.pt.x, struct.pt.y
                            if dx != 0 or dy != 0: self.ring_buffer.add_event(dx, dy, is_ai=False)
                except Exception:
                    pass
            return self._CallNextHookEx(None, nCode, wParam, lParam)

        pt = wintypes.POINT()
        self._GetCursorPos(byref(pt))
        last_x, last_y = pt.x, pt.y
        self.pointer = self.HOOKPROC(hook_proc)
        h_mod = self._GetModuleHandleW(None)
        self.hook = self._SetWindowsHookExA(self.WH_MOUSE_LL, self.pointer, h_mod, 0)
        if not self.hook: self.hook = self._SetWindowsHookExA(self.WH_MOUSE_LL, self.pointer, 0, 0)

        msg = wintypes.MSG()
        try:
            while not shutdown_event.is_set():
                if self._PeekMessageW(byref(msg), None, 0, 0, 1) != 0:
                    self._TranslateMessage(byref(msg));
                    self._DispatchMessageW(byref(msg))
                else:
                    time.sleep(0.001)
        finally:
            self.stop_hook()

    def stop_hook(self):
        if self.hook and self._UnhookWindowsHookEx:
            try:
                self._UnhookWindowsHookEx(self.hook)
            except:
                pass
            self.hook = None


def signal_handler(sig, frame): shutdown_event.set()


def toggle_pause(): global paused; paused = not paused; print(f"PAUSE: {paused}")


def main():
    global paused
    signal.signal(signal.SIGINT, signal_handler)

    # 初始化
    ring_buffer = RingBuffer(max_duration=2.0)
    frame_bus = FrameBus()
    recorder = TraceRecorder(save_path="debug_trace.pkl")
    if config.getbool("Debug", "enable_trace", False): recorder.enable()

    output_device.set_ring_buffer(ring_buffer)
    input_monitor = InputMonitor(ring_buffer)
    world_model = WorldModel()
    strategy = create_aim_strategy()
    controller = get_controller()

    capture_thread = CaptureThread(frame_bus, shutdown_event)
    inference_thread = InferenceThread(frame_bus, world_model, shutdown_event)

    input_monitor.start()
    capture_thread.start()
    inference_thread.start()
    keyboard.add_hotkey('p', toggle_pause)

    # 坐标参数
    capture_size = config.getint("General", "capture_size", 256)
    sc_x = capture_size // 2
    sc_y = capture_size // 2

    print(f"[System] 坐标中心: ({sc_x}, {sc_y}) | 截图尺寸: {capture_size}")
    print("[System] Ready. Press 'P' to pause.")

    ctx = InferenceContext()
    last_loop_time = time.perf_counter()

    # [关键修复 1] 用于防止重复消费同一帧
    last_processed_t_cap = 0.0

    # 自动校准状态
    last_target_id = -1
    last_target_xy = (0, 0)
    calib_last_frame_time = 0.0

    enable_trigger = config.getbool("General", "enable_triggerbot", True)
    enable_auto_calib = config.getbool("AimStrategy", "enable_auto_calibration", True)

    while not shutdown_event.is_set():
        if paused: time.sleep(0.1); continue

        loop_start = time.perf_counter()
        dt = loop_start - last_loop_time
        last_loop_time = loop_start

        try:
            # 1. 认知感知 Step
            world_model.step(ctx, ring_buffer)

            # [关键修复 2] 如果当前帧没有目标，或者数据已经处理过了，直接跳过
            # 这能彻底解决 "0 ms" 连发导致的飞天问题
            if not ctx.is_valid or ctx.t_cap == last_processed_t_cap:
                if not ctx.is_valid:
                    last_target_id = -1
                # 即使没操作，也记录一下状态以便调试
                # recorder.record_frame(ctx)
                time.sleep(0.0005)  # 极短休眠，让出CPU给Capture线程
                continue

            # 标记此帧已处理
            last_processed_t_cap = ctx.t_cap

            # 2. 预测坐标
            pred_x, pred_y = ctx.p_predict

            # [关键修复 3] 安全边界检查
            # 如果坐标超出截图范围太多（比如之前日志里的 697），认为是异常值，忽略
            # 放宽一点点边界以允许轻微的预测溢出
            safe_margin = 50
            if not (-safe_margin <= pred_x <= capture_size + safe_margin and
                    -safe_margin <= pred_y <= capture_size + safe_margin):
                # print(f"[Safety] 忽略异常坐标: ({pred_x:.1f}, {pred_y:.1f})")
                continue

            target_dx = pred_x - sc_x
            target_dy = pred_y - sc_y

            # 3. 策略计算
            if hasattr(strategy, 'calculate_mouse_move'):
                res = strategy.calculate_mouse_move(target_dx, target_dy)
                if isinstance(res, tuple):
                    intent_x, intent_y = res
                    ctx.strategy_result = (intent_x, intent_y)
                else:
                    intent_x, intent_y = res.raw_counts_x, res.raw_counts_y
                    ctx.strategy_result = res
            else:
                intent_x, intent_y = target_dx, target_dy
                ctx.strategy_result = (intent_x, intent_y)

            # 4. 控制器平滑
            final_x, final_y = controller.compute(intent_x, intent_y, dt)

            # 5. 执行输出
            if abs(final_x) > 0.5 or abs(final_y) > 0.5:
                output_device.mouse_xy(final_x, final_y)

                # --- 自动校准逻辑 (保持不变) ---
                if enable_auto_calib and hasattr(ctx, 'targets') and ctx.targets:
                    if ctx.selected_id == last_target_id and last_target_id != -1:
                        curr_det = next((t for t in ctx.targets if t.class_id == ctx.selected_id), None)
                        if curr_det:
                            curr_x, curr_y = curr_det.x, curr_det.y
                            prev_x, prev_y = last_target_xy
                            actual_pixel_dx = curr_x - prev_x
                            actual_pixel_dy = curr_y - prev_y
                            ai_counts_x, ai_counts_y = ring_buffer.get_cursor_delta_sum(calib_last_frame_time,
                                                                                        ctx.t_cap)

                            if abs(ai_counts_x) > 5 or abs(ai_counts_y) > 5:
                                if hasattr(strategy, 'feedback_update'):
                                    strategy.feedback_update(ai_counts_x, -actual_pixel_dx, ai_counts_y,
                                                             -actual_pixel_dy)
                            last_target_xy = (curr_x, curr_y)
                            calib_last_frame_time = ctx.t_cap
                    elif ctx.selected_id != -1:
                        curr_det = next((t for t in ctx.targets if t.class_id == ctx.selected_id), None)
                        if curr_det:
                            last_target_id = ctx.selected_id
                            last_target_xy = (curr_det.x, curr_det.y)
                            calib_last_frame_time = ctx.t_cap
                    else:
                        last_target_id = -1

            # 6. Triggerbot
            trigger_target = {
                "screen_x": pred_x, "screen_y": pred_y,
                "conf": ctx.conf if hasattr(ctx, 'conf') else 0.8
            }
            should_fire = controller.should_trigger(trigger_target, (sc_x, sc_y))

            if should_fire and enable_trigger:
                output_device.mouse_down(1)
                time.sleep(random.uniform(0.015, 0.03))
                output_device.mouse_up(1)
                time.sleep(0.04)

            recorder.record_frame(ctx)

        except Exception as e:
            print(f"[Loop] Error: {e}")
            time.sleep(0.1)

    recorder.save_to_disk()
    try:
        input_monitor.stop_hook()
    except:
        pass
    keyboard.unhook_all()
    capture_thread.join()
    inference_thread.join()


if __name__ == "__main__":
    main()