# main.py - Phase 3 System Coordinator (High Performance Mode)
# 核心职责：
# 1. 启动并编排所有异步子系统
# 2. 维护 RingBuffer 因果闭环
# 3. 全速执行 "感知 -> 决策 -> 控制 -> 执行" 循环 (无人工限速)
# 4. 实现自动调参 (Auto-Tune) 闭环反馈

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
# 使用新的 output_system (SendInput版)
from output import gHub as output_device, AI_SIGNATURE
# (新增) 追踪回放
from utils.recorder import TraceRecorder

# --- 全局控制 ---
shutdown_event = threading.Event()
paused = False


# ==============================================================================
# [Internal Class] InputMonitor
# ==============================================================================
class InputMonitor(threading.Thread):
    def __init__(self, ring_buffer: RingBuffer):
        super().__init__(name="InputMonitor", daemon=True)
        self.ring_buffer = ring_buffer
        self.hook = None
        self.WH_MOUSE_LL = 14
        self.WM_MOUSEMOVE = 0x0200
        self.user32 = windll.user32
        self.HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

    def run(self):
        print("[Input] 鼠标监听线程启动 (因果分离模式)")

        class MSLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [
                ('pt', wintypes.POINT),
                ('mouseData', ctypes.c_ulong),
                ('flags', ctypes.c_ulong),
                ('time', ctypes.c_ulong),
                ('dwExtraInfo', ctypes.c_ulong)
            ]

        last_x, last_y = 0, 0
        first_run = True

        def hook_proc(nCode, wParam, lParam):
            nonlocal last_x, last_y, first_run
            if nCode >= 0 and wParam == self.WM_MOUSEMOVE:
                struct = ctypes.cast(lParam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                # 敌我识别：忽略 AI 发出的指令
                if struct.dwExtraInfo == AI_SIGNATURE:
                    pass
                else:
                    if first_run:
                        last_x, last_y = struct.pt.x, struct.pt.y
                        first_run = False
                    else:
                        dx = struct.pt.x - last_x
                        dy = struct.pt.y - last_y
                        last_x, last_y = struct.pt.x, struct.pt.y
                        if dx != 0 or dy != 0:
                            self.ring_buffer.add_event(dx, dy, is_ai=False)
            return self.user32.CallNextHookEx(None, nCode, wParam, lParam)

        pt = wintypes.POINT()
        self.user32.GetCursorPos(byref(pt))
        last_x, last_y = pt.x, pt.y

        pointer = self.HOOKPROC(hook_proc)
        self.hook = self.user32.SetWindowsHookExA(self.WH_MOUSE_LL, pointer, windll.kernel32.GetModuleHandleW(None), 0)

        msg = wintypes.MSG()
        while not shutdown_event.is_set():
            if self.user32.PeekMessageA(byref(msg), None, 0, 0, 1) != 0:
                self.user32.TranslateMessage(byref(msg))
                self.user32.DispatchMessageA(byref(msg))
            else:
                time.sleep(0.001)  # 监听线程稍微休眠不影响主线程性能

        if self.hook: self.user32.UnhookWindowsHookEx(self.hook)
        print("[Input] 监听结束")


# ==============================================================================
# [Helpers]
# ==============================================================================
def signal_handler(sig, frame):
    print("\n[System] 正在停止所有服务...")
    shutdown_event.set()


def toggle_pause():
    global paused
    paused = not paused
    state = "PAUSED ⏸️" if paused else "ACTIVE ▶️"
    print(f"\n[System] >>> {state} <<<\n")


# ==============================================================================
# [Main] 系统入口
# ==============================================================================
def main():
    global paused

    # 1. 基础环境
    signal.signal(signal.SIGINT, signal_handler)
    print("=" * 60)
    print(" YOLOv13 Aimbot System (Phase 3 Performance Mode)")
    print(" World Model | Causal Hedging | High Frequency Loop")
    print("=" * 60)

    # 2. 初始化
    ring_buffer = RingBuffer(max_duration=2.0)
    frame_bus = FrameBus()

    recorder = TraceRecorder(save_path="debug_trace.pkl")
    if config.getbool("Debug", "enable_trace", False):
        recorder.enable()

    # 3. 注入
    output_device.set_ring_buffer(ring_buffer)
    input_monitor = InputMonitor(ring_buffer)
    world_model = WorldModel()

    # 4. 工厂
    strategy = create_aim_strategy()
    controller = get_controller()

    # 5. 启动
    capture_thread = CaptureThread(frame_bus, shutdown_event)
    inference_thread = InferenceThread(frame_bus, world_model, shutdown_event)
    input_monitor.start()
    capture_thread.start()
    inference_thread.start()

    keyboard.add_hotkey('p', toggle_pause)

    sc_x = config.getint("General", "screen_width", 1920) // 2
    sc_y = config.getint("General", "screen_height", 1080) // 2

    print("[System] 系统已全速启动 (No Sleep Mode)...")

    # ==========================================================================
    # [Loop] 主控制循环
    # ==========================================================================
    ctx = InferenceContext()
    last_loop_time = time.perf_counter()

    # [Auto-Tune] 自动调参专用状态记忆
    last_target_id = -1
    last_target_xy = (0, 0)
    last_frame_time = 0.0

    # 缓存配置
    enable_trigger = config.getbool("General", "enable_triggerbot", False)

    while not shutdown_event.is_set():
        if paused:
            time.sleep(0.1)  # 暂停时必须休眠
            continue

        loop_start = time.perf_counter()
        dt = loop_start - last_loop_time
        last_loop_time = loop_start

        try:
            # 1. World Model Step (CPU Bound)
            world_model.step(ctx, ring_buffer)

            # 2. Decision
            if not ctx.is_valid:
                recorder.record_frame(ctx)
                # 丢失目标时重置调参状态
                last_target_id = -1
                continue

            pred_x, pred_y = ctx.p_predict
            target_dx = pred_x - sc_x
            target_dy = pred_y - sc_y

            # Strategy (Pure Math)
            if hasattr(strategy, 'calculate_mouse_move'):
                res = strategy.calculate_mouse_move(target_dx, target_dy)
                if isinstance(res, tuple):
                    intent_x, intent_y = res
                else:
                    intent_x, intent_y = res.raw_counts_x, res.raw_counts_y
                    ctx.strategy_result = res
            else:
                intent_x, intent_y = target_dx, target_dy

            # 3. Controller (Math + State)
            final_x, final_y = controller.compute(intent_x, intent_y, dt)

            # 4. Trigger Check
            trigger_target = {
                "screen_x": pred_x,
                "screen_y": pred_y,
                "conf": 1.0
            }
            should_fire = controller.should_trigger(trigger_target, (sc_x, sc_y))

            # 5. Output (IO Bound)
            # 移动
            if abs(final_x) > 0.5 or abs(final_y) > 0.5:
                output_device.mouse_xy(final_x, final_y)

                # --- [Auto-Tune] 自校准闭环反馈 ---
                if config.getbool("AimStrategy", "enable_auto_calibration", False):

                    # 只有当连续锁定同一个目标时，才能计算有效位移
                    if ctx.selected_id == last_target_id and last_target_id != -1:

                        # 获取当前帧目标的原始检测坐标 (不使用预测值)
                        curr_det = next((t for t in ctx.targets if t.class_id == ctx.selected_id), None)

                        if curr_det:
                            curr_x, curr_y = curr_det.x, curr_det.y
                            prev_x, prev_y = last_target_xy

                            # [果] 目标在屏幕上的实际位移 (Pixels)
                            actual_pixel_dx = curr_x - prev_x
                            actual_pixel_dy = curr_y - prev_y

                            # [因] 查询上一帧到现在的鼠标指令 (Counts)
                            # 必须使用 t_cap 对齐时间轴
                            ai_counts_x, ai_counts_y = ring_buffer.get_cursor_delta_sum(last_frame_time, ctx.t_cap)

                            # 只有发生显著移动时才校准，避免静止噪声
                            if abs(ai_counts_x) > 5 or abs(ai_counts_y) > 5:
                                if hasattr(strategy, 'feedback_update'):
                                    # 喂给 Strategy：鼠标发了 +Counts，导致物体移了 -Pixels
                                    # Strategy 内部会处理绝对值，这里直接传原始差值
                                    strategy.feedback_update(
                                        ai_counts_x, -actual_pixel_dx,
                                        ai_counts_y, -actual_pixel_dy
                                    )

                            # 更新状态
                            last_target_xy = (curr_x, curr_y)
                            last_frame_time = ctx.t_cap

                    # 状态维护：初始化或切换目标
                    elif ctx.selected_id != -1:
                        curr_det = next((t for t in ctx.targets if t.class_id == ctx.selected_id), None)
                        if curr_det:
                            last_target_id = ctx.selected_id
                            last_target_xy = (curr_det.x, curr_det.y)
                            last_frame_time = ctx.t_cap
                    else:
                        last_target_id = -1

            # 开火
            if should_fire and enable_trigger:
                output_device.mouse_down(1)
                time.sleep(random.uniform(0.015, 0.03))
                output_device.mouse_up(1)
                time.sleep(0.04)

            # 6. Trace
            recorder.record_frame(ctx)

        except Exception as e:
            print(f"[Loop] 异常: {e}")
            time.sleep(0.01)

    # --- 退出 ---
    print("[System] 正在关闭...")
    recorder.save_to_disk()
    keyboard.unhook_all()
    capture_thread.join(timeout=1.0)
    inference_thread.join(timeout=1.0)
    print("[System] Bye.")


if __name__ == "__main__":
    main()