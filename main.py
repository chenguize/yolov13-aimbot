# main.py - Phase 3 System Coordinator
# 核心职责：
# 1. 启动并编排所有异步子系统 (Capture, Inference, Input, Output)
# 2. 维护 RingBuffer 因果闭环
# 3. 执行 "感知(World) -> 决策(Strategy) -> 控制(Controller) -> 执行(Output)" 循环
# 4. (新增) 决策追踪回放录制

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
# [Internal Class] InputMonitor (Phase 3 关键组件)
# 职责：监听物理鼠标输入，过滤掉带水印的 AI 输入，将人类输入写入 RingBuffer
# ==============================================================================
class InputMonitor(threading.Thread):
    def __init__(self, ring_buffer: RingBuffer):
        super().__init__(name="InputMonitor", daemon=True)
        self.ring_buffer = ring_buffer
        self.hook = None

        # 定义 C Types
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
                ('dwExtraInfo', ctypes.c_ulong)  # 关键字段
            ]

        last_x, last_y = 0, 0
        first_run = True

        # 钩子回调函数
        def hook_proc(nCode, wParam, lParam):
            nonlocal last_x, last_y, first_run

            if nCode >= 0 and wParam == self.WM_MOUSEMOVE:
                struct = ctypes.cast(lParam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents

                # [核心逻辑] 敌我识别
                # 如果 ExtraInfo == AI_SIGNATURE，说明是 Output 模块发出的
                # 直接放行，且【绝对不】写入 RingBuffer (防止双重记录)
                if struct.dwExtraInfo == AI_SIGNATURE:
                    pass
                else:
                    # 这是人类的手搓操作
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

        # 初始化当前位置
        pt = wintypes.POINT()
        self.user32.GetCursorPos(byref(pt))
        last_x, last_y = pt.x, pt.y

        # 安装钩子
        pointer = self.HOOKPROC(hook_proc)
        self.hook = self.user32.SetWindowsHookExA(
            self.WH_MOUSE_LL,
            pointer,
            windll.kernel32.GetModuleHandleW(None),
            0
        )

        # 消息循环 (阻塞直到收到 WM_QUIT 或线程结束)
        msg = wintypes.MSG()
        # PeekMessage 非阻塞方式配合 shutdown event，或者 GetMessage 阻塞
        # 这里用 GetMessage 因为是独立线程
        while not shutdown_event.is_set():
            if self.user32.PeekMessageA(byref(msg), None, 0, 0, 1) != 0:  # PM_REMOVE
                self.user32.TranslateMessage(byref(msg))
                self.user32.DispatchMessageA(byref(msg))
            else:
                time.sleep(0.001)

        # 卸载钩子
        if self.hook:
            self.user32.UnhookWindowsHookEx(self.hook)
        print("[Input] 监听结束")


# ==============================================================================
# [Helpers] 信号与控制
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
    print(" YOLOv13 Aimbot System (Phase 3 Architecture)")
    print(" World Model | Causal Hedging | Intent Arbitration")
    print("=" * 60)

    # 2. 核心数据结构初始化
    # [Structure A] 因果账本
    ring_buffer = RingBuffer(max_duration=2.0)

    # [Bus] 视觉通道
    frame_bus = FrameBus()

    # [Recorder] 追踪回放 (Phase 3 新增)
    recorder = TraceRecorder(save_path="debug_trace.pkl")
    if config.getbool("Debug", "enable_trace", False):  # 需在 config 开启
        recorder.enable()

    # 3. 依赖注入 (Dependency Injection)
    output_device.set_ring_buffer(ring_buffer)
    input_monitor = InputMonitor(ring_buffer)
    world_model = WorldModel()

    # 4. 模块工厂
    strategy = create_aim_strategy()
    controller = get_controller()

    # 5. 启动线程
    capture_thread = CaptureThread(frame_bus, shutdown_event)
    inference_thread = InferenceThread(frame_bus, world_model, shutdown_event)
    input_monitor.start()
    capture_thread.start()
    inference_thread.start()

    # 6. 快捷键
    keyboard.add_hotkey('p', toggle_pause)

    # 获取屏幕中心
    sc_x = config.getint("General", "screen_width", 1920) // 2
    sc_y = config.getint("General", "screen_height", 1080) // 2

    print("[System] 所有子系统已就绪。进入主控制循环 (1000Hz)...")

    # ==========================================================================
    # [Loop] 主控制循环 (The "Cognition" Loop)
    # ==========================================================================
    ctx = InferenceContext()
    last_loop_time = time.perf_counter()

    while not shutdown_event.is_set():
        loop_start = time.perf_counter()
        dt = loop_start - last_loop_time
        last_loop_time = loop_start

        if paused:
            time.sleep(0.1)
            continue

        try:
            # ---------------------------------------------------------
            # Step 1: 世界状态步进 (World Model Step)
            # ---------------------------------------------------------
            world_model.step(ctx, ring_buffer)

            # ---------------------------------------------------------
            # Step 2: 决策层 (Decision / Strategy)
            # ---------------------------------------------------------
            if not ctx.is_valid:
                # 录制无效帧用于分析为何丢失
                recorder.record_frame(ctx)
                time.sleep(0.001)
                continue

            # 获取预测点 (Strategy 只需要知道"目标在哪")
            pred_x, pred_y = ctx.p_predict
            target_dx = pred_x - sc_x
            target_dy = pred_y - sc_y

            # Strategy: 纯几何映射
            if hasattr(strategy, 'calculate_mouse_move'):
                # Phase 3 新接口返回 StrategyResult 对象或 tuple
                res = strategy.calculate_mouse_move(target_dx, target_dy)
                if isinstance(res, tuple):
                    intent_x, intent_y = res
                    # 补充 context 用于录制
                    # ctx.strategy_result = StrategyResult(...) # 如果需要详细录制
                else:
                    # 假设返回 StrategyResult 对象
                    intent_x, intent_y = res.raw_counts_x, res.raw_counts_y
                    ctx.strategy_result = res
            else:
                intent_x, intent_y = target_dx, target_dy

            # ---------------------------------------------------------
            # Step 3: 控制层 (Controller)
            # ---------------------------------------------------------
            final_x, final_y = controller.compute(intent_x, intent_y, dt)

            trigger_target = {
                "screen_x": pred_x,
                "screen_y": pred_y,
                "conf": 1.0
            }
            should_fire = controller.should_trigger(trigger_target, (sc_x, sc_y))

            # ---------------------------------------------------------
            # Step 4: 执行层 (Output & Feedback)
            # ---------------------------------------------------------
            if abs(final_x) > 0.5 or abs(final_y) > 0.5:
                output_device.mouse_xy(final_x, final_y)

                # 自校准反馈 (可选)
                if hasattr(strategy, 'feedback_update'):
                    # 这里简化处理，实际应传入 RingBuffer 历史
                    pass

            if should_fire and config.getbool("General", "enable_triggerbot", False):
                output_device.mouse_down(1)
                time.sleep(random.uniform(0.01, 0.03))
                output_device.mouse_up(1)
                time.sleep(0.05)

            # ---------------------------------------------------------
            # Step 5: 追踪录制 (Trace Replay)
            # ---------------------------------------------------------
            # 录制这一帧的所有状态（世界、策略、控制）
            recorder.record_frame(ctx)

        except Exception as e:
            print(f"[Loop] 异常: {e}")
            time.sleep(0.01)

        time.sleep(0.0005)

    # --- 退出清理 ---
    print("[System] 正在关闭...")
    recorder.save_to_disk()  # 保存录制数据
    keyboard.unhook_all()
    capture_thread.join(timeout=1.0)
    inference_thread.join(timeout=1.0)
    print("[System] Bye.")


if __name__ == "__main__":
    main()