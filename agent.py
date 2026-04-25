# agent.py
# ═══════════════════════════════════════════════════════════════════════════════
# 实战 Agent —— Perception / Inference / WorldModel / Controller 组装与 tick 驱动
# ═══════════════════════════════════════════════════════════════════════════════
#
# 线程拓扑：
#   CaptureThread        —— dxcam @ target_fps，写 FrameBus → set frame_ready
#   InferenceThread      —— 消费 FrameBus，YOLO + 推理，写 WorldModel
#   MouseWorker          —— 1000Hz 消费 controller.tick_mouse()，下发 SendInput
#   HumanMouseListener   —— 阻塞式 RawInput，记录人类物理位移到 RingBuffer
#   TriggerWorker        —— 单独线程处理 click down/up，主 tick 不再被 sleep 阻塞
#   Main loop (tick)     —— 每帧调 world_model.step + controller.compute
#
# 与 sim_agent 的对应：
#   实战 tick() ≈ sim_agent.step()
#   实战 agent 遵循同样的 chase_mode 判定：
#     · 最近 100ms 人类速度 > 500 px/s → human_flick (人拉枪，AI 补枪微调)
#     · 否则 → pure_ai (目标自己进 FOV，AI 独立瞄准)

import logging
import math
import random
import threading
import time

import numpy as np
import win32api
from ctypes import windll

from config import config
from inference import InferenceThread
from output import gHub as output_device
from perception.bus import FrameBus
from perception.capture import CaptureThread
from perception.ring_buffer import RingBuffer
from utils.recorder import TraceRecorder
from utils.types import InferenceContext
from world_model import WorldModel

logger = logging.getLogger("Agent")


# ══════════════════════════════════════════════════════════════════════════════
# § 1 │ MovementTracker —— 人物急停状态
# ══════════════════════════════════════════════════════════════════════════════
class MovementTracker:
    """监控 WASD 按键，判断 '已停止位移 + 滑行衰减' 窗口。"""

    def __init__(self):
        self.W, self.A, self.S, self.D = 0x57, 0x41, 0x53, 0x44
        # Valorant 0.08-0.12s，CS2 0.15s；这里 0.10 偏均衡
        self.stop_cooldown_duration = 0.10
        self.last_moving_time = 0.0

    def is_accurate_to_shoot(self) -> bool:
        keys = (self.W, self.A, self.S, self.D)
        currently_moving = any(win32api.GetAsyncKeyState(k) & 0x8000 for k in keys)
        current_time = time.perf_counter()
        if currently_moving:
            self.last_moving_time = current_time
            return False
        return (current_time - self.last_moving_time) >= self.stop_cooldown_duration


# ══════════════════════════════════════════════════════════════════════════════
# § 2 │ HumanMouseListener —— 物理鼠标输入监听
# ══════════════════════════════════════════════════════════════════════════════
class HumanMouseListener(threading.Thread):
    """
    阻塞式 RawInput 采样。把物理位移（is_ai=False）写入 RingBuffer，
    供人机离合器 / 反馈校准使用。
    """

    def __init__(self, ring_buffer: RingBuffer, shutdown_evt: threading.Event):
        super().__init__(name="RawInputListener", daemon=True)
        self.ring_buffer = ring_buffer
        self.shutdown_evt = shutdown_evt

    def run(self):
        logger.info("RawInput Mouse Listener starting")
        try:
            from inputs import get_mouse
        except ImportError:
            logger.error("缺少依赖 'inputs'，请 pip install inputs；人机离合器不可用")
            return
        logger.info("RawInput: inputs 已加载，人类鼠标位移会写入 RingBuffer")

        # 某些 inputs 版本无 UnpluggedError，用 OSError 兜底
        try:
            from inputs import UnpluggedError
        except (ImportError, AttributeError):
            UnpluggedError = OSError

        while not self.shutdown_evt.is_set():
            try:
                events = get_mouse()
                dx = dy = 0
                for event in events:
                    if event.ev_type == 'Relative':
                        if event.code == 'REL_X':
                            dx += event.state
                        elif event.code == 'REL_Y':
                            dy += event.state
                if dx or dy:
                    self.ring_buffer.add_event(dx, dy, is_ai=False)
            except UnpluggedError:
                time.sleep(1.0)
            except Exception as e:
                logger.debug("HumanMouseListener transient: %s", e)
                time.sleep(0.001)
        logger.info("RawInput Mouse Listener stopped")


# ══════════════════════════════════════════════════════════════════════════════
# § 3 │ MouseWorker —— 1000Hz 输出循环
# ══════════════════════════════════════════════════════════════════════════════
class MouseWorker(threading.Thread):
    """1kHz 主 tick + 高斯抖动（±100μs 打破机器特征）。"""

    def __init__(self, controller, output_dev, shutdown_evt: threading.Event):
        super().__init__(name="MouseWorker", daemon=True)
        self.controller = controller
        self.output = output_dev
        self.shutdown_evt = shutdown_evt

    def run(self):
        logger.info("MouseWorker (1000Hz) starting")
        try:
            windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

        base_period = 0.001
        while not self.shutdown_evt.is_set():
            loop_start = time.perf_counter()
            jitter = random.gauss(0, 0.0001)
            target_period = max(0.0008, min(0.0012, base_period + jitter))

            try:
                dx, dy = self.controller.tick_mouse()
                if dx or dy:
                    self.output.mouse_xy(dx, dy)
            except Exception as e:
                logger.debug("MouseWorker tick error: %s", e)

            elapsed = time.perf_counter() - loop_start
            remaining = target_period - elapsed
            if remaining > 0.0005:
                time.sleep(remaining - 0.0002)
            while (time.perf_counter() - loop_start) < target_period:
                pass

        try:
            windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        logger.info("MouseWorker stopped")


# ══════════════════════════════════════════════════════════════════════════════
# § 4 │ TriggerWorker —— 异步射击
# ══════════════════════════════════════════════════════════════════════════════
class TriggerWorker(threading.Thread):
    """
    把 mouse_down/sleep/mouse_up 异步化，避免 sleep 阻塞主 tick。
    调用方只需 fire(click_duration_s)；类内保证 15~60ms 点击时长。
    """

    def __init__(self, output_dev, shutdown_evt: threading.Event):
        super().__init__(name="TriggerWorker", daemon=True)
        self.output = output_dev
        self.shutdown_evt = shutdown_evt
        self._fire_evt = threading.Event()
        self._click_duration = 0.03

    def fire(self, click_duration: float):
        self._click_duration = float(np.clip(click_duration, 0.015, 0.06))
        self._fire_evt.set()

    def run(self):
        logger.info("TriggerWorker starting")
        while not self.shutdown_evt.is_set():
            if not self._fire_evt.wait(timeout=0.1):
                continue
            self._fire_evt.clear()
            try:
                self.output.mouse_down(1)
                time.sleep(self._click_duration)
                self.output.mouse_up(1)
            except Exception as e:
                logger.debug("TriggerWorker fire error: %s", e)
        logger.info("TriggerWorker stopped")


# ══════════════════════════════════════════════════════════════════════════════
# § 5 │ AIAgent —— 主驱动
# ══════════════════════════════════════════════════════════════════════════════
class AIAgent:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.paused = False
        self.enable_aimbot = True

        self.ring_buffer = RingBuffer(max_duration=2.0)
        self.frame_bus = FrameBus()
        output_device.set_ring_buffer(self.ring_buffer)

        self.recorder = TraceRecorder(save_path="debug_trace.pkl")
        if config.getbool("Debug", "enable_trace", False):
            self.recorder.enable()

        self.world_model = WorldModel()
        self.controller = self.world_model.controller
        self.aim_strategy = self.world_model.strategy

        self.cap_to_inf_event = threading.Event()

        self.capture_thread = CaptureThread(self.frame_bus, self.shutdown_event, self.cap_to_inf_event)
        self.inference_thread = InferenceThread(
            self.frame_bus, self.world_model, self.shutdown_event, self.cap_to_inf_event
        )
        self.mouse_worker = MouseWorker(self.controller, output_device, self.shutdown_event)
        self.human_mouse_listener = HumanMouseListener(self.ring_buffer, self.shutdown_event)
        self.trigger_worker = TriggerWorker(output_device, self.shutdown_event)

        self.ctx = InferenceContext()
        self.crop_center = self.world_model.crop_center

        self.loop_counter = 0
        self.last_stat_time = time.perf_counter()
        self.last_tick_time = time.perf_counter()
        self.frames_in_cycle = 0

        self.enable_trigger = config.getbool("General", "enable_triggerbot", True)
        self.trigger_fov = config.getfloat("Triggerbot", "trigger_fov_x", 3.0)
        self.trigger_conf = config.getfloat("Triggerbot", "trigger_confidence", 0.6)
        self.last_shot_time = 0.0

        # ── 目标生命周期 ──
        self.target_first_seen_time = 0.0
        self.target_in_crosshair_time = 0.0
        self.is_target_in_crosshair = False

        # ── 人机 Flick 状态检测 ──
        self._prev_human_flicking = False

        # ── 当前目标的 chase_mode（由首帧决定，丢失后下一个目标重判定）──
        self._current_chase_mode: str = 'pure_ai'

        self.movement_tracker = MovementTracker()
        # 首帧前 last_tick 距当前可达数百 ms，若参与 dt>0.1 判定会误整段不 compute
        self._first_tick: bool = True
        # conf / is_valid 单帧毛刺 若立刻 target_first_seen=0 → 每帧都 reset_target_state → 准星旁摆
        self._invalid_streak: int = 0

        logger.info("Agent core constructed (WorldModel+threads configured, not started)")
        self._log_runtime_summary()

    def _freeze_mouse_motion(self):
        c = self.controller
        if hasattr(c, "freeze_output_integrators"):
            c.freeze_output_integrators()

    def _log_runtime_summary(self) -> None:
        if not config.getbool("Debug", "startup_diag", True):
            return
        st = self.aim_strategy
        bp = getattr(st, "bypass_mapping", None)
        cap = int(self.crop_center * 2) if self.crop_center is not None else config.getint("General", "capture_size", 256)
        mac = config.getfloat("General", "min_aim_conf", 0.32)
        mdr = min(mac - 1e-3, config.getfloat("General", "min_aim_conf_drop", 0.20))
        ninv = max(1, int(config.getint("General", "aim_drop_invalid_frames", 2)))
        logger.info(
            "Runtime | capture=%dpx | inf_conf>=%.2f | min_aim=%.2f (drop<%.2f) inv_n=%d | strategy_bypass=%s | model=%s",
            cap,
            config.getfloat("Inference", "conf_threshold", 0.4),
            mac,
            mdr,
            ninv,
            bp,
            config.getstr("Inference", "model_path", ""),
        )
        logger.info(
            "Pipeline | [Capture+cap_event] -> Inference -> update_detections&frame_ready -> main.tick; "
            "worker threads: Mouse1000Hz, RawInput, Trigger"
        )

    # ────────────────────────────────────────────────────────────────────
    def start(self):
        self.capture_thread.start()
        self.inference_thread.start()
        self.mouse_worker.start()
        self.human_mouse_listener.start()
        self.trigger_worker.start()
        logger.info("All worker threads started (order: capture, inference, then mouse/listener/trigger)")

    def stop(self):
        self.shutdown_event.set()
        self.cap_to_inf_event.set()
        self.world_model.frame_ready_event.set()
        time.sleep(0.5)
        self.recorder.save_to_disk()
        logger.info("Shutdown sequence completed")

    # ────────────────────────────────────────────────────────────────────
    # Main tick — 由 frame_ready_event 驱动，通常 ~200-300Hz（随推理帧率）
    # ────────────────────────────────────────────────────────────────────
    def tick(self):
        wfe = self.world_model.frame_ready_event
        tmo = config.getfloat("Debug", "tick_wait_timeout_sec", 2.0)
        if tmo > 0:
            if not wfe.wait(timeout=tmo):
                if self.shutdown_event.is_set():
                    return
                now_log = time.perf_counter()
                if now_log - getattr(self, "_last_stall_log", 0.0) >= 1.0:
                    logger.warning(
                        "main.tick: %.0fs 内无新推理包（未收到 frame_ready）。"
                        "采图/推理停住或极慢时会出现；有目标后才会移动鼠标。",
                        tmo,
                    )
                    self._last_stall_log = now_log
                self._freeze_mouse_motion()
                time.sleep(0.02)
                return
        else:
            wfe.wait()
        wfe.clear()

        now = time.perf_counter()
        if self.paused:
            self._freeze_mouse_motion()
            self.last_tick_time = now
            return

        self.world_model.step(self.ctx, self.ring_buffer)

        dt = now - self.last_tick_time
        self.last_tick_time = now
        if self._first_tick:
            self._first_tick = False
        elif dt <= 0 or dt > 0.1:
            # 异常 dt（首帧外）：关 emit
            self._freeze_mouse_motion()
            return

        # ── 目标有效性判定 ───────────────────────────────────────────────
        # Bug J 修：p_predict 是 tuple，tuple 永远 truthy。显式 is None 判断
        invalid = (not self.ctx.is_valid) or (self.ctx.p_predict is None)
        if invalid:
            self._invalid_streak += 1
        else:
            self._invalid_streak = 0

        if invalid:
            n_inv_drop = max(1, int(config.getint("General", "aim_drop_invalid_frames", 2)))
            # 已锁时：短无效帧只 freeze 不拆锁，避免丢框一帧就 reset 控制器 → 贴脸来回摆
            if self.target_first_seen_time > 0.0 and self._invalid_streak < n_inv_drop:
                self._freeze_mouse_motion()
                return
            self._freeze_mouse_motion()
            self.target_first_seen_time = 0.0
            self.is_target_in_crosshair = False
            return

        # conf 滞回：未锁时须 ≥ min_aim_conf；已锁时须 ≥ min_aim_conf_drop 才继续，否者拆锁
        # 单阈值时 conf 在 0.30~0.38 间抖会每帧「丢→锁→reset」→ 日志狂刷 Target acquired
        min_aim = config.getfloat("General", "min_aim_conf", 0.32)
        min_drop = min(
            min_aim - 1e-3,
            config.getfloat("General", "min_aim_conf_drop", 0.20),
        )
        if self.target_first_seen_time == 0.0:
            if self.ctx.conf < min_aim:
                self._freeze_mouse_motion()
                return
        else:
            if self.ctx.conf < min_drop:
                self._freeze_mouse_motion()
                self.target_first_seen_time = 0.0
                self.is_target_in_crosshair = False
                return

        c = self.controller
        if hasattr(c, "set_mouse_emit"):
            c.set_mouse_emit(True)

        # ── 首次见到目标：决定 chase_mode 并 reset controller ───────────
        first_frame = (self.target_first_seen_time == 0.0)
        if first_frame:
            self.target_first_seen_time = now
            # Bug B 修：按最近 100ms 人类速度判定模式
            #   实战大部分场景是"人拉枪到 256px 内 AI 接管" → human_flick
            #   目标自己走进 FOV（被动追）→ pure_ai
            dx_100, dy_100 = self.ring_buffer.get_pure_human_delta_sum(now - 0.1, now)
            recent_speed = math.hypot(dx_100, dy_100) / 0.1
            self._current_chase_mode = 'human_flick' if recent_speed > 500.0 else 'pure_ai'
            if hasattr(self.controller, 'reset_target_state'):
                try:
                    self.controller.reset_target_state(mode=self._current_chase_mode)
                except TypeError:
                    self.controller.reset_target_state()
            logger.info("Target acquired (mode=%s, human_speed_100ms=%.0f)",
                        self._current_chase_mode, recent_speed)

        dx_h_inst, dy_h_inst = self.ring_buffer.get_pure_human_delta_sum(now - dt, now)
        human_vx_inst = dx_h_inst / dt if dt > 0 else 0.0
        human_vy_inst = dy_h_inst / dt if dt > 0 else 0.0

        bbox_w = self.ctx.targets[0].w if self.ctx.targets else 60.0

        # ── 视觉坐标映射 ──────────────────────────────────────────────
        # Bug F 修：实战路径不叠 perlin drift。
        # drift 的初衷是"sim 里模拟人手抖"，但实战 Kalman 已经做了平滑，
        # 再叠 ±3px 抖动反而让精度掉 15-30%。拟人特征完全由 controller 的
        # OU tremor + postural drift 负责（在 tick_mouse 里，有物理意义的谱形）。
        p_x = self.ctx.p_predict[0]
        p_y = self.ctx.p_predict[1]

        intent_x, intent_y = self.aim_strategy.calculate_mouse_move(p_x, p_y, bbox_w=bbox_w)

        v_real_pixels = self.ctx.v_real
        intent_vx, intent_vy = self.aim_strategy.calculate_velocity_move(
            v_real_pixels[0], v_real_pixels[1], bbox_w=bbox_w
        )
        a_real_pixels = getattr(self.ctx, 'a_real', (0.0, 0.0))
        intent_ax, intent_ay = self.aim_strategy.calculate_velocity_move(
            a_real_pixels[0], a_real_pixels[1], bbox_w=bbox_w
        )

        pixel_error_dist = math.hypot(p_x, p_y)

        # ── power_factor = spatial × reaction × human_override ─────────
        spatial_factor = float(np.clip(1.0 - (pixel_error_dist / 800.0) ** 2, 0.1, 1.0))

        time_since_seen = now - self.target_first_seen_time
        if self._current_chase_mode == 'pure_ai':
            reaction_factor = 1.0  # AI 独立瞄准，无视觉反应斜坡
        else:
            reaction_factor = float(np.clip(time_since_seen / 0.15, 0.0, 1.0))

        # 人机动态离合器：人速度大时 AI 让位
        dx_h_recent, dy_h_recent = self.ring_buffer.get_pure_human_delta_sum(now - 0.1, now)
        human_speed = math.hypot(dx_h_recent, dy_h_recent) / 0.1

        if pixel_error_dist < 40.0:
            s_min, s_max = 1200.0, 2500.0
        elif pixel_error_dist > 150.0:
            s_min, s_max = 150.0, 800.0
        else:
            progress = (150.0 - pixel_error_dist) / 110.0
            s_min = 150.0 + progress * 1050.0
            s_max = 800.0 + progress * 1700.0

        if human_speed > s_max:
            human_override = 0.0
        elif human_speed < s_min:
            human_override = 1.0
        else:
            human_override = 1.0 - (human_speed - s_min) / (s_max - s_min)

        power_factor = spatial_factor * reaction_factor * human_override

        # ── 人类甩枪结束沿检测 → 通知控制器清积分 ──────────────────────
        cur_human_flicking = human_speed > (s_max * 0.7)
        if self._prev_human_flicking and not cur_human_flicking:
            if hasattr(self.controller, 'notify_flick_end'):
                self.controller.notify_flick_end()
        self._prev_human_flicking = cur_human_flicking

        # ── 调用控制器 ────────────────────────────────────────────────
        self.controller.compute(
            target_x=intent_x, target_y=intent_y, dt=dt,
            v_real=np.array([intent_vx, intent_vy]),
            a_real=np.array([intent_ax, intent_ay]),
            power_factor=power_factor,
            bbox_w=bbox_w,
        )

        if config.getbool("Debug", "aim_diagnostics", False) or config.getbool("Debug", "aim_diag_warn_only", False):
            from utils.aim_diagnostics import aim_diag
            _d = aim_diag()
            ar = np.asarray(self.controller.crosshair_velocity, dtype=np.float64)
            _d.record(
                px=p_x,
                py=p_y,
                intent_x=intent_x,
                intent_y=intent_y,
                power=power_factor,
                human_override=human_override,
                spatial=spatial_factor,
                reaction=reaction_factor,
                arm_vx=float(ar.flat[0]),
                arm_vy=float(ar.flat[1]),
                mode=str(getattr(self.controller, "mode", "?")),
                lead_ms=float(getattr(self.ctx, "dynamic_lag_ms", 0.0)),
                vh_s=float(getattr(self.world_model, "dynamic_vh_latency", 0.0)),
                inf_ema=float(getattr(self.world_model, "inference_ms_ema", 0.0)),
                dt=dt,
                chase=self._current_chase_mode,
                bypass_map=bool(getattr(self.aim_strategy, "bypass_mapping", False)),
            )
            _d.maybe_emit()

        if self.enable_aimbot:
            self._check_and_trigger()

        self.frames_in_cycle += 1
        self._print_stats()

    # ────────────────────────────────────────────────────────────────────
    def _check_and_trigger(self):
        if self._check_trigger_condition() and self.movement_tracker.is_accurate_to_shoot():
            self._perform_shoot()

    def _check_trigger_condition(self) -> bool:
        if not (self.enable_aimbot and self.enable_trigger and self.ctx.is_valid):
            self.is_target_in_crosshair = False
            return False
        if self.ctx.conf < self.trigger_conf:
            self.is_target_in_crosshair = False
            return False
        if self.ctx.p_predict is None:
            self.is_target_in_crosshair = False
            return False

        tx, ty = self.ctx.p_predict
        if abs(tx) <= self.trigger_fov and abs(ty) <= self.trigger_fov:
            if not self.is_target_in_crosshair:
                self.is_target_in_crosshair = True
                self.target_in_crosshair_time = time.perf_counter()
            return True
        self.is_target_in_crosshair = False
        return False

    def _perform_shoot(self):
        now = time.perf_counter()
        dx_sum, dy_sum = self.ring_buffer.get_pure_human_delta_sum(now - 0.2, now)
        human_movement_dist = math.hypot(dx_sum, dy_sum)

        raw_reaction = np.random.gamma(shape=8.0, scale=0.0125)
        base_delay = float(np.clip(raw_reaction, 0.05, 0.35))
        activity_factor = min(human_movement_dist / 80.0, 1.0)
        dynamic_delay = base_delay * (1.0 - activity_factor)

        if now - self.target_in_crosshair_time < dynamic_delay:
            return
        if now - self.last_shot_time < 0.15:
            return

        # Bug C 修：异步 fire，主 tick 立即返回，不再被 30ms sleep 阻塞
        raw_click = random.gauss(0.03, 0.005)
        self.trigger_worker.fire(raw_click)
        self.last_shot_time = now

    # ────────────────────────────────────────────────────────────────────
    def _print_stats(self):
        now = time.perf_counter()
        dt = now - self.last_stat_time
        if dt < 1.0:
            return
        real_fps = self.frames_in_cycle / dt
        lag = getattr(self.ctx, 'dynamic_lag_ms', 0.0)
        conf = self.ctx.conf
        n_dets = len(self.ctx.targets) if self.ctx.targets else 0
        inf_ema = getattr(self.world_model, "inference_ms_ema", 0.0)
        if config.getbool("Debug", "detailed_stats", True):
            logger.info(
                "Tick | fps=%.1f | infer_ema=%.1fms | valid=%s | dets=%d | conf=%.2f | lat=%.1fms | mode=%s | paused=%s | aim_on=%s",
                real_fps, inf_ema, self.ctx.is_valid, n_dets, conf, lag,
                self._current_chase_mode, self.paused, self.enable_aimbot,
            )
        else:
            logger.info("FPS: %.1f | Lat: %.1fms | Conf: %.2f | Mode: %s",
                        real_fps, lag, conf, self._current_chase_mode)
        self.last_stat_time = now
        self.frames_in_cycle = 0

    def toggle_pause(self):
        self.paused = not self.paused
        logger.warning("PAUSED: %s", self.paused)

    def toggle_aimbot(self):
        self.enable_aimbot = not self.enable_aimbot
        if not self.enable_aimbot:
            self._freeze_mouse_motion()
        logger.warning("AIMBOT: %s", self.enable_aimbot)
