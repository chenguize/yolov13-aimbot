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

import math
import threading
import time
from typing import Tuple
import numpy as np

from config import config
from output import gHub as output_device
from perception.bus import FrameBus
from perception.capture import CaptureThread
from perception.human_input import HumanMouseListener
from perception.ring_buffer import RingBuffer
from utils import runtime_defaults as _rtd
from utils.human_intent import HumanIntentTracker
from utils.logger import get_logger
from utils.recorder import TraceRecorder
from utils.types import InferenceContext
from utils.workers import MovementTracker, MouseWorker, TriggerWorker
from world_model import WorldModel
from controllers.pro_controller import CIPHER_KERNEL_BACKEND

logger = get_logger("Agent")

# 与 inference_aimlab 对应；大小写不敏感时在外层 .lower() 后比较
_AIMLAB_BACKENDS = frozenset({
    "aimlab", "aimlab_ball", "ball", "aymlab", "opencv", "opencv_aimlab",
    "hsv", "color", "colour",
})


# ══════════════════════════════════════════════════════════════════════════════
# § 1 │ AIAgent —— 主驱动
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
        _inf_be = (config.getstr("Inference", "backend", "yolo") or "yolo").strip().lower()
        if _inf_be in _AIMLAB_BACKENDS:
            from inference_aimlab import AimlabBallInferenceThread
            self.inference_thread = AimlabBallInferenceThread(
                self.frame_bus, self.world_model, self.shutdown_event, self.cap_to_inf_event
            )
            self._is_aimlab_backend = True
        else:
            from inference import InferenceThread
            self.inference_thread = InferenceThread(
                self.frame_bus, self.world_model, self.shutdown_event, self.cap_to_inf_event
            )
            self._is_aimlab_backend = False
        self.mouse_worker = MouseWorker(self.controller, output_device, self.shutdown_event)
        if id(self.mouse_worker.output) != id(output_device):
            logger.error(
                "gHub 实例不一致: mouse_worker.output 与 from output gHub 不同 id"
            )
        self.human_mouse_listener = HumanMouseListener(self.ring_buffer, self.shutdown_event)
        self.trigger_worker = TriggerWorker(output_device, self.shutdown_event)

        self.ctx = InferenceContext()
        self.crop_center = self.world_model.crop_center

        self.loop_counter = 0
        self.last_stat_time = time.perf_counter()
        self.last_tick_time = time.perf_counter()
        self.frames_in_cycle = 0

        self.enable_trigger = config.getbool("General", "enable_triggerbot", False)
        self.trigger_fov = config.getfloat("Triggerbot", "trigger_fov_x", 3.0)
        self.trigger_conf = config.getfloat("Triggerbot", "trigger_conf_threshold", 0.50)
        self.last_shot_time = 0.0

        # ── 目标生命周期 ──
        self.target_first_seen_time = 0.0
        self.target_in_crosshair_time = 0.0
        self.is_target_in_crosshair = False
        # ── 多帧确认计数器：首次锁目标前需连续 N 帧有效（防单帧噪声/误检锁假目标）──
        self._lock_confirm_count: int = 0

        # ── 人机 Flick 状态检测 ──
        self._prev_human_flicking = False

        # ── 当前目标的 chase_mode（由首帧决定，丢失后下一个目标重判定）──
        self._current_chase_mode: str = 'pure_ai'
        # ── chase_mode 切换确认计数器（连续 N 帧满足条件才切换，防边界震荡）──
        self._mode_switch_confirm: int = 0

        self.movement_tracker = MovementTracker()
        # 首帧前 last_tick 距当前可达数百 ms，若参与 dt>0.1 判定会误整段不 compute
        self._first_tick: bool = True
        # conf / is_valid 单帧毛刺 若立刻 target_first_seen=0 → 每帧都 reset_target_state → 准星旁摆
        self._invalid_streak: int = 0
        self._n_target_acquire_logs: int = 0
        self._aim_lock_diag: bool = config.getbool("Debug", "aim_lock_diag", False)
        self._last_move_emit_log: float = 0.0
        # ── reaction_factor 计时起点：flick_end 时刻，非 target_first_seen ──
        self._flick_end_time: float = 0.0
        # 人机权包络状态(与 s_min/s_max 带统一, 无 pure_ai_bypass 旁路)
        self._intent_tracker = HumanIntentTracker()
        self._human_clutch_diag_last = 0.0
        self._pipeline_lat_log_last = 0.0

        logger.info("Agent core constructed (WorldModel+threads configured, not started)")
        self._log_runtime_summary()

    def _freeze_mouse_motion(self):
        c = self.controller
        if hasattr(c, "freeze_output_integrators"):
            c.freeze_output_integrators()

    def _sync_aimbot_move_block(self) -> None:
        """已废弃到点停手门控，恒不拦 aim 相对移动。保留调用点以免改 finally。"""
        output_device.set_block_aimbot_move(False)

    def _intent_delta(self, t_start: float, t_end: float) -> Tuple[int, int]:
        """人手意图增量：委托 RingBuffer 自动处理后端差异。"""
        return self.ring_buffer.get_intent_delta(t_start, t_end)

    def _log_aim_lock(self, event: str, **fields) -> None:
        """委托集中化日志系统输出锁定诊断。"""
        logger.aim_lock(event, enabled=self._aim_lock_diag, **fields)

    @staticmethod
    def _ctx_best_cls(ctx: InferenceContext) -> int:
        t = ctx.targets
        if not t:
            return -1
        try:
            return int(t[0].class_id)
        except (TypeError, ValueError, IndexError):
            return -1

    def _log_runtime_summary(self) -> None:
        if not config.getbool("Debug", "startup_diag", True):
            return
        st = self.aim_strategy
        bp = getattr(st, "bypass_mapping", None)
        cap = int(self.crop_center * 2) if self.crop_center is not None else config.getint("General", "capture_size", 256)
        mac = config.getfloat("General", "min_aim_conf", 0.32)
        mdr = min(mac - 1e-3, config.getfloat("General", "min_aim_conf_drop", 0.20))
        ninv = max(1, int(config.getint("General", "aim_drop_invalid_frames", 2)))
        stream_ms = float(getattr(self.world_model, "_stream_ingress_s", 0.0)) * 1000.0
        _be = (config.getstr("Inference", "backend", "yolo") or "yolo").strip().lower()
        _model_disp = "HSV+contour (aimlab)" if _be in _AIMLAB_BACKENDS else config.getstr("Inference", "model_path", "")
        logger.info(
            "Runtime | capture=%dpx | inf_conf>=%.2f | min_aim=%.2f (drop<%.2f) inv_n=%d | stream_ingress=%.0fms | strategy_bypass=%s | backend=%s | model=%s",
            cap,
            config.getfloat("Inference", "conf_threshold", 0.4),
            mac,
            mdr,
            ninv,
            stream_ms,
            bp,
            _be,
            _model_disp,
        )
        logger.info(
            "WorldModel | predict_ahead=%s | False 时无时间前推，动目标/高延迟会偏「拖尾」，可对照过冲/穿零是否由 lead 引起",
            config.getbool("WorldModel", "predict_ahead", True),
        )
        if stream_ms < 1.0 and config.getfloat("WorldModel", "moonlight_latency_ms", 0.0) < 0.5:
            logger.info(
                "WorldModel | stream_ingress≈0：若用 Moonlight/云游戏仍摆/穿零，把 [WorldModel] moonlight_latency_ms 调到 25–45 再试"
            )
        _hib = (config.getstr("General", "human_input_backend", "inputs") or "inputs").strip()
        logger.info(
            "HumanInput | human_input_backend=%s | intent 由 RingBuffer.get_intent_delta() 统一处理",
            _hib,
        )
        logger.info(
            "IntentTracker | 方向感知融合 + 紧急通道 | attack=%.1f/s decay=%.1f/s oppose_cos<%.2f cooperate_cos>%.2f | 见 config intent_*",
            float(getattr(self._intent_tracker, 'attack_rate', _rtd.INTENT_ATTACK_RATE)),
            float(getattr(self._intent_tracker, 'decay_rate', _rtd.INTENT_DECAY_RATE)),
            float(getattr(self._intent_tracker, 'oppose_cos', _rtd.INTENT_OPPOSE_COS_THRESHOLD)),
            float(getattr(self._intent_tracker, 'cooperate_cos', _rtd.INTENT_COOPERATE_COS_THRESHOLD)),
        )
        logger.info(
            "IntentTracker | 紧急: EMA速>%.0f px/s→全权 | 退出<%.0f px/s+hold %.2fs | persist兜底: %d帧+%.0fpx",
            float(getattr(self._intent_tracker, "emergency_speed", 1500.0)),
            float(getattr(self._intent_tracker, "emergency_exit_speed", 400.0)),
            float(getattr(self._intent_tracker, "emergency_exit_hold", 0.08)),
            int(getattr(self._intent_tracker, "emergency_persist_frames", 6)),
            float(getattr(self._intent_tracker, "emergency_persist_dist", 55.0)),
        )
        logger.info(
            "IntentTracker | 抑误触: 人速窗=%.0fms persist仅当速≥%.0fpx/s且err≤%.0fpx | EMA(紧急)τ=%.0fms",
            float(getattr(self._intent_tracker, "human_vel_window_s", 0.032)) * 1000.0,
            float(getattr(self._intent_tracker, "persist_min_speed", 280.0)),
            float(getattr(self._intent_tracker, "skip_persist_emergency_dist", 95.0)),
            float(getattr(self._intent_tracker, "emergency_spd_ema_tau", 0.045)) * 1000.0,
        )
        _safety_cap = float(getattr(self._intent_tracker, 'safety_max_ai_power', 0.0))
        if _safety_cap > 0.0:
            logger.info(
                "IntentTracker | 安全包络: AI输出上限=%.0f%% (score>%.2f时) | 防黑洞吸住",
                _safety_cap * 100.0,
                float(getattr(self._intent_tracker, 'safety_engage_score', 0.20)),
            )
        else:
            logger.info(
                "IntentTracker | 安全包络: 关闭 (intent_safety_max_ai_power=0)"
            )
        if self._aim_lock_diag:
            logger.info(
                "AimLock diag: ON | 见日志前缀 AimLock | DROP/HOLD/ACQUIRE/COAST/SKIP，定位为何重复 Target acquired"
            )
        if config.getbool("Debug", "human_clutch_diag", False):
            logger.info(
                "IntentTrack diag: ON | 每 %.2fs 一条 | [Debug] human_clutch_diag=False 关闭 |"
                " score=人手控制强度[0,1] ai_w=AI输出权重[0,1] |"
                " v_human=人手瞬时速度 | err=AI瞄准误差",
                float(config.getfloat("Debug", "human_clutch_diag_interval_sec", 0.25)),
            )
        if config.getbool("Debug", "pipeline_latency_log", False):
            logger.info(
                "PipelineLat diag: ON | 每 %.2fs 一条 | infer_ema/vh/stream/lead/hw + wall(wm/aim/cipher/tick) | "
                "cipher 行为对比 Rust/Numba 看 cipher(...) 耗时",
                float(config.getfloat("Debug", "pipeline_latency_interval_sec", 1.0)),
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
        try:
            if self.paused:
                self._freeze_mouse_motion()
                self.last_tick_time = now
                return
    
            _t_wm0 = time.perf_counter()
            self.world_model.step(self.ctx, self.ring_buffer)
            _t_after_wm = time.perf_counter()
            self._dbg_wm_step_us = (_t_after_wm - _t_wm0) * 1e6
            self._dbg_after_wm_mono = _t_after_wm
    
            # 关自瞄时仍跑 WM 推理，但绝不允许再发移动（原先只停扳机、仍会 compute → OU 白噪微颤）
            if not self.enable_aimbot:
                self._freeze_mouse_motion()
                self.last_tick_time = now
                return
    
            # coast：无新框但 Kalman 仍在 150ms 内续跑；勿当「无效」累加 streak → 否则 2 帧拆锁 → 狂刷 Target acquired
            if self.ctx.is_coasting:
                if self._aim_lock_diag:
                    t0 = time.perf_counter()
                    if t0 - getattr(self, "_aim_coast_log_t", 0) > 0.25:
                        self._aim_coast_log_t = t0
                        self._log_aim_lock(
                            "COAST",
                            lock_kept=self.target_first_seen_time > 0.0,
                            dets_ctx=len(self.ctx.targets or []),
                        )
                self._freeze_mouse_motion()
                self.last_tick_time = now
                return
    
            dt = now - self.last_tick_time
            self.last_tick_time = now
            if self._first_tick:
                self._first_tick = False
            elif dt <= 0 or dt > 0.1:
                # 异常 dt（首帧外）：关 emit
                self._log_aim_lock("SKIP", reason="bad_dt", dt_ms=round(dt * 1000.0, 2), note="lock_not_cleared")
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
                # Aimlab/HSV 色块会偶发 1~2 帧无有效框，General=2 时易拆锁 → 狂刷「Target acquired」+ reset_target_state
                if getattr(self, "_is_aimlab_backend", False):
                    _aim_min = int(_rtd.AIMLAB_AIM_DROP_INVALID_MIN)
                    if _aim_min > 0:
                        n_inv_drop = max(n_inv_drop, max(1, _aim_min))
                # 已锁时：短无效帧只 freeze 不拆锁，避免丢框一帧就 reset 控制器 → 贴脸来回摆
                if self.target_first_seen_time > 0.0 and self._invalid_streak < n_inv_drop:
                    if self._aim_lock_diag and self._invalid_streak == 1:
                        self._log_aim_lock(
                            "HOLD",
                            reason="invalid",
                            streak=self._invalid_streak,
                            need_drop_at=n_inv_drop,
                            is_valid=self.ctx.is_valid,
                            has_p=(self.ctx.p_predict is not None),
                            dets=len(self.ctx.targets or []),
                            cls=self._ctx_best_cls(self.ctx),
                        )
                    self._freeze_mouse_motion()
                    return
                self._log_aim_lock(
                    "DROP",
                    reason="invalid",
                    streak=self._invalid_streak,
                    need_streak=">=" + str(n_inv_drop),
                    is_valid=self.ctx.is_valid,
                    has_p=(self.ctx.p_predict is not None),
                    dets=len(self.ctx.targets or []),
                    cls=self._ctx_best_cls(self.ctx),
                )
                self._freeze_mouse_motion()
                self.target_first_seen_time = 0.0
                self._lock_confirm_count = 0
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
                    self._lock_confirm_count = 0
                    self._freeze_mouse_motion()
                    return
                # ── 三帧确认计数：目标需连续 N 帧有效后才锁定，防止单帧噪声/误检锁假目标 ──
                lock_need = max(1, int(config.getint("General", "aim_lock_confirm_count", 3)))
                self._lock_confirm_count += 1
                if self._lock_confirm_count < lock_need:
                    if self._aim_lock_diag:
                        self._log_aim_lock(
                            "CONFIRM",
                            count=self._lock_confirm_count,
                            need=lock_need,
                            conf=round(self.ctx.conf, 3),
                            cls=self._ctx_best_cls(self.ctx),
                        )
                    self._freeze_mouse_motion()
                    return
            else:
                if self.ctx.conf < min_drop:
                    self._log_aim_lock(
                        "DROP",
                        reason="conf",
                        conf=round(self.ctx.conf, 3),
                        min_drop=round(min_drop, 3),
                        dets=len(self.ctx.targets or []),
                        cls=self._ctx_best_cls(self.ctx),
                    )
                    self._freeze_mouse_motion()
                    self.target_first_seen_time = 0.0
                    self._lock_confirm_count = 0
                    self.is_target_in_crosshair = False
                    return
    
            c = self.controller
            # ── 首次见到目标：决定 chase_mode 并 reset controller ───────────
            first_frame = (self.target_first_seen_time == 0.0)
            if first_frame:
                self._intent_tracker.reset()
                self.target_first_seen_time = now
                # Bug B 修：按最近 100ms 人类速度判定模式
                #   实战大部分场景是"人拉枪到 256px 内 AI 接管" → human_flick
                #   目标自己走进 FOV（被动追）→ pure_ai
                dx_100, dy_100 = self._intent_delta(now - 0.1, now)
                recent_speed = math.hypot(dx_100, dy_100) / 0.1
                self._current_chase_mode = 'human_flick' if recent_speed > 500.0 else 'pure_ai'
                if hasattr(self.controller, 'reset_target_state'):
                    try:
                        self.controller.reset_target_state(mode=self._current_chase_mode)
                    except TypeError:
                        self.controller.reset_target_state()
                # ── 确认控制器 chase_mode 已生效 ──
                ctrl_chase = getattr(self.controller, 'chase_mode', None)
                if ctrl_chase is not None and ctrl_chase != self._current_chase_mode:
                    logger.error(
                        "chase_mode 切换失败！agent=%s controller=%s",
                        self._current_chase_mode, ctrl_chase,
                    )
                else:
                    logger.info(
                        "chase_mode 确认: agent=%s controller=%s (human_speed=%.0f px/s)",
                        self._current_chase_mode, ctrl_chase, recent_speed,
                    )
                self._n_target_acquire_logs += 1
                _acq_n = self._n_target_acquire_logs
                _msg = "Target acquired (mode=%s, human_speed_100ms=%.0f)" % (
                    self._current_chase_mode, recent_speed
                )
                if _acq_n == 1 or config.getbool("Debug", "aim_reacquire_log", False):
                    logger.info(_msg)
                else:
                    # 再次 ACQUIRE=中间曾拆锁，多为连续 invalid(丢框)。与 pure_ai 无关
                    logger.debug(
                        "Target re-acquired (#%d) %s", _acq_n, _msg
                    )
                self._log_aim_lock(
                    "ACQUIRE",
                    mode=self._current_chase_mode,
                    dets=len(self.ctx.targets or []),
                    conf=round(self.ctx.conf, 3),
                    cls=self._ctx_best_cls(self.ctx),
                )

            # ── chase_mode 持续更新：人放手后自动切 pure_ai ──────────────────
            # 原逻辑只在首帧判定一次。按人类速度连续调度：
            #   human_flick 模式 + 人最近 80ms 速度 < FLICK_END_PX_S → 切 pure_ai
            # warm_start 在 flick_end 侦测点统一处理，此处不再重复。
            FLICK_END_PX_S = 450.0  # 唯一阈值，与 flick_end 侦测共用
            if self._current_chase_mode == 'human_flick':
                _recent_dx, _recent_dy = self._intent_delta(now - 0.08, now)
                _recent_spd = math.hypot(_recent_dx, _recent_dy) / 0.08
                # ── 3 帧滞回：连续满足条件才切换，防止人手在 450px/s 边界微操导致震荡 ──
                if _recent_spd < FLICK_END_PX_S and (now - self.target_first_seen_time) > 0.15:
                    self._mode_switch_confirm += 1
                    if self._mode_switch_confirm >= 3:
                        _prev_mode = self._current_chase_mode
                        self._current_chase_mode = 'pure_ai'
                        self._flick_end_time = now
                        self._mode_switch_confirm = 0
                        logger.info(
                            "chase_mode 切换: %s → %s (human_speed=%.0f < %.0f px/s, age=%.0f ms)",
                            _prev_mode, self._current_chase_mode,
                            _recent_spd, FLICK_END_PX_S,
                            (now - self.target_first_seen_time) * 1000.0,
                        )
                else:
                    self._mode_switch_confirm = 0

            # p_predict 须紧跟首帧/锁逻辑之后，避免 1kHz 在本帧里多跑上帧 arm_vel/OU
            p_x = self.ctx.p_predict[0]
            p_y = self.ctx.p_predict[1]
            pixel_error_dist = math.hypot(p_x, p_y)
    
            # ── 人手速度：单帧 raw（甩枪/flick_end）+ 时间窗 smooth（意图/紧急，抑 pynput 减账尖峰）──
            dx_h_inst, dy_h_inst = self._intent_delta(now - dt, now)
            human_speed_raw = math.hypot(dx_h_inst, dy_h_inst) / dt if dt > 0 else 0.0
            _vw = float(
                np.clip(
                    max(float(self._intent_tracker.human_vel_window_s), float(dt)),
                    0.015,
                    0.120,
                )
            )
            dx_hs, dy_hs = self._intent_delta(now - _vw, now)
            human_vx = (dx_hs / _vw) if _vw > 0 else 0.0
            human_vy = (dy_hs / _vw) if _vw > 0 else 0.0
            bbox_w = self.ctx.targets[0].w if self.ctx.targets else 60.0
    
            intent_x, intent_y = self.aim_strategy.calculate_mouse_move(p_x, p_y, bbox_w=bbox_w)
    
            v_real_pixels = self.ctx.v_real
            intent_vx, intent_vy = self.aim_strategy.calculate_velocity_move(
                v_real_pixels[0], v_real_pixels[1], bbox_w=bbox_w
            )
            a_real_pixels = getattr(self.ctx, 'a_real', (0.0, 0.0))
            intent_ax, intent_ay = self.aim_strategy.calculate_velocity_move(
                a_real_pixels[0], a_real_pixels[1], bbox_w=bbox_w
            )
    
            # ── spatial_factor: 越近目标越放权，越远越收敛 ──
            spatial_factor = float(np.clip(1.0 - (pixel_error_dist / 800.0) ** 2, 0.1, 1.0))
    
            # ── reaction_factor: AI 接管后的视觉反应斜坡 ──
            if self._current_chase_mode == 'pure_ai':
                reaction_factor = 1.0
            else:
                t_ref = self._flick_end_time if self._flick_end_time > 0.0 else self.target_first_seen_time
                reaction_factor = float(np.clip((now - t_ref) / 0.03, 0.30, 1.0))
    
            # ── 人手意图 → AI 权重（唯一融合点：方向感知 + 非对称响应 + 紧急通道）──
            ai_weight = self._intent_tracker.update(
                human_vx, human_vy,
                p_x, p_y,  # AI 瞄准误差 ∈ px
                dt, now,
            )
    
            power_factor = spatial_factor * reaction_factor * ai_weight
    
            # ── Debug: 意图追踪诊断 ──
            if config.getbool("Debug", "human_clutch_diag", False):
                _div = max(0.05, config.getfloat("Debug", "human_clutch_diag_interval_sec", 0.25))
                if now - self._human_clutch_diag_last >= _div:
                    self._human_clutch_diag_last = now
                    s = self._intent_tracker
                    logger.intent_diag(
                        score=s.score, ai_weight=ai_weight,
                        human_vx=int(round(human_vx)), human_vy=int(round(human_vy)),
                        speed_ema=s._speed_ema,
                        err_x=p_x, err_y=p_y, err_dist=pixel_error_dist,
                        mode=self._current_chase_mode,
                        spatial_factor=spatial_factor,
                        reaction_factor=reaction_factor,
                        power_factor=power_factor,
                        emergency=getattr(s, '_emergency_active', False),
                    )
    
            # ── 人类甩枪结束沿检测 → 通知控制器清积分 ──────────────────────
            # 固定门限 450px/s：低于此值即判定人已放手。
            # 热启动速度 = 人速 × 0.7 + 目标速 × 0.3（人速为主，保证接管连续）
            FLICK_END_PX_S = 450.0
            cur_human_flicking = human_speed_raw > FLICK_END_PX_S
            if self._prev_human_flicking and not cur_human_flicking:
                self._flick_end_time = now  # reaction_factor 计时起点
                if hasattr(self.controller, 'notify_flick_end'):
                    self.controller.notify_flick_end()
                if hasattr(self.controller, 'warm_start_from_velocity'):
                    try:
                        # 人手残余速度（瞬时，20ms 窗口）
                        _hdx, _hdy = self._intent_delta(now - 0.02, now)
                        human_v_inst = np.array([_hdx / 0.02, _hdy / 0.02],
                                                dtype=np.float64)
                        vr = self.ctx.v_real
                        if vr is not None:
                            k = getattr(self.aim_strategy, 'k_factor_x', 1.0)
                            tgt_v = np.array([float(vr[0]) * k, float(vr[1]) * k],
                                             dtype=np.float64)
                            seed = human_v_inst * 0.7 + tgt_v * 0.3
                        else:
                            seed = human_v_inst
                        self.controller.warm_start_from_velocity(
                            float(seed[0]), float(seed[1]))
                    except Exception as e:
                        logger.debug("warm_start(flick_end) failed: %s", e)
            self._prev_human_flicking = cur_human_flicking
    
            # ── 调用控制器 ────────────────────────────────────────────────
            if hasattr(c, "set_mouse_emit"):
                c.set_mouse_emit(True)
            _t_cipher0 = time.perf_counter()
            _aim_chain_us = (_t_cipher0 - self._dbg_after_wm_mono) * 1e6
            self.controller.compute(
                target_x=intent_x, target_y=intent_y, dt=dt,
                v_real=np.array([intent_vx, intent_vy]),
                a_real=np.array([intent_ax, intent_ay]),
                power_factor=power_factor,
                bbox_w=bbox_w,
            )
            _t_cipher1 = time.perf_counter()
            _cipher_us = (_t_cipher1 - _t_cipher0) * 1e6
            _tick_wall_us = (_t_cipher1 - now) * 1e6
            if config.getbool("Debug", "pipeline_latency_log", False):
                _ival = max(0.05, config.getfloat("Debug", "pipeline_latency_interval_sec", 1.0))
                if _t_cipher1 - self._pipeline_lat_log_last >= _ival:
                    self._pipeline_lat_log_last = _t_cipher1
                    logger.pipeline_latency(
                        infer_ema_ms=float(self.world_model.inference_ms_ema),
                        vh_ms=float(self.world_model.dynamic_vh_latency * 1000.0),
                        stream_cfg_ms=float(
                            getattr(self.world_model, "_stream_ingress_s", 0.0) * 1000.0
                        ),
                        lead_ms=float(getattr(self.ctx, "dynamic_lag_ms", 0.0)),
                        base_hw_ms=float(self.world_model.base_hardware_lag * 1000.0),
                        wm_us=float(self._dbg_wm_step_us),
                        aim_us=float(_aim_chain_us),
                        cipher_us=float(_cipher_us),
                        tick_us=float(_tick_wall_us),
                        backend=CIPHER_KERNEL_BACKEND,
                        mode=str(self._current_chase_mode),
                        predict_ahead=config.getbool("WorldModel", "predict_ahead", True),
                        ctrl_lead_enable=config.getbool("WorldModel", "ctrl_lead_enable", True),
                        adaptive_latency=config.getbool(
                            "WorldModel", "adaptive_latency_enable", True
                        ),
                    )

            if config.getbool("Debug", "move_emit_diag", False):
                ival = config.getfloat("Debug", "move_emit_log_interval_sec", 0.2)
                t_m = time.perf_counter()
                if t_m - self._last_move_emit_log >= ival:
                    self._last_move_emit_log = t_m
                    d = self.controller.get_move_emit_diag()
                    if d is not None:
                        eix, eiy = d["prev_emit_int"]
                        efx, efy = d["prev_emit_flt"]
                        cvx, cvy = d["cmd_vel_ct_s"]
                        dts = d["dt_s"]
                        ex = cvx * dts
                        ey = cvy * dts
                        logger.info(
                            "MoveEmit | 上周期(两次compute间)实发 ct_int=(%d,%d) ct_flt=(%.3f,%.3f) | "
                            "本帧末 cmd_vel=(%.1f,%.1f)ct/s dt=%.4fs 粗 v*dt=(%.2f,%.2f) [v*dt 为快照粗算；上周期实发由 1kHz+腕/OU+取整 积分]",
                            eix, eiy, efx, efy, cvx, cvy, dts, ex, ey,
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
                    human_override=ai_weight,
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
        finally:
            self._sync_aimbot_move_block()


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
        dx_sum, dy_sum = self._intent_delta(now - 0.2, now)
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
            logger.tick_stats(real_fps, inf_ema, self.ctx.is_valid, n_dets,
                            conf, lag, self._current_chase_mode, self.paused,
                            self.enable_aimbot)
        else:
            logger.tick_stats_short(real_fps, lag, conf, self._current_chase_mode)
        self.last_stat_time = now
        self.frames_in_cycle = 0

    def toggle_pause(self):
        self.paused = not self.paused
        logger.warning("PAUSED: %s", self.paused)

    def toggle_aimbot(self):
        self.enable_aimbot = not self.enable_aimbot
        if not self.enable_aimbot:
            self._freeze_mouse_motion()
            self._sync_aimbot_move_block()
        logger.warning("AIMBOT: %s", self.enable_aimbot)
