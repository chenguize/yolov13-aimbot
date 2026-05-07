# utils/logger.py
"""
集中化日志管理。

使用方式：
    from utils.logger import get_logger
    logger = get_logger("Agent")
    logger.tick_stats(...)    # 统一的 tick 统计日志
    logger.aim_lock(...)      # 统一的瞄准锁/拆锁诊断
    logger.intent_diag(...)   # 统一的意图追踪诊断

所有模块共用此入口，便于全局调整日志格式、级别、输出目标。
"""
import logging
from typing import Optional


class AimbotLogger(logging.LoggerAdapter):
    """
    扩展 logging.LoggerAdapter，在标准 logger 基础上增加项目特定的日志方法。
    所有日志条目的格式由 logging_bootstrap.setup_root_logging() 统一控制。
    """

    def tick_stats(self, fps: float, infer_ema: float, is_valid: bool,
                   n_dets: int, conf: float, lag_ms: float,
                   mode: str, paused: bool, aim_on: bool):
        self.info(
            "Tick | fps=%.1f | infer_ema=%.1fms | valid=%s | dets=%d | "
            "conf=%.2f | lat=%.1fms | mode=%s | paused=%s | aim_on=%s",
            fps, infer_ema, is_valid, n_dets, conf, lag_ms,
            mode, paused, aim_on,
        )

    def tick_stats_short(self, fps: float, lag_ms: float, conf: float, mode: str):
        self.info("FPS: %.1f | Lat: %.1fms | Conf: %.2f | Mode: %s",
                  fps, lag_ms, conf, mode)

    def aim_lock(self, event: str, enabled: bool = True, **fields):
        """锁定/拆锁诊断（受 Debug.aim_lock_diag 控制）。"""
        if not enabled:
            return
        extra = " | ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        if extra:
            self.info("AimLock | %s | %s", event, extra)
        else:
            self.info("AimLock | %s", event)

    def intent_diag(self, score: float, ai_weight: float,
                    human_vx: int, human_vy: int, speed_ema: float,
                    err_x: float, err_y: float, err_dist: float,
                    mode: str, spatial_factor: float,
                    reaction_factor: float, power_factor: float,
                    emergency: bool = False):
        em_tag = " [EMERGENCY]" if emergency else ""
        self.info(
            "IntentTrack%s | score=%.3f ai_w=%.3f | "
            "v_human=(%d,%d) spd_ema=%.0f px/s | "
            "err=(%.1f,%.1f) dist=%.1f | "
            "mode=%s sf=%.2f rf=%.2f pf=%.3f",
            em_tag,
            score, ai_weight,
            human_vx, human_vy, speed_ema,
            err_x, err_y, err_dist,
            mode, spatial_factor, reaction_factor, power_factor,
        )

    def controller_mode(self, old_mode: str, new_mode: str,
                        dist: float, threshold: float, unit: str = "px"):
        self.debug("mode 切换: %s → %s (dist=%.1f %s, thresh=%.1f %s)",
                   old_mode, new_mode, dist, unit, threshold, unit)

    def phase_switch(self, from_phase: str, to_phase: str, **fields):
        extra = " | ".join(f"{k}={v}" for k, v in fields.items())
        self.debug("phase 切换: %s → %s%s", from_phase, to_phase,
                   (" | " + extra) if extra else "")

    # ── TrackManager ──────────────────────────────────────────────────────

    def track_switch(self, from_id, to_id: int, policy: str, conf: float,
                     coast: int, matches: int, prev_info: str = ""):
        self.info(
            "track 切换: %s → track_%d | policy=%s | conf=%.2f coast=%d matches=%d%s",
            ("None" if from_id is None else f"track_{from_id}"),
            to_id, policy, conf, coast, matches, prev_info,
        )

    def track_spawned(self, track_id: int, cls: int, conf: float):
        self.debug("Track %d spawned (cls=%d, conf=%.2f)", track_id, cls, conf)

    def track_pruned(self, track_id: int, coast: int):
        self.debug("Track %d pruned (coast=%d)", track_id, coast)

    # ── WorldModel: Kalman / 延迟自适应 / 校准 ────────────────────────────

    def kalman_adaptive_latency(self, latency_ms: float, wan_mode: bool = False):
        tag = " [WAN]" if wan_mode else ""
        self.info("自适应 vh 延迟: %.1f ms%s", latency_ms, tag)

    def kalman_wan_jitter_safe(self, smoothed_ms: float, floor_ms: float):
        self.info("  WAN jitter-safe: %.1f ms | floor=%.1f ms", smoothed_ms, floor_ms)

    def ctrl_speed_scale_diag(self, scale: float, arm_spd: float, tgt_spd: float):
        self.info("ctrl_speed_scale: %.3f | arm=%.0f tgt=%.0f ct/s",
                  scale, arm_spd, tgt_spd)

    def arrival_calib_diag(self, arrival_ratio: float, expected_spd: float,
                           ai_spd: float, speed_scale: float):
        self.info(
            "arrival_calib: arrival=%.2f | expected=%.0f ai=%.0f ct/s | speed_scale→%.3f",
            arrival_ratio, expected_spd, ai_spd, speed_scale,
        )

    # ── Inference ─────────────────────────────────────────────────────────

    def inference_result(self, n_dets: int, best_cls: int, best_conf: float,
                         infer_ms: float):
        self.info("%d target(s) | best: cls=%d conf=%.2f | infer=%.1fms",
                  n_dets, best_cls, best_conf, infer_ms)

    def inference_no_target(self, infer_ms: float):
        self.debug("No targets in FOV | infer=%.1fms", infer_ms)

    def inference_first_push(self, frame_id: int, n_dets: int, infer_ms: float):
        self.info(
            "Pipeline: first YOLO → world_model.update_detections "
            "(frame_id=%d, dets=%d, infer=%.1fms). Main.tick should wake next.",
            frame_id, n_dets, infer_ms,
        )

    def pipeline_latency(
        self,
        *,
        infer_ema_ms: float,
        vh_ms: float,
        stream_cfg_ms: float,
        lead_ms: float,
        base_hw_ms: float,
        wm_us: float,
        aim_us: float,
        cipher_us: float,
        tick_us: float,
        backend: str,
        mode: str,
        predict_ahead: bool,
        ctrl_lead_enable: bool,
        adaptive_latency: bool,
    ):
        """
        主循环单帧：异步推理 EMA、模型侧延迟/前视、WorldModel.step / 瞄准链路 / CIPHER compute  wall 耗时。
        cipher_us 在 Rust/Numba 切换时最易对比；鼠标 1kHz 积分在独立线程，不含在此 tick 内。
        """
        self.info(
            "PipelineLat | model: infer_ema=%.1fms vh=%.1fms stream*=%.1fms lead=%.1fms hw=%.1fms | "
            "pred=%s ctrl_lead=%s adapt_vh=%s | "
            "wall: wm=%.0fµs aim=%.0fµs cipher(%s)=%.0fµs tick=%.0fµs | mode=%s",
            infer_ema_ms,
            vh_ms,
            stream_cfg_ms,
            lead_ms,
            base_hw_ms,
            bool(predict_ahead),
            bool(ctrl_lead_enable),
            bool(adaptive_latency),
            wm_us,
            aim_us,
            backend,
            cipher_us,
            tick_us,
            mode,
        )

    # ── 引擎 / 基础设施 ──────────────────────────────────────────────────

    def engine_status(self, msg: str, level: str = "info"):
        getattr(self, level)(msg)

    def recorder_event(self, msg: str, level: str = "info"):
        getattr(self, level)(msg)


# ── 缓存已创建的 AimbotLogger 实例 ──
_loggers: dict = {}


def get_logger(name: str) -> AimbotLogger:
    """获取或创建指定名称的 AimbotLogger。"""
    if name not in _loggers:
        raw = logging.getLogger(name)
        _loggers[name] = AimbotLogger(raw, {})
    return _loggers[name]
