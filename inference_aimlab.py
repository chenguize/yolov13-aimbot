# inference_aimlab.py
# Aim Lab 等「纯色小球」目标：不加载 YOLO，BGR 画面做 HSV 阈值 + 最大连通域，输出与 YOLO 同形的 (N,6) 检测数组。

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from perception.bus import FrameBus
from utils import runtime_defaults as D
from utils.logger import get_logger
from world_model import WorldModel

logger = get_logger("InferenceAimlab")


def _build_hsv_mask(
    hsv: np.ndarray,
    color_mode: str,
    hsv_lo: np.ndarray,
    hsv_hi: np.ndarray,
    red_h1_max: int,
    red_h2_min: int,
) -> np.ndarray:
    """
    range：单段 H in [h_min, h_max]（OpenCV H 为 0–180）。
    red_wrap：红色在 Hue 上跨 0°/180°，需两段 inRange 再 OR，否则 0~180 全含虽能拍到红
    但会混入大量非红像素；Aim Lab 默认红球应用 red_wrap。
    """
    mode = (color_mode or "range").strip().lower()
    s0, s1 = int(hsv_lo[1]), int(hsv_hi[1])
    v0, v1 = int(hsv_lo[2]), int(hsv_hi[2])
    h0, h1 = int(hsv_lo[0]), int(hsv_hi[0])

    if mode in ("red", "red_wrap", "red_aimlab", "default_red"):
        r1 = min(180, max(0, red_h1_max))
        r2 = min(180, max(0, red_h2_min))
        m1 = cv2.inRange(
            hsv, np.array([0, s0, v0], dtype=np.uint8), np.array([r1, s1, v1], dtype=np.uint8)
        )
        m2 = cv2.inRange(
            hsv, np.array([r2, s0, v0], dtype=np.uint8), np.array([180, s1, v1], dtype=np.uint8)
        )
        return cv2.bitwise_or(m1, m2)
    return cv2.inRange(hsv, hsv_lo, hsv_hi)


def detect_ball_bgr(
    bgr: np.ndarray,
    hsv_lo: np.ndarray,
    hsv_hi: np.ndarray,
    min_area: int,
    morph_ksize: int,
    *,
    color_mode: str = "range",
    red_h1_max: int = 15,
    red_h2_min: int = 165,
    ignore_center_margin_px: float = 14.0,
    mask_out_center_radius: int = 0,
    reject_if_only_center_blobs: bool = True,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    返回 (x1,y1,x2,y2,conf,cls) 单框 6 元组，无有效目标时 None。
    conf 在 [0.35, 0.99] 之间，保证通过 WorldModel 的 conf<0.3 过滤。
    """
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    if h < 4 or w < 4:
        return None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = _build_hsv_mask(hsv, color_mode, hsv_lo, hsv_hi, red_h1_max, red_h2_min)

    mor = int(mask_out_center_radius) if mask_out_center_radius else 0
    if mor > 0:
        cv2.circle(mask, (w // 2, h // 2), min(mor, min(w, h) // 2 - 1), 0, -1)

    if morph_ksize and morph_ksize >= 3:
        k = morph_ksize | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cx0, cy0 = w * 0.5, h * 0.5
    significant = [c for c in contours if cv2.contourArea(c) >= float(min_area)]
    if not significant:
        return None

    def _candidate(ctr):
        area = float(cv2.contourArea(ctr))
        perimeter = float(cv2.arcLength(ctr, True))
        x, y, bw, bh = cv2.boundingRect(ctr)
        hull_area = float(cv2.contourArea(cv2.convexHull(ctr)))
        circularity = (
            4.0 * np.pi * area / (perimeter * perimeter)
            if perimeter > 1e-6 else 0.0
        )
        solidity = area / hull_area if hull_area > 1e-6 else 0.0

        # Fill the component before distance transform. This removes HSV holes
        # from highlights while the maximum inscribed circle ignores thin
        # same-colour crosshair arms attached to the ball.
        component = np.zeros((bh + 2, bw + 2), dtype=np.uint8)
        shifted = ctr.astype(np.int32).copy()
        shifted[:, 0, 0] += 1 - x
        shifted[:, 0, 1] += 1 - y
        cv2.drawContours(component, [shifted], -1, 255, thickness=-1)
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        _, radius, _, max_loc = cv2.minMaxLoc(distance)
        center_x = float(x + max_loc[0] - 1)
        center_y = float(y + max_loc[1] - 1)
        near_center = np.hypot(center_x - cx0, center_y - cy0) < float(
            ignore_center_margin_px
        )

        min_core = max(2.0, np.sqrt(float(min_area) / np.pi) * 0.75)
        shape_like_ball = circularity >= 0.32 and solidity >= 0.62
        strong_round_core = radius >= max(4.0, min_core * 1.5)
        ball_like = radius >= min_core and (shape_like_ball or strong_round_core)
        score = radius * 12.0 + np.sqrt(max(area, 0.0)) + 4.0 * circularity
        return {
            "contour": ctr,
            "area": area,
            "center": (center_x, center_y),
            "radius": float(radius),
            "near_center": bool(near_center),
            "ball_like": bool(ball_like),
            "score": float(score),
        }

    candidates = [_candidate(ctr) for ctr in significant]
    if reject_if_only_center_blobs:
        # Keep the old permissive behaviour away from the crosshair. At the
        # center, require a round core instead of rejecting every observation.
        candidates = [
            c for c in candidates if not c["near_center"] or c["ball_like"]
        ]
    if not candidates:
        return None

    best = max(candidates, key=lambda c: c["score"])
    area = best["area"]
    center_x, center_y = best["center"]
    radius = max(best["radius"], 1.0)
    x1 = float(np.clip(center_x - radius, 0.0, w - 1.0))
    y1 = float(np.clip(center_y - radius, 0.0, h - 1.0))
    x2 = float(np.clip(center_x + radius, x1 + 1.0, float(w)))
    y2 = float(np.clip(center_y + radius, y1 + 1.0, float(h)))
    # 面积占画面比例高 → 置信略高；小球通常只占一小部分
    rel = (area / float(h * w)) if (h * w) > 0 else 0.0
    conf = float(np.clip(0.4 + 0.55 * min(1.0, rel * 50.0), 0.35, 0.99))
    return (x1, y1, x2, y2, conf, 0.0)


class AimlabBallInferenceThread(threading.Thread):
    """
    与 InferenceThread 相同接口，消费 FrameBus，写 world_model.update_detections。
    不依赖 TensorRT/Ultralytics。

    使用前：[Inference] backend=aimlab|opencv|…；HSV/色域 见 utils/runtime_defaults.AIMLAB_*。
    [WorldModel] aim_target_classes 留空或含 runtime_defaults.AIMLAB_SYNTHETIC_CLASS_ID。
    """

    def __init__(
        self,
        bus: FrameBus,
        world_model: WorldModel,
        shutdown_event: threading.Event,
        frame_ready_event: threading.Event,
    ):
        super().__init__(name="AimlabBallInference", daemon=True)
        self.bus = bus
        self.world_model = world_model
        self.shutdown_event = shutdown_event
        self.frame_ready_event = frame_ready_event
        self.last_processed_id = -1
        self._first_push_logged = False
        self._hsv_log_once = False

    def _read_hsv_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        h0 = int(D.AIMLAB_H_MIN)
        h1 = int(D.AIMLAB_H_MAX)
        s0 = int(D.AIMLAB_S_MIN)
        s1 = int(D.AIMLAB_S_MAX)
        v0 = int(D.AIMLAB_V_MIN)
        v1 = int(D.AIMLAB_V_MAX)
        h0, h1 = max(0, h0), min(180, h1)
        s0, s1 = max(0, s0), min(255, s1)
        v0, v1 = max(0, v0), min(255, v1)
        if h0 > h1:
            h0, h1 = h1, h0
        if s0 > s1:
            s0, s1 = s1, s0
        if v0 > v1:
            v0, v1 = v1, v0
        lo = np.array([h0, s0, v0], dtype=np.uint8)
        hi = np.array([h1, s1, v1], dtype=np.uint8)
        return lo, hi

    def run(self):
        min_area = max(1, int(D.AIMLAB_MIN_AREA))
        morph_ksize = int(D.AIMLAB_MORPH_KSIZE)
        synthetic_class_id = int(D.AIMLAB_SYNTHETIC_CLASS_ID)
        hsv_lo, hsv_hi = self._read_hsv_bounds()
        color_mode = (D.AIMLAB_COLOR_MODE or "red_wrap").strip()
        red_h1_max = int(D.AIMLAB_RED_H1_MAX)
        red_h2_min = int(D.AIMLAB_RED_H2_MIN)
        ign_margin = float(D.AIMLAB_IGNORE_CENTER_MARGIN_PX)
        mask_ctr_r = int(D.AIMLAB_MASK_OUT_CENTER_RADIUS)
        rej_center = bool(D.AIMLAB_REJECT_ONLY_CENTER_BLOBS)
        out_conf_mode = float(D.AIMLAB_OUTPUT_CONF)

        if not self._hsv_log_once:
            self._hsv_log_once = True
            if color_mode.lower() in ("red", "red_wrap", "red_aimlab", "default_red"):
                logger.info(
                    "Aimlab 红球(双段H) | color_mode=%s | H[0..%d] U H[%d..180] | S/V lo=%s hi=%s | min_area=%d | class=%d",
                    color_mode,
                    red_h1_max,
                    red_h2_min,
                    hsv_lo.tolist(),
                    hsv_hi.tolist(),
                    min_area,
                    synthetic_class_id,
                )
            else:
                logger.info(
                    "Aimlab 小球 | color_mode=%s | HSV lo=%s hi=%s | min_area=%d morph=%d | class=%d",
                    color_mode,
                    hsv_lo.tolist(),
                    hsv_hi.tolist(),
                    min_area,
                    morph_ksize,
                    synthetic_class_id,
                )
            logger.info(
                "准星与球同色时：ignore_center_margin=%.0fpx | mask挖中心 r=%d | reject_only_center=%s",
                ign_margin,
                mask_ctr_r,
                rej_center,
            )
            logger.info(
                "须 Inference.backend=aimlab|opencv|…；仍无目标可改 runtime_defaults 中 AIMLAB_MIN_AREA 或 S/V 下限。"
            )
            logger.info(
                "若 aim_target_classes 非空，须含 class=%d 或留空。",
                synthetic_class_id,
            )
            if out_conf_mode > 0.0:
                logger.info("output_conf=%.2f 覆盖面积 conf；默认 0=面积公式(推荐)", out_conf_mode)
            else:
                logger.info("output_conf=0 使用每帧面积 conf（快动球勿开 ghost/强平滑，易滞后过冲）")

        last_print_time = 0.0
        while not self.shutdown_event.is_set():
            if not self.frame_ready_event.wait(timeout=0.005):
                continue
            self.frame_ready_event.clear()

            frame_info = self.bus.get_latest()
            if frame_info is None or frame_info.frame_id <= self.last_processed_id:
                continue

            try:
                t0 = time.perf_counter()
                frame = frame_info.frame
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)

                row = detect_ball_bgr(
                    frame,
                    hsv_lo,
                    hsv_hi,
                    min_area,
                    morph_ksize,
                    color_mode=color_mode,
                    red_h1_max=red_h1_max,
                    red_h2_min=red_h2_min,
                    ignore_center_margin_px=ign_margin,
                    mask_out_center_radius=mask_ctr_r,
                    reject_if_only_center_blobs=rej_center,
                )
                if row is not None:
                    x1, y1, x2, y2, c_raw, _ = row
                    if out_conf_mode > 0.0:
                        c_use = float(np.clip(out_conf_mode, 0.35, 0.99))
                    else:
                        c_use = c_raw
                    detections = np.array(
                        [[x1, y1, x2, y2, c_use, float(synthetic_class_id)]],
                        dtype=np.float32,
                    )
                else:
                    detections = np.empty((0, 6), dtype=np.float32)

                t1 = time.perf_counter()
                last_ms = (t1 - t0) * 1000.0
                now = t1
                if now - last_print_time > 1.0:
                    if len(detections) > 0:
                        logger.info(
                            "Aimlab | 1 目标 | conf=%.2f | infer=%.2fms",
                            float(detections[0, 4]),
                            last_ms,
                        )
                    else:
                        logger.debug("Aimlab | 无有效掩膜 | infer=%.2fms", last_ms)
                    last_print_time = now

                self.world_model.update_detections(
                    detections=detections,
                    frame_id=frame_info.frame_id,
                    t_capture=frame_info.t_cap,
                    t_done=t1,
                )
                if not self._first_push_logged:
                    self._first_push_logged = True
                    logger.info(
                        "Pipeline: first Aimlab → world_model (frame_id=%d, dets=%d, infer=%.2fms).",
                        frame_info.frame_id,
                        len(detections),
                        last_ms,
                    )
                self.last_processed_id = frame_info.frame_id

            except Exception as e:
                logger.error("Aimlab inference loop error: %s", e)
