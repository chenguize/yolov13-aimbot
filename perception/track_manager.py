# perception/track_manager.py
# ═══════════════════════════════════════════════════════════════════════════════
# Hungarian-IoU 多目标追踪管理器  │  per-track Kalman 池
# ═══════════════════════════════════════════════════════════════════════════════
#
# 核心改进（替代原 PRIMARY_TRACK 单槽）：
#   1. 每个物理目标分配独立 track_id，享受独立 Kalman
#   2. IoU 贪心级联匹配 → 检测框与历史轨迹稳定关联
#   3. 未匹配轨迹 Kalman predict-only coast，未匹配检测 spawn 新轨迹
#   4. 从池中按策略选出「最佳瞄准目标」→ 对外接口与旧逻辑完全兼容
#
# 匹配算法：贪心级联（SORT 风格），O(T·D·log(T·D))，无额外依赖。

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import config
from utils.logger import get_logger

_log = get_logger("TrackManager")

# 复用 world_model 中的 TargetState / SimpleKalman（避免循环导入）
# 实际在 world_model 初始化时注入


def _compute_iou(box_a: Tuple[float, float, float, float],
                 box_b: Tuple[float, float, float, float]) -> float:
    """IoU = |A ∩ B| / |A ∪ B|。输入均为 (x1, y1, x2, y2)。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    if denom < 1e-9:
        return 0.0
    return inter / denom


def cv2_bhatt_correlation(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """
    HSV 颜色直方图相似度。返回 [0, 1]，1 表示完全相同。
    优先用 cv2.compareHist（HSV 直方图标准方法）；不可用时退化为余弦相似度。
    """
    try:
        import cv2
        # cv2.compareHist 返回 [0, 1]，1=完全匹配（CV_COMP_CORREL）
        return float(cv2.compareHist(hist_a.reshape(-1, 1).astype(np.float32),
                                      hist_b.reshape(-1, 1).astype(np.float32),
                                      cv2.HISTCMP_CORREL))
    except Exception:
        # 退化：余弦相似度
        a = hist_a.flatten().astype(np.float64)
        b = hist_b.flatten().astype(np.float64)
        na = float(np.linalg.norm(a) + 1e-9)
        nb = float(np.linalg.norm(b) + 1e-9)
        return float(np.dot(a, b) / (na * nb))


@dataclass
class Track:
    """单一物理目标的 Kalman 轨迹。"""

    track_id: int
    target: "TargetState"          # Kalman 6-state (pos/vel/acc)
    kalman: "SimpleKalman"         # 独立自适应状态 (innov_ema / Q)
    last_bbox: Tuple[float, float, float, float]  # 最近匹配框 (x1,y1,x2,y2)
    class_id: int
    conf_ema: float                # 置信度 EMA
    created: float                 # time.perf_counter()
    last_matched: float            # 最后匹配时间
    coast_count: int               # 连续未匹配帧数
    total_matches: int             # 历史匹配总数
    last_innovation: Optional[np.ndarray] = None  # 最近一次 Kalman innovation
    # ── 重构新增：ReID 颜色直方图（HSV 8x8x8 = 512 bin，归一化）──
    # 用于短时遮挡场景下的关联，IoU 失效时仍能匹配同一目标
    color_hist: Optional[np.ndarray] = None
    # ── 重构新增：上次被选为 best 的时间，用于加权评分 ──
    last_selected: float = 0.0
    # ── Fire-gate 死亡降权：death_penalty_until > now 时 select_best 排除 ──
    death_penalty_until: float = 0.0
    seed_position: Optional[np.ndarray] = None
    seed_time: float = 0.0
    velocity_seeded: bool = False

    @property
    def is_active(self) -> bool:
        return self.coast_count == 0

    @property
    def abs_position(self) -> np.ndarray:
        return self.target.state[:2]

    @property
    def abs_velocity(self) -> np.ndarray:
        return self.target.state[2:4]

    @property
    def abs_accel(self) -> np.ndarray:
        return self.target.state[4:6]


class TrackManager:
    """
    多目标 Kalman 池 + IoU 匹配 + 目标选择。

    对外接口：
      match_and_update()  每帧调用，传入检测列表
      select_best()       从活跃轨迹中选出最佳瞄准目标
      predict_all()       无检测帧时纯预测所有轨迹
    """

    def __init__(self):
        self.tracks: Dict[int, Track] = {}
        self._next_id: int = 0

        # ── 匹配参数 ──
        self.iou_threshold: float = config.getfloat(
            "WorldModel", "track_iou_threshold", 0.25
        )
        self.max_coast: int = config.getint(
            "WorldModel", "track_max_coast", 8
        )
        self.max_tracks: int = config.getint(
            "WorldModel", "track_max_tracks", 12
        )
        # 快动小球 IoU 常低于 track_iou_threshold，仍用框心距作弱关联(0=关闭)
        self.center_match_max_px: float = max(
            0.0, config.getfloat("WorldModel", "track_center_match_max_px", 48.0)
        )
        # 每帧仅 1 个检测时：与「预测一步」位置最近的轨迹若在门内则强制关联，避免快球断轨→狂刷
        # track_id（Aimlab 单球最明显）。0 关闭门控（仍可用 hard_assign）。
        self._single_det_gate_px: float = max(
            0.0, config.getfloat("WorldModel", "track_single_det_gate_px", 90.0)
        )
        # True：仅 1 检测时**永远**并入预测最近轨，不为该检测 spawn（Aimlab 单球必开；
        # 多目标场景可能出现两敌人叠在一起只检出一个时再设 False）。
        self._single_det_hard_assign: bool = config.getbool(
            "WorldModel", "track_single_det_hard_assign", True
        )
        # 单检测时：若上一帧 select_best 的轨预测落在测量此半径内，优先并入该轨（防幽灵轨抢最近邻）
        self._single_det_sticky_px: float = max(
            0.0, config.getfloat("WorldModel", "track_single_det_sticky_px", 150.0)
        )
        # select_best(closest) 时：上一帧 best 与几何最近比不必更近这么多(px) 则保持，减 p_predict 抖
        self._select_sticky_px: float = max(
            0.0, config.getfloat("WorldModel", "track_select_sticky_px", 22.0)
        )

        # ── 选择策略 ──
        self.selection_policy: str = config.getstr(
            "WorldModel", "track_selection_policy", "closest_to_crosshair"
        ).strip().lower()

        # ── 上一次选中的轨迹 ID（用于检测目标切换）──
        self._last_best_id: Optional[int] = None

        # ── 重构新增：ReID 颜色直方图配置 ──
        # IoU + 框心距不足时，加颜色直方图相关性维度（HSV 8x8x8=512 bin）
        self._reid_enable: bool = config.getbool(
            "WorldModel", "track_reid_enable", True
        )
        self._reid_weight: float = float(np.clip(config.getfloat(
            "WorldModel", "track_reid_weight", 0.15
        ), 0.0, 0.5))
        # ── 重构新增：速度先验预热权重 ──
        # 新目标 spawn 时从同帧活跃 track 速度均值借用多少（0=禁用，1=完全借用）
        self._velocity_prior_enable: bool = config.getbool(
            "WorldModel", "track_velocity_prior_enable", True
        )
        self._velocity_prior_alpha: float = float(np.clip(config.getfloat(
            "WorldModel", "track_velocity_prior_alpha", 0.6
        ), 0.0, 1.0))
        # ── 重构新增：加权评分选择策略权重 ──
        # policy=weighted_score 时启用，综合 距离/conf/threat/velocity
        self._ws_w_dist: float = config.getfloat("WorldModel", "track_ws_w_dist", 1.0)
        self._ws_w_conf: float = config.getfloat("WorldModel", "track_ws_w_conf", 0.5)
        self._ws_w_threat: float = config.getfloat("WorldModel", "track_ws_w_threat", 0.3)
        self._ws_w_velocity: float = config.getfloat("WorldModel", "track_ws_w_velocity", 0.2)

    def mark_track_death_penalty(self, track_id: int, duration_s: float, now: float) -> None:
        """标记某轨迹为「疑似死亡」，在 duration_s 内 select_best 排除该轨迹。

        由 agent._confirm_fire_gate_death 调用：开枪后几何确认死亡 → 降权排除尸体。
        """
        if track_id in self.tracks:
            self.tracks[track_id].death_penalty_until = now + duration_s

    def _prune_stale_tracks(self) -> None:
        """coast_count 超过 max_coast 的轨迹移除。单球 hard_assign 时池中仅 1 条则不删——
        否则 long coast 后 select_best 将其剔出 active → 空池 spawn → track_id 狂刷。"""
        n = len(self.tracks)
        for _tid in list(self.tracks.keys()):
            t = self.tracks[_tid]
            if t.coast_count <= self.max_coast:
                continue
            if self._single_det_hard_assign and n == 1:
                continue
            cc = t.coast_count
            del self.tracks[_tid]
            _log.track_pruned(_tid, cc)

    # ═══════════════════════════════════════════════════════════════════════
    # § 1 │ 匹配与更新（主入口）
    # ═══════════════════════════════════════════════════════════════════════

    def match_and_update(
        self,
        detections: List[dict],       # [{abs_meas, bbox_xyxy, conf, class_id}, ...]
        now: float,
        dt: float,
    ) -> None:
        """
        一帧内完成：匹配 → 更新轨迹 Kalman → 创建新轨迹 → 清理过期轨迹。
        detections 格式: {abs_meas: ndarray(2), bbox: (x1,y1,x2,y2), conf, class_id}
        """
        # ── 0. 先删「失配过久」轨迹。旧逻辑要求 total_matches>honeymoon 才删，导致
        # spawn 后若再也配不上会 coast→∞，池子里全是僵尸轨，单目标最近邻也会刷 ID。
        self._prune_stale_tracks()

        active_tracks = list(self.tracks.values())

        # ── 1. 匹配：单检测硬指派 / 门控 / IoU 贪心 ─────────────────────────
        gate = float(self._single_det_gate_px)
        hard1 = bool(self._single_det_hard_assign)
        if len(detections) == 1 and len(active_tracks) >= 1:
            det0 = detections[0]
            meas = np.asarray(det0["abs_meas"], dtype=np.float64).reshape(2)
            best_ti = -1
            best_d = 1e18
            stick_px = float(self._single_det_sticky_px)
            last_id = self._last_best_id
            if stick_px > 0.0 and last_id is not None:
                for ti, trk in enumerate(active_tracks):
                    if int(trk.track_id) != int(last_id):
                        continue
                    vx, vy = float(trk.abs_velocity[0]), float(trk.abs_velocity[1])
                    px, py = float(trk.abs_position[0]), float(trk.abs_position[1])
                    pred = np.array([px + vx * dt, py + vy * dt], dtype=np.float64)
                    d_last = float(np.linalg.norm(pred - meas))
                    if d_last <= stick_px:
                        best_ti = ti
                        best_d = d_last
                    break
            if best_ti < 0:
                for ti, trk in enumerate(active_tracks):
                    vx, vy = float(trk.abs_velocity[0]), float(trk.abs_velocity[1])
                    px, py = float(trk.abs_position[0]), float(trk.abs_position[1])
                    pred = np.array([px + vx * dt, py + vy * dt], dtype=np.float64)
                    d = float(np.linalg.norm(pred - meas))
                    if d < best_d:
                        best_d = d
                        best_ti = ti
            if best_ti < 0:
                matches, unmatched_track_idxs, unmatched_det_idxs = self._greedy_match(
                    active_tracks, detections, dt
                )
            elif hard1 or (gate > 0.0 and best_d <= gate):
                matches = [(best_ti, 0)]
                unmatched_track_idxs = [i for i in range(len(active_tracks)) if i != best_ti]
                unmatched_det_idxs = []
            else:
                matches, unmatched_track_idxs, unmatched_det_idxs = self._greedy_match(
                    active_tracks, detections, dt
                )
        else:
            matches, unmatched_track_idxs, unmatched_det_idxs = self._greedy_match(
                active_tracks, detections, dt
            )

        # ── 2. 更新已匹配轨迹 ──
        for ti, dj in matches:
            track = active_tracks[ti]
            det = detections[dj]
            track.kalman.predict(track.target, dt)
            innovation = track.kalman.update(track.target, det["abs_meas"])
            if not track.velocity_seeded:
                seed_dt = now - track.seed_time
                if track.seed_position is not None and seed_dt >= 0.020:
                    measured_vel = (
                        np.asarray(det["abs_meas"], dtype=np.float64)
                        - track.seed_position
                    ) / seed_dt
                    speed = float(np.linalg.norm(measured_vel))
                    max_seed_speed = 3000.0
                    if speed > max_seed_speed and speed > 1e-9:
                        measured_vel *= max_seed_speed / speed
                    # Blend instead of replacing: retain Kalman's covariance-
                    # weighted estimate while removing its zero-velocity bias.
                    track.target.state[2:4] = (
                        0.20 * track.target.state[2:4]
                        + 0.80 * measured_vel
                    )
                    track.target.state[4:6] = 0.0
                    track.velocity_seeded = True
            track.last_innovation = innovation
            track.last_bbox = det["bbox"]
            track.class_id = det["class_id"]
            track.last_matched = now
            track.coast_count = 0
            track.total_matches += 1
            # conf EMA 平滑
            alpha = 0.3 if track.total_matches < 5 else 0.15
            track.conf_ema = (1.0 - alpha) * track.conf_ema + alpha * det["conf"]
            # 重构新增：刷新颜色直方图（EMA 平滑，避免单帧抖动）
            if self._reid_enable:
                new_hist = det.get("color_hist", None)
                if new_hist is None:
                    frame = det.get("frame", None)
                    bbox = det.get("bbox", None)
                    if frame is not None and bbox is not None:
                        new_hist = self._compute_color_hist(frame, bbox)
                if new_hist is not None:
                    if track.color_hist is None:
                        track.color_hist = new_hist
                    else:
                        # EMA：hist = 0.85*old + 0.15*new
                        track.color_hist = (
                            0.85 * track.color_hist + 0.15 * new_hist
                        ).astype(np.float32)

        # ── 3. 未匹配轨迹：纯预测（coast）──
        for ti in unmatched_track_idxs:
            track = active_tracks[ti]
            track.kalman.predict(track.target, dt)
            track.coast_count += 1

        # ── 4. 未匹配检测：spawn 新轨迹 ──
        for dj in unmatched_det_idxs:
            if len(self.tracks) >= self.max_tracks:
                break
            det = detections[dj]
            self._spawn_track(det, now)

        # ── 5. 单检测 + hard_assign：只保留本帧命中的轨，避免多轨同时 coast_count==0
        #     时在 select_best(离准星最近) 下来回换 track_id。
        if hard1 and len(detections) == 1 and matches:
            ti_kept, _dj = matches[0]
            kept_id = active_tracks[ti_kept].track_id
            for tid in list(self.tracks.keys()):
                if tid != kept_id:
                    del self.tracks[tid]

    def predict_all(self, dt: float):
        """无检测帧：所有轨迹纯预测。"""
        self._prune_stale_tracks()
        for track in self.tracks.values():
            track.kalman.predict(track.target, dt)
            track.coast_count += 1

    # ═══════════════════════════════════════════════════════════════════════
    # § 2 │ 目标选择策略
    # ═══════════════════════════════════════════════════════════════════════

    def select_best(
        self,
        ego_pos_px: np.ndarray,
        crop_center: float,
    ) -> Optional[Track]:
        """
        从活跃轨迹中按策略选出最佳瞄准目标。
        策略由 [WorldModel] track_selection_policy 控制。

        closest_to_crosshair : 选离准星最近的轨迹（直觉：瞄准你正在看的人）
        highest_conf         : 选置信度最高的轨迹
        newest               : 选最近匹配的（最少切换）
        largest              : 选最大边界框（最近威胁）

        ── 活轨迹优先 ──
        当存在 coast_count==0（本帧刚匹配）的轨迹时，跳过 coast 中的幽灵轨迹。
        否则 ghost 的 Kalman 预测位置因准星在追它而始终"最近"，形成自洽循环：
        球死了 → 追 ghost → ghost 离准星最近 → select ghost → 继续追 ghost。
        只有在所有轨迹都 coast 时（如全部敌人躲掩体后）才回退到全量选择。
        """
        now = time.perf_counter()
        if self._single_det_hard_assign and len(self.tracks) == 1:
            active = list(self.tracks.values())
        else:
            active = [
                t for t in self.tracks.values()
                if t.coast_count <= self.max_coast
            ]
        if not active:
            return None

        # ── 活轨迹优先：有本帧匹配的 track 时排除 ghost ──
        matched = [t for t in active if t.coast_count == 0]
        candidates = matched if matched else active

        # ── Fire-gate 死亡降权：排除被标记死亡的轨迹（全部被标记时保留全部）──
        if candidates:
            unpenalized = [t for t in candidates if t.death_penalty_until <= now]
            if unpenalized:
                candidates = unpenalized

        policy = self.selection_policy

        best: Optional[Track] = None

        if policy in ("highest_conf", "conf", "confidence"):
            # conf_ema + 轻微的存活时间加成（优先稳定跟踪的目标）
            best = max(candidates, key=lambda t: t.conf_ema + 0.001 * min(t.total_matches, 50))
        elif policy in ("newest", "recent", "latest"):
            best = max(candidates, key=lambda t: t.last_matched)
        elif policy in ("largest", "biggest", "bbox"):
            best = max(candidates, key=lambda t: (
                (t.last_bbox[2] - t.last_bbox[0]) * (t.last_bbox[3] - t.last_bbox[1])
            ))
        elif policy in ("weighted_score", "weighted", "score"):
            # 重构新增：综合评分 = w_dist·距离反比 + w_conf·conf + w_threat·最近可见 + w_vel·速度反比
            best = self._select_weighted_score(candidates, ego_pos_px, now)
        else:
            # default: closest_to_crosshair
            best = min(candidates, key=lambda t: float(
                np.linalg.norm(t.abs_position - ego_pos_px)
            ))
            spx = float(self._select_sticky_px)
            if spx > 0.0 and self._last_best_id is not None:
                prev_t = next(
                    (t for t in candidates if t.track_id == self._last_best_id),
                    None,
                )
                if prev_t is not None:
                    d_prev = float(np.linalg.norm(prev_t.abs_position - ego_pos_px))
                    d_best = float(np.linalg.norm(best.abs_position - ego_pos_px))
                    if d_prev <= d_best + spx:
                        best = prev_t

        # ── 检测目标切换：best track 的 ID 与上一帧不同时记录 ──
        if best.track_id != self._last_best_id:
            _prev = self._last_best_id
            _prev_info = ""
            if _prev is not None and _prev in self.tracks:
                pt = self.tracks[_prev]
                _prev_info = " (prev_conf=%.2f, prev_coast=%d)" % (pt.conf_ema, pt.coast_count)
            _log.track_switch(
                _prev, best.track_id, policy,
                best.conf_ema, best.coast_count, best.total_matches,
                prev_info=_prev_info,
            )
            self._last_best_id = best.track_id

        # 重构新增：标记 last_selected 用于加权评分
        best.last_selected = now
        return best

    def _select_weighted_score(
        self, candidates: List[Track], ego_pos_px: np.ndarray, now: float
    ) -> Track:
        """
        综合加权评分选择最佳目标。
        score = w_dist * (1 / (1 + d_px))           ← 距离越近分越高
              + w_conf * conf_ema                    ← 置信度越高分越高
              + w_threat * (1 / (1 + t_since_seen))  ← 最近可见威胁分高
              + w_velocity * (1 / (1 + v_rel))      ← 相对速度越小分越高（易命中）
        并加 sticky 加成：上一帧 best 在 22px 内不切换。
        """
        def _score(t: Track) -> float:
            d = float(np.linalg.norm(t.abs_position - ego_pos_px))
            t_seen = max(0.0, now - t.last_matched)
            v_mag = float(np.linalg.norm(t.abs_velocity))
            # 各维度归一化到 [0, 1]
            s_dist = 1.0 / (1.0 + d * 0.02)              # d=0→1, d=50→0.5, d=200→0.2
            s_conf = float(np.clip(t.conf_ema, 0.0, 1.0))
            s_threat = 1.0 / (1.0 + t_seen * 4.0)         # 0ms→1, 250ms→0.5, 1s→0.2
            s_vel = 1.0 / (1.0 + v_mag * 0.001)           # 0px/s→1, 1000px/s→0.5
            return (
                self._ws_w_dist * s_dist
                + self._ws_w_conf * s_conf
                + self._ws_w_threat * s_threat
                + self._ws_w_velocity * s_vel
            )
        best = max(candidates, key=_score)
        # sticky：上一帧 best 比几何最近多 sticky_px 也不切
        spx = float(self._select_sticky_px)
        if spx > 0.0 and self._last_best_id is not None:
            prev_t = next(
                (t for t in candidates if t.track_id == self._last_best_id),
                None,
            )
            if prev_t is not None and prev_t.track_id != best.track_id:
                if _score(prev_t) >= _score(best) - 0.05:  # 5% 容差
                    best = prev_t
        return best

    # ═══════════════════════════════════════════════════════════════════════
    # § 3 │ 内部：匹配 / 衍生
    # ═══════════════════════════════════════════════════════════════════════

    def _pair_affinity(self, trk: Track, det: dict, dt: float) -> float:
        """
        IoU 优先；不足阈值时用框心距弱匹配（Aimlab 快动球 IoU 常断裂）。
        重构新增：ReID 颜色直方图相关性作为附加维度，处理短时遮挡。
        """
        iou = _compute_iou(trk.last_bbox, det["bbox"])
        if iou > self.iou_threshold:
            # IoU 高时仍加 ReID 微调（防止同色目标错配）
            reid_bonus = 0.0
            if self._reid_enable and self._reid_weight > 0.0:
                reid_bonus = self._reid_weight * self._reid_correlation(trk, det)
            return 1.0 + float(iou) + reid_bonus

        cmax = float(self.center_match_max_px)
        if cmax <= 0.0:
            return 0.0
        b, d = trk.last_bbox, det["bbox"]
        vx, vy = float(trk.abs_velocity[0]), float(trk.abs_velocity[1])
        cx1 = 0.5 * (b[0] + b[2]) + vx * dt
        cy1 = 0.5 * (b[1] + b[3]) + vy * dt
        cx2 = 0.5 * (d[0] + d[2])
        cy2 = 0.5 * (d[1] + d[3])
        dist = math.hypot(cx1 - cx2, cy1 - cy2)
        if dist >= cmax:
            return 0.0

        # 框心距弱匹配：基础分 0.5 + 距离衰减
        base_score = 0.5 + 0.5 * (1.0 - dist / cmax)
        # ReID 加成：颜色直方图相关性 [0, 1]
        if self._reid_enable and self._reid_weight > 0.0:
            reid_corr = self._reid_correlation(trk, det)
            base_score += self._reid_weight * reid_corr
        return base_score

    def _reid_correlation(self, trk: Track, det: dict) -> float:
        """
        HSV 颜色直方图相关性。返回 [0, 1]。
        det 必须含 "frame" 字段（BGR 图像），否则跳过。
        """
        if trk.color_hist is None:
            return 0.0
        det_hist = det.get("color_hist", None)
        if det_hist is None:
            # 没有预计算，尝试现场计算（如果 frame 可用）
            frame = det.get("frame", None)
            bbox = det.get("bbox", None)
            if frame is None or bbox is None:
                return 0.0
            det_hist = self._compute_color_hist(frame, bbox)
            if det_hist is None:
                return 0.0
            det["color_hist"] = det_hist  # 缓存
        # Bhattacharyya 距离 → 相似度 [0, 1]
        try:
            bc = float(cv2_bhatt_correlation(trk.color_hist, det_hist))
            return max(0.0, min(1.0, bc))
        except Exception:
            return 0.0

    def _compute_color_hist(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """
        从 frame 的 bbox 区域提取 HSV 8x8x8 颜色直方图，归一化。
        frame 必须是 BGR (H, W, 3)。失败返回 None。
        """
        if frame is None or frame.size == 0:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                return None
            roi = frame[y1:y2, x1:x2]
            # BGR → HSV
            try:
                import cv2
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                                    [0, 180, 0, 256, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                return hist.flatten().astype(np.float32)
            except ImportError:
                return None
        except Exception:
            return None

    def _greedy_match(
        self,
        tracks: List[Track],
        detections: List[dict],
        dt: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        贪心级联：IoU 与框心弱匹配取 affinity 降序。

        Returns:
          matches              : [(track_idx, det_idx), ...]
          unmatched_track_idxs : 未匹配的 track 索引列表
          unmatched_det_idxs   : 未匹配的 detection 索引列表
        """
        n_t = len(tracks)
        n_d = len(detections)
        if n_t == 0:
            return [], [], list(range(n_d))
        if n_d == 0:
            return [], list(range(n_t)), []

        pairs: List[Tuple[float, int, int]] = []
        for i, trk in enumerate(tracks):
            for j, det in enumerate(detections):
                a = self._pair_affinity(trk, det, dt)
                if a > 0.0:
                    pairs.append((a, i, j))
        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_t = set()
        matched_d = set()
        matches: List[Tuple[int, int]] = []

        for _, ti, dj in pairs:
            if ti not in matched_t and dj not in matched_d:
                matches.append((ti, dj))
                matched_t.add(ti)
                matched_d.add(dj)

        unmatched_t = [i for i in range(n_t) if i not in matched_t]
        unmatched_d = [j for j in range(n_d) if j not in matched_d]

        return matches, unmatched_t, unmatched_d

    def _spawn_track(self, det: dict, now: float) -> Track:
        """
        从检测创建新轨迹。
        重构新增：
          1. 速度先验预热 — 用同帧活跃 track 速度均值作初值，收敛时间 80ms→20ms
          2. 颜色直方图提取 — 用于后续 ReID 关联
          3. IMM-Kalman 可选 — 配置 Kalman.use_imm=True 时启用，处理动目标变向
        """
        from world_model import TargetState

        tid = self._next_id
        self._next_id += 1

        # 选择 Kalman 实现
        try:
            from config import config as _cfg
            use_imm = _cfg.getbool("Kalman", "use_imm", False)
        except Exception:
            use_imm = False

        if use_imm:
            try:
                from perception.imm_kalman import IMMKalman
                kalman = IMMKalman()
            except Exception as e:
                from world_model import SimpleKalman
                kalman = SimpleKalman()
                _log.warning("IMMKalman unavailable, fallback SimpleKalman: %s", e)
        else:
            from world_model import SimpleKalman
            kalman = SimpleKalman()
        target = TargetState(
            id=tid,
            first_seen=now,
            last_seen=now,
            confidence=det["conf"],
        )
        target.state[:2] = det["abs_meas"].copy()

        # ── 速度先验预热 ──────────────────────────────────────────────────
        # 旧版：速度初值置零，2 帧后 Kalman 自行收敛到合理速度估计（~80ms）。
        # 新版：从同帧活跃 track 速度加权均值借用（按距离衰减权重），收敛 80ms→20ms。
        # 适用 peek 场景：新目标出现时就有 60% 速度估计，首帧命中显著提升。
        if self._velocity_prior_enable and self.tracks:
            try:
                meas = np.asarray(det["abs_meas"], dtype=np.float64).reshape(2)
                # 仅用本帧已匹配的活跃 track（coast_count==0），按距离反比加权
                active_for_prior = [
                    t for t in self.tracks.values() if t.coast_count == 0
                ]
                if active_for_prior:
                    weights = []
                    velocities = []
                    for t in active_for_prior:
                        d = float(np.linalg.norm(t.abs_position - meas))
                        w = 1.0 / (1.0 + d * 0.01)  # 距离越近权重越大
                        weights.append(w)
                        velocities.append(t.abs_velocity.copy())
                    w_sum = float(sum(weights))
                    if w_sum > 1e-6:
                        v_prior = np.average(
                            np.array(velocities), weights=np.array(weights), axis=0
                        )
                        # 60% 先验 + 40% 零（留收敛空间）
                        alpha_prior = float(self._velocity_prior_alpha)
                        target.state[2:4] = v_prior * alpha_prior
                        # 加速度初值也借用部分
                        accs = [t.abs_accel.copy() for t in active_for_prior]
                        a_prior = np.average(
                            np.array(accs), weights=np.array(weights), axis=0
                        )
                        target.state[4:6] = a_prior * (alpha_prior * 0.5)
            except Exception as e:
                _log.warning("velocity_prior failed: %s", e)

        # ── 颜色直方图提取 ──────────────────────────────────────────────
        color_hist = None
        if self._reid_enable:
            frame = det.get("frame", None)
            bbox = det.get("bbox", None)
            if frame is not None and bbox is not None:
                color_hist = self._compute_color_hist(frame, bbox)
                if color_hist is not None:
                    det["color_hist"] = color_hist  # 缓存到 det 供后续匹配复用

        track = Track(
            track_id=tid,
            target=target,
            kalman=kalman,
            last_bbox=det["bbox"],
            class_id=det["class_id"],
            conf_ema=det["conf"],
            created=now,
            last_matched=now,
            coast_count=0,
            total_matches=1,
            color_hist=color_hist,
            seed_position=np.asarray(det["abs_meas"], dtype=np.float64).copy(),
            seed_time=now,
        )
        self.tracks[tid] = track
        _log.track_spawned(tid, det["class_id"], det["conf"])
        return track
