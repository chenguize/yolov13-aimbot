# utils/human_intent.py
"""
人手意图追踪器 —— 方向感知融合 + 紧急通道 + 安全包络。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三层控制架构

① 正常层（α 连续变化）
   - 人手朝 AI 反方向 → 快速提升 score（attack_rate）
   - 人手朝 AI 同方向 → 缓慢衰减（配合辅助）
   - 人手不动 → 衰减交还 AI

② 紧急通道（α → 1 瞬间）
   - 触发：人手速度 > emergency_speed 且方向对抗
   - 兜底：连续 N 帧朝反方向推 + 累积位移 > persist_dist
   - 退出：速度降到 exit_speed 以下 + 额外保持 exit_hold 秒防抖

③ 安全包络（限制 AI 最大输出）
   - 非紧急但人手在对抗时，AI 功率上限压到 safety_max_ai_power
   - 防止"黑洞吸住"——即使紧急未触发，AI 也不会全力拽回去

单变量 intent_score ∈ [0, 1]：
  - 0 = AI 全控（人手没在动）
  - 1 = 人手全控（AI 完全不插手）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import time


class HumanIntentTracker:
    """
    人手意图追踪器。

    每帧调用 update(human_vx, human_vy, ai_err_x, ai_err_y, dt, now)，
    返回 ai_weight ∈ [0, 1]，直接用作 controller.power_factor 乘数。
    """

    def __init__(self):
        from config import config
        from utils import runtime_defaults as _rtd

        self.score = 0.0         # [0, 1] 人手控制强度
        self._speed_ema = 0.0    # 平滑人速 (仅用于诊断日志)

        # ── 正常层参数 ──
        self.speed_deadzone = config.getfloat(
            "General", "intent_speed_deadzone_px_s",
            getattr(_rtd, "INTENT_SPEED_DEADZONE_PX_S", 4.0),
        )
        self.attack_rate = config.getfloat(
            "General", "intent_attack_rate",
            getattr(_rtd, "INTENT_ATTACK_RATE", 5.0),
        )
        self.decay_rate = config.getfloat(
            "General", "intent_decay_rate",
            getattr(_rtd, "INTENT_DECAY_RATE", 1.5),
        )
        self.oppose_cos = config.getfloat(
            "General", "intent_oppose_cos_threshold",
            getattr(_rtd, "INTENT_OPPOSE_COS_THRESHOLD", -0.15),
        )
        self.cooperate_cos = config.getfloat(
            "General", "intent_cooperate_cos_threshold",
            getattr(_rtd, "INTENT_COOPERATE_COS_THRESHOLD", 0.35),
        )

        # ── 紧急通道参数 ──
        # 速度门槛：人手瞬时速度超过此值且方向对抗 → 瞬间全权
        self.emergency_speed = config.getfloat(
            "General", "intent_emergency_speed_px_s",
            getattr(_rtd, "INTENT_EMERGENCY_SPEED_PX_S", 1500.0),
        )
        # ── jerk² 快速触发（v5.0 升级，捕捉运动起步）─────────────────────────
        # 检测加速度的导数（jerk²）来识别"刚启动"的人类输入。
        # 速度触发有 EMA 平滑延迟（tau=45ms），jerk² 在 5-10ms 内就能识别急动量。
        # 仅作为辅助通道，不替代速度触发（持续对抗仍走速度路径）。
        self.jerk_emergency_enabled = config.getbool(
            "General", "intent_jerk_emergency_enabled",
            getattr(_rtd, "INTENT_JERK_EMERGENCY_ENABLED", True),
        )
        # jerk 阈值（px/s³）：人手 flick 起步时 acc 上升 → jerk 峰值 ~1-2e7
        # 噪声 + 正常扫视的 jerk² 一般 < 5e6，所以 1e7 是个安全门槛
        self.jerk_threshold = config.getfloat(
            "General", "intent_jerk_threshold_px_s3",
            getattr(_rtd, "INTENT_JERK_THRESHOLD_PX_S3", 10_000_000.0),
        )
        # 退出速度：人手降到以下才允许退出紧急
        self.emergency_exit_speed = config.getfloat(
            "General", "intent_emergency_exit_speed_px_s",
            getattr(_rtd, "INTENT_EMERGENCY_EXIT_SPEED_PX_S", 400.0),
        )
        # 退出保持：满足退出速度后额外保持全权时间（防抖）
        self.emergency_exit_hold = config.getfloat(
            "General", "intent_emergency_exit_hold_s",
            getattr(_rtd, "INTENT_EMERGENCY_EXIT_HOLD_S", 0.200),
        )
        # 持久化兜底：连续对抗帧数
        self.emergency_persist_frames = config.getint(
            "General", "intent_emergency_persist_frames",
            getattr(_rtd, "INTENT_EMERGENCY_PERSIST_FRAMES", 4),
        )
        # 持久化兜底：累积对抗位移门槛(px)
        self.emergency_persist_dist = config.getfloat(
            "General", "intent_emergency_persist_dist_px",
            getattr(_rtd, "INTENT_EMERGENCY_PERSIST_DIST_PX", 40.0),
        )
        self.human_vel_window_s = config.getfloat(
            "General", "intent_human_vel_window_sec",
            getattr(_rtd, "INTENT_HUMAN_VEL_WINDOW_S", 0.032),
        )
        self.emergency_spd_ema_tau = config.getfloat(
            "General", "intent_emergency_spd_ema_tau_sec",
            getattr(_rtd, "INTENT_EMERGENCY_SPD_EMA_TAU_S", 0.045),
        )
        self.persist_min_speed = config.getfloat(
            "General", "intent_persist_min_speed_px_s",
            getattr(_rtd, "INTENT_PERSIST_MIN_SPEED_PX_S", 280.0),
        )
        self.skip_persist_emergency_dist = config.getfloat(
            "General", "intent_skip_persist_emergency_dist_px",
            getattr(_rtd, "INTENT_SKIP_PERSIST_EMERGENCY_DIST_PX", 95.0),
        )

        # ── 安全包络参数 ──
        # 非紧急时，若人手对抗且 score > 此值，AI 功率上限压到此比例
        # 0 = 关闭安全包络（向后兼容）
        self.safety_max_ai_power = config.getfloat(
            "General", "intent_safety_max_ai_power",
            getattr(_rtd, "INTENT_SAFETY_MAX_AI_POWER", 0.0),
        )
        self.safety_engage_score = config.getfloat(
            "General", "intent_safety_engage_score",
            getattr(_rtd, "INTENT_SAFETY_ENGAGE_SCORE", 0.20),
        )

        # ── 紧急通道状态 ──
        self._emergency_active: bool = False
        self._emergency_exit_until: float = 0.0  # perf_counter，此前不能退出
        # ── 持久化兜底状态 ──
        self._persist_frames: int = 0
        self._persist_dist: float = 0.0
        # 紧急入口判定用人速 EMA（与诊断 _speed_ema 分离：后者 30ms 偏快）
        self._emergency_spd_ema: float = 0.0
        # ── jerk² 快速触发状态（v5.0 升级）────────────────────────────
        # 跟踪上一帧的速度和加速度，用 (v_curr - v_prev)/dt = acc, (acc - acc_prev)/dt = jerk
        self._last_human_vx: float = 0.0
        self._last_human_vy: float = 0.0
        self._last_acc_x: float = 0.0
        self._last_acc_y: float = 0.0
        self._jerk_init: bool = False  # 首帧保护：acc/jerk 需先有初值再算

    def _enter_emergency(self, now: float) -> None:
        """进入紧急通道。"""
        if self._emergency_active:
            return
        self._emergency_active = True
        self._emergency_exit_until = now + self.emergency_exit_hold
        self.score = 1.0  # 瞬间全权

    def _try_exit_emergency(self, now: float, human_speed: float) -> bool:
        """
        尝试退出紧急通道。
        条件：速度 < exit_speed 且 已过 exit_hold 时间。
        """
        if not self._emergency_active:
            return True
        if human_speed < self.emergency_exit_speed and now >= self._emergency_exit_until:
            self._emergency_active = False
            self._persist_frames = 0
            self._persist_dist = 0.0
            # 不立即重置 score：正常层衰减会自然把它降回 0
            return True
        return False

    def update(
        self,
        human_vx: float, human_vy: float,
        ai_err_x: float, ai_err_y: float,
        dt: float,
        now: float = 0.0,
    ) -> float:
        """
        更新意图状态，返回 AI 应输出的功率权重 ai_weight ∈ [0, 1]。

        Args:
            human_vx, human_vy: 人手瞬时速度 (px/s)，来自 RingBuffer 意图增量
            ai_err_x, ai_err_y:  AI 瞄准误差向量 p_predict (px)
            dt:                 距上次更新的时间间隔 (秒)
            now:                perf_counter() 时间戳，供紧急通道滞回

        Returns:
            ai_weight: 应传递给 controller.compute(power_factor=...) 的 AI 功率系数
                       0=AI 完全不输出，1=AI 全功率
        """
        if now <= 0.0:
            now = time.perf_counter()

        human_speed = math.hypot(human_vx, human_vy)

        # 平滑人速 (30ms EMA，仅用于诊断日志)
        alpha_spd = min(1.0, dt / 0.030)
        self._speed_ema = (1.0 - alpha_spd) * self._speed_ema + alpha_spd * human_speed

        tau_e = max(1e-4, float(self.emergency_spd_ema_tau))
        a_e = min(1.0, dt / tau_e)
        self._emergency_spd_ema = (1.0 - a_e) * self._emergency_spd_ema + a_e * human_speed
        spd_emg = self._emergency_spd_ema

        # ═══════════════════════════════════════════════════════════════════
        # 紧急通道：已经在紧急模式 → 只检查退出条件
        # ═══════════════════════════════════════════════════════════════════
        if self._emergency_active:
            if self._try_exit_emergency(now, human_speed):
                # 已退出，继续往下走正常逻辑
                pass
            else:
                # 仍在紧急：保持 score=1.0，AI 零输出
                self.score = 1.0
                return 0.0

        ai_dist = math.hypot(ai_err_x, ai_err_y)

        # ── 情况 1：无目标 或 人手没在动 → 衰减意图，交还 AI ──
        if ai_dist < 1.0 or human_speed < self.speed_deadzone:
            self._persist_frames = 0
            self._persist_dist = 0.0
            self.score = max(0.0, self.score - self.decay_rate * dt)
            return 1.0 - self.score

        # ── 方向判定 ──
        ai_dir_x = -ai_err_x / ai_dist
        ai_dir_y = -ai_err_y / ai_dist
        dot = (human_vx * ai_dir_x + human_vy * ai_dir_y) / human_speed

        # ═══════════════════════════════════════════════════════════════════
        # 紧急通道入口检测（在正常融合之前）
        # ═══════════════════════════════════════════════════════════════════
        is_opposing = dot < self.oppose_cos

        if is_opposing:
            # ── jerk² 快速触发（v5.0 升级，运动起步识别）─────────────────────
            # 在速度 EMA 之前先看 jerk：人手 flick 起步时 acc 急升 → jerk 峰值
            # 在 ~5-10ms 内就能识别，比 45ms EMA 速度快 20-30ms 触发紧急
            if (self.jerk_emergency_enabled
                    and self._jerk_init
                    and dt > 1e-5
                    and ai_dist <= self.skip_persist_emergency_dist):
                # acc = (v_curr - v_prev) / dt
                acc_x = (human_vx - self._last_human_vx) / dt
                acc_y = (human_vy - self._last_human_vy) / dt
                # jerk = (acc - acc_prev) / dt
                jerk_x = (acc_x - self._last_acc_x) / dt
                jerk_y = (acc_y - self._last_acc_y) / dt
                jerk_mag = math.hypot(jerk_x, jerk_y)
                if jerk_mag > self.jerk_threshold:
                    # jerk 方向应该和 human_v 一致（acc 在上升）→ 真正急动量
                    jerk_dir_x = jerk_x / jerk_mag if jerk_mag > 1e-6 else 0.0
                    jerk_dir_y = jerk_y / jerk_mag if jerk_mag > 1e-6 else 0.0
                    human_dir_x = human_vx / human_speed
                    human_dir_y = human_vy / human_speed
                    jerk_aligned = (jerk_dir_x * human_dir_x + jerk_dir_y * human_dir_y) > 0.5
                    if jerk_aligned:
                        self._enter_emergency(now)
                        # 更新状态后返回（重要：避免被速度逻辑再次覆盖）
                        self._last_human_vx = human_vx
                        self._last_human_vy = human_vy
                        self._last_acc_x = acc_x
                        self._last_acc_y = acc_y
                        return 0.0
                # 更新 acc 状态供下帧计算 jerk
                self._last_acc_x = acc_x
                self._last_acc_y = acc_y
            elif not self._jerk_init:
                # 首帧：初始化 acc 为当前 v/dt
                self._jerk_init = True
                self._last_acc_x = human_vx / max(dt, 1e-3)
                self._last_acc_y = human_vy / max(dt, 1e-3)

            # 速度触发：EMA 人速够快 + 方向对抗 → 紧急（抑单帧尖峰）
            if spd_emg > self.emergency_speed:
                self._enter_emergency(now)
                self._last_human_vx = human_vx
                self._last_human_vy = human_vy
                return 0.0

            # 持久化兜底：仅在小误差 + 明显人手对抗时启用；大误差时「对抗」多为扫视/减账残差
            if ai_dist > self.skip_persist_emergency_dist:
                self._persist_frames = 0
                self._persist_dist = 0.0
            elif spd_emg >= self.persist_min_speed:
                self._persist_frames += 1
                self._persist_dist += spd_emg * dt
                if (self._persist_frames >= self.emergency_persist_frames
                        and self._persist_dist > self.emergency_persist_dist):
                    self._enter_emergency(now)
                    self._last_human_vx = human_vx
                    self._last_human_vy = human_vy
                    return 0.0
            else:
                self._persist_frames = 0
                self._persist_dist = 0.0
        else:
            # 非对抗方向 → 重置持久化计数器
            self._persist_frames = 0
            self._persist_dist = 0.0

        # ── 更新 jerk 状态（无论是否触发，都为下帧准备）──
        self._last_human_vx = human_vx
        self._last_human_vy = human_vy

        # ═══════════════════════════════════════════════════════════════════
        # 正常层：连续 α 融合
        # ═══════════════════════════════════════════════════════════════════
        if dot < self.oppose_cos:
            strength = min(1.0, abs(dot))
            self.score = min(1.0, self.score + self.attack_rate * strength * dt)
        elif dot > self.cooperate_cos:
            self.score = max(0.0, self.score - self.decay_rate * 0.2 * dt)
        else:
            self.score = max(0.0, self.score - self.decay_rate * 0.45 * dt)

        ai_weight = 1.0 - self.score

        # ═══════════════════════════════════════════════════════════════════
        # 安全包络：非紧急时，人手在对抗 → 限制 AI 最大输出
        # ═══════════════════════════════════════════════════════════════════
        if (self.safety_max_ai_power > 0.0
                and is_opposing
                and self.score > self.safety_engage_score):
            ai_weight = min(ai_weight, self.safety_max_ai_power)

        return ai_weight

    def reset(self):
        """新目标出现时调用，清空意图累积和紧急状态。"""
        self.score = 0.0
        self._speed_ema = 0.0
        self._emergency_spd_ema = 0.0
        self._emergency_active = False
        self._emergency_exit_until = 0.0
        self._persist_frames = 0
        self._persist_dist = 0.0
        # jerk² 状态重置
        self._last_human_vx = 0.0
        self._last_human_vy = 0.0
        self._last_acc_x = 0.0
        self._last_acc_y = 0.0
        self._jerk_init = False
