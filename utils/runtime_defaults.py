# utils/runtime_defaults.py
# Aimlab 色球、意图追踪等门限。

# ── 人手意图追踪（HumanIntentTracker）──
# 人手速度低于此值(px/s)视为「没在动」，意图衰减
INTENT_SPEED_DEADZONE_PX_S = 4.0
# 对抗时人手意图上升速率 (/s)：约 0.2s 内意图从 0→1
INTENT_ATTACK_RATE = 5.0
# 静止时意图衰减速率 (/s)：约 0.7s 内意图从 1→0
INTENT_DECAY_RATE = 1.5
# 方向内积阈值：dot < 此值判定为「对抗」（人手反方向移动）
INTENT_OPPOSE_COS_THRESHOLD = -0.15
# 方向内积阈值：dot > 此值判定为「配合」（人手同方向移动）
INTENT_COOPERATE_COS_THRESHOLD = 0.35
# 紧急通道速度门槛(px/s)：超过+对抗=瞬间全权。典型flick 2k-15k，跟踪 50-300
INTENT_EMERGENCY_SPEED_PX_S = 1500.0
# 紧急通道退出速度(px/s)
INTENT_EMERGENCY_EXIT_SPEED_PX_S = 400.0
# 紧急通道退出保持(s)：满足退出速度后再保持这么久；过长会手已停仍 ai_w=0
INTENT_EMERGENCY_EXIT_HOLD_S = 0.080
# 持久化兜底：连续对抗帧数
INTENT_EMERGENCY_PERSIST_FRAMES = 6
# 持久化兜底：累积对抗位移门槛(px)
INTENT_EMERGENCY_PERSIST_DIST_PX = 55.0
# agent 人手速度估计最短窗(s)：抑 pynput/SendInput 减账在单帧 dt 上的尖峰
INTENT_HUMAN_VEL_WINDOW_S = 0.032
# 紧急速度判定 EMA 时间常数(s)：瞬时尖峰不易误触 emergency_speed
INTENT_EMERGENCY_SPD_EMA_TAU_S = 0.045
# 低于此人速不计入 persist 兜底（微抖、减账残差不当「强对抗」）
INTENT_PERSIST_MIN_SPEED_PX_S = 280.0
# 准星误差大于此：仅允许「高速甩枪」进紧急，禁止 persist 兜底（大误差时常误判对抗）
INTENT_SKIP_PERSIST_EMERGENCY_DIST_PX = 95.0
# 安全包络：非紧急时人手对抗+score>此值 → AI功率上限压到此比例(0=关闭)
INTENT_SAFETY_MAX_AI_POWER = 0.30
INTENT_SAFETY_ENGAGE_SCORE = 0.20

# ── Aimlab 色球推理（[Inference] backend=aimlab|opencv|hsv|… 时；不再使用 [Aimlab] 段）─
# color_mode=red_wrap：Aim Lab 红球；range 时用 H_MIN/H_MAX 单段
AIMLAB_COLOR_MODE = "red_wrap"
AIMLAB_RED_H1_MAX = 15
AIMLAB_RED_H2_MIN = 165
AIMLAB_H_MIN = 0
AIMLAB_H_MAX = 180
AIMLAB_S_MIN = 30
AIMLAB_S_MAX = 255
AIMLAB_V_MIN = 30
AIMLAB_V_MAX = 255
AIMLAB_MIN_AREA = 20
AIMLAB_MORPH_KSIZE = 3
AIMLAB_IGNORE_CENTER_MARGIN_PX = 16.0
AIMLAB_MASK_OUT_CENTER_RADIUS = 0
AIMLAB_REJECT_ONLY_CENTER_BLOBS = True
AIMLAB_SYNTHETIC_CLASS_ID = 0
# <=0 用面积公式 conf；>0 写死 conf
AIMLAB_OUTPUT_CONF = 0.0
# 与 General.aim_drop_invalid 取 max；0=不额外加
AIMLAB_AIM_DROP_INVALID_MIN = 3
