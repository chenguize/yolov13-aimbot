# 控制算法详解

> 本文档覆盖 YOLOv13 Aimbot 项目中所有控制相关算法，从像素映射到最终鼠标输出，逐层拆解。

---

## 目录

1. [系统总览与数据流](#1-系统总览与数据流)
2. [AimStrategy — 像素↔鼠标 Count 映射](#2-aimstrategy--像素鼠标-count-映射)
3. [WorldModel — 状态估计与时空回溯](#3-worldmodel--状态估计与时空回溯)
4. [Kalman Filter — 6 状态恒加速模型](#4-kalman-filter--6-状态恒加速模型)
5. [TrackManager — 多目标匈牙利 IoU 匹配](#5-trackmanager--多目标匈牙利-iou-匹配)
6. [CIPHER v1.0 控制器](#6-cipher-v10-控制器)
7. [Agent 编排层](#7-agent-编排层)
8. [附录：所有 config.ini 参数速查](#8-附录所有-configini-参数速查)

---

## 1. 系统总览与数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              MAIN LOOP (~200-500Hz)                      │
│                                                                          │
│  CaptureThread          InferenceThread         WorldModel.step()        │
│  ┌──────────┐           ┌──────────┐           ┌───────────────────┐    │
│  │ dxcam    │──frame──►│ YOLO /   │──dets──►│ 1. 时空回溯        │    │
│  │ @240fps  │           │ TensorRT │           │ 2. 检测→绝对坐标   │    │
│  └──────────┘           └──────────┘           │ 3. TrackManager    │    │
│                                                │ 4. Kalman 预测     │    │
│  RawInput Listener                             │ 5. predict_ahead   │    │
│  ┌──────────────┐                              └────────┬──────────┘    │
│  │ 人手位移      │──RingBuffer──►                        │               │
│  └──────────────┘                              ┌────────▼──────────┐    │
│                                                │ Agent.tick()      │    │
│  MouseWorker (1000Hz)                          │ · chase_mode      │    │
│  ┌──────────────┐                              │ · power_factor    │    │
│  │ tick_mouse() │◄──controller◄────────────────│ · controller.     │    │
│  │ SendInput    │                              │   compute()       │    │
│  └──────────────┘                              └───────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

**线程拓扑**：

| 线程 | 频率 | 职责 |
|------|------|------|
| CaptureThread | ~240fps | dxcam 截图 → FrameBus |
| InferenceThread | ~240fps | YOLO/TensorRT 推理 → WorldModel |
| MouseWorker | 1000Hz | `controller.tick_mouse()` → `SendInput` |
| RawInputListener | 阻塞 | 监听人手物理位移 → RingBuffer |
| TriggerWorker | 按需 | 异步射击（不阻塞主循环） |
| Main loop | ~200-500Hz | `world_model.step()` + `controller.compute()` |

---

## 2. AimStrategy — 像素↔鼠标 Count 映射

**文件**: `aim_strategies/valorant/strategy.py`

### 2.1 核心问题

游戏内准星移动 ≠ 屏幕像素移动。Valorant 使用 UE5 透视投影，边缘像素对应更大的角度变化。因此：

```
counts → 游戏角度 → 屏幕像素（非线性）
```

### 2.2 数学原理

**FOV 透视投影**：

$$f = \frac{W/2}{\tan(\text{FOV}/2)}$$

其中 $W$ = 屏幕宽度（默认 1920），FOV = 103°，$f$ = 焦距。

**正变换**（像素 → 角度 → counts）：

$$\theta_x = \arctan\left(\frac{dx}{f}\right), \quad \theta_y = \arctan\left(\frac{dy}{f}\right)$$

$$\text{counts}_x = \theta_x \cdot f \cdot k_x,\quad \text{counts}_y = \theta_y \cdot f \cdot k_y$$

其中 $k_x, k_y$ 是鼠标灵敏度校准因子，将"归一化角度位移"映射到实际鼠标 counts。

**逆变换**（counts → 像素）：

$$dx = \tan\left(\frac{\text{counts}_x}{k_x \cdot f}\right) \cdot f$$

$$dy = \tan\left(\frac{\text{counts}_y}{k_y \cdot f}\right) \cdot f$$

### 2.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k_factor_x` | 1 | X 轴灵敏度映射系数 |
| `k_factor_y` | 1 | Y 轴灵敏度映射系数 |
| `game_fov` | 103 | 游戏水平 FOV |
| `bypass_strategy_mapping` | True | 为 True 时正逆变换均为 1:1（调试用） |

### 2.4 正逆对称性

正变换 `calculate_mouse_move` 和逆变换 `reverse_map` 使用完全相同的数学模型，确保 `reverse_map(calculate_mouse_move(dx, dy)) ≈ (dx, dy)` 精确。WorldModel 的 ego-motion 追踪不会因映射不对称产生累积漂移。

### 2.5 速度映射

速度映射使用线性近似（不经过 FOV 反正切），因为速度是微小增量：

```python
counts_per_sec = px_per_sec * k_factor
```

---

## 3. WorldModel — 状态估计与时空回溯

**文件**: `world_model.py`

### 3.1 自我位置追踪 (Ego-Motion)

WorldModel 维护准星在屏幕空间中的"绝对位置" `ego_pos_px`。

每帧：
1. 从 RingBuffer 读取两帧之间累计的鼠标 counts（人+AI）
2. 通过 `reverse_map_velocity` 将 counts 转为像素
3. 累积到 `ego_pos_px`

**关键设计**：使用 `get_total_delta_sum`（人+AI）而非 `get_cursor_delta_sum`（仅人）。因为 AI 的 SendInput 同样转动游戏相机，只有全量位移才能正确追踪真实视角。

### 3.2 时空回溯 (Temporal Rewind)

检测框来自 `t_capture` 时刻的画面，但准星位置是当前时刻的。需要将准星回溯到拍照瞬间：

```
ego_at_capture = ego_now - 从(t_capture - vh_latency)到now的所有位移
```

**`dynamic_vh_latency`** 是自适应视频硬件延迟（详见 §3.5），用于微调回退时间窗口。

### 3.3 检测→绝对世界坐标

检测框中心是相对于 capture_size 中心的偏移。转换为 TrackManager 绝对坐标：

```
abs_meas = (det_center - crop_center) + ego_at_capture
```

这样即使准星在移动，所有检测的绝对坐标都在同一个世界参考系中。

### 3.4 Predict-Ahead（时间前视）

**触发条件**: `predict_ahead = True`

预测目标在未来 `smart_lead` 秒后的位置：

$$\text{pred\_abs\_pos} = \text{abs\_position} + \text{velocity} \cdot \text{smart\_lead} + \frac{1}{2} \cdot \text{accel} \cdot \text{smart\_lead}^2$$

其中 `smart_lead = pipeline_lead + ctrl_lead`

**pipeline_lead**（管道延迟）:

$$\text{pipeline\_lead} = \text{base\_lead} \cdot \text{accel\_penalty} \cdot \text{cov\_penalty}$$

| 分量 | 来源 | 说明 |
|------|------|------|
| `base_lead` | `stream_ingress + software_lag + hardware_lag` 经 EMA | 总管道延迟估计 |
| `accel_penalty` | $1.0 + 0.0003 \cdot \|\text{accel}\|$, clamp [1.0, 1.4] | 加速度越大越需提前量 |
| `cov_penalty` | $1.0 - (\text{trace} - 450) / 1400$, clamp [0.75, 1.0] | 协方差大（新目标）时降低预测置信度 |

**ctrl_lead**（控制器剩余时间）:
调用 `controller.get_expected_lead()` 解析求解阻抗动力学收敛时间：
- BALLISTIC 阶段：`T_prog - t_elapsed`
- TRACKING 阶段：二阶系统 2% settling time $4/(\zeta \cdot \omega_n)$ + 饱和段

**加速度补偿上限**: $|\frac{1}{2}at^2|$ 钳制在 60px

### 3.5 自适应延迟进化引擎

当 ego 速度 > 80 px/s 时，利用 Kalman innovation 校准 `dynamic_vh_latency`：

$$\text{time\_error} = \frac{\text{innovation} \cdot \vec{v}_{ego}}{\|\vec{v}_{ego}\|^2}$$

$$\text{vh\_latency} \leftarrow \text{vh\_latency} + 0.02 \cdot \text{time\_error}$$

钳制在 [2ms, 50ms]。自动适应串流延迟的缓慢变化。

### 3.6 Sticky Target（粘滞目标）

多目标场景下，两人 conf 接近时 best 会在两人间来回跳。解决方法：

1. 按 conf 降序排列所有检测
2. 找到 conf 在 `best_conf - conf_margin` 内的候选项
3. 选中心离上一帧首选最近的那个
4. 只有当新旧中心距离 < `max_center_px` 时才粘滞

### 3.7 Control Speed Calibration（控制速度闭环校准）

**问题**：`ctrl_lead` 基于阻抗解析解计算预期收敛时间，但实际 arm_vel 受 LPF（τ≈12ms）、speed_cap、anti-orbit damping 和 1000Hz 离散化影响，响应偏慢。解析解预测的 lead 系统性地偏短，导致 BALLISTIC 着陆时有 ~5-15px 残差。

**解决**：EMA 学习实际 arm 速度与 Kalman 目标速度的比率，反比放大 ctrl_lead。

$$\text{ratio} = \frac{\|\vec{v}_{\text{target}}\|}{\|\vec{v}_{\text{arm}}\|}$$

$$\text{scale} \leftarrow 0.95 \cdot \text{scale} + 0.05 \cdot \text{ratio}$$

$$\text{ctrl\_lead} \leftarrow \text{ctrl\_lead} \cdot \text{scale}$$

**关键设计**：
- 只在 `arm_spd > 0.5` 且 `tgt_spd > 10` 时更新（静止时不学习）
- ratio 钳制在 [0.5, 2.0]，scale 钳制在 [0.80, 1.40]
- arm_vel 来自控制器 `crosshair_velocity`（上一帧的 arm LPF 后速度）
- 每 2 秒 INFO 日志输出当前 scale 值

**本质**：将控制器视为黑箱，直接学其有效增益，不修单个参数。这比修 lead time 更鲁棒——speed_cap、LPF、离散化等全部非线性被包进了一个变量。

**第二路校准：Arrival-Based（到达确认）**

第一路（上节）比较 arm 速度 vs Kalman 目标速度，依赖 Kalman 速度估计的准确性。第二路更直接：从 ring_buffer 取出本帧实际 AI 位移（总位移 − 人手位移），对比预期位移（`arm_vel × dt`）。

$$\text{arrival\_ratio} = \frac{\|\text{ai\_actual\_dspl}\|}{\|\text{arm\_vel} \cdot dt\|}$$

$$\text{correction} = \frac{1}{\text{clamp}(\text{arrival\_ratio},\ 0.15,\ \infty)}$$

$$\text{scale} \leftarrow 0.97 \cdot \text{scale} + 0.03 \cdot \text{correction}$$

**触发条件**：`prev_arm_spd > 50 ct/s` 且 `expected_spd > 5 ct` 且 `ai_spd > 3 ct`（AI 在主动控制时才校准）。

**与第一路的关系**：
- 第一路（velocity-based）：每帧更新（比率 0.05），响应快但依赖 Kalman 估计
- 第二路（arrival-based）：每帧更新（比率 0.03），更直接但只在 arm 显著移动时生效
- 两路**共同更新同一个 `_ctrl_speed_scale`**，scale 钳制放宽到 [0.70, 1.60]（arrival 校准需要更大范围应对 WAN jitter）

### 3.8 WAN 模式（Sunshine+Moonlight 远程串流）

**问题**：杭州笔记本运行代码 → Sunshine 采集温州台式机画面 → Moonlight 串流回来。链路总延迟 60-100ms，且每帧抖动 ±10-30ms。本地设计的 `dynamic_vh_latency` 范围（2-50ms）和自适应速率（0.02）完全不适用。

**三个核心改动**（由 `wan_mode = True` 触发）：

#### 3.8.1 延迟自适应范围扩大

| 参数 | 本地 | WAN |
|------|------|-----|
| `dynamic_vh_latency` 范围 | [2ms, 50ms] | [5ms, 150ms] |
| 自适应速率 α | 0.02 | 0.03 |
| 初始值 | 8ms | `wan_min_latency_ms × 0.6`（≈36ms） |
| time_error 钳制 | ±15ms | ±25ms |

**设计要点**：WAN 模式下速率降低（0.03 vs 0.02 等价），因为每秒 240 帧的原始 innovation 噪声太大，追帧级 jitter 会导致 `dynamic_vh_latency` 来回振荡。0.03 的速率等效时间常数 ~33 帧 ≈ 140ms，足以滤掉单帧 jitter。

#### 3.8.2 Pipeline Lead 最小地板

```python
if wan_mode:
    raw_lead_time = max(raw_lead_time, wan_min_latency_ms / 1000)
    raw_lead_time = clamp(raw_lead_time, 10ms, 300ms)
```

即使 `software_lag + base_hardware_lag` 只有 25ms，实际 WAN 链路至少 60ms。地板防止系统性低估 lead 导致准星永远在目标后面"追逐"。

#### 3.8.3 Jitter-Safe 平滑估计

```python
vh_lat_smoothed = 0.95 * vh_lat_smoothed + 0.05 * dynamic_vh_latency
```

`dynamic_vh_latency` 追踪短期残差变化（用于 temporal rewind），`vh_lat_smoothed` 是更强的 EMA（τ≈20 帧），用于 lead 计算，确保 predict_ahead 不会因单帧 jitter 而跳动。

**如何使用**：

```ini
[WorldModel]
wan_mode = True
wan_min_latency_ms = 60    # 你的实际体感延迟，保守填
wan_jitter_ms = 25         # 网络抖动幅度
moonlight_latency_ms = 50  # Moonlight 统计的网络延迟
```

启动后观察日志中的 `自适应 vh 延迟` 和 `WAN jitter-safe` 两行，确认延迟估计收敛到合理值（通常 50-90ms）。

---

## 4. Kalman Filter — 6 状态恒加速模型

**文件**: `world_model.py`（`kf6_predict`, `kf6_update`, `SimpleKalman`）

### 4.1 状态向量

$$\mathbf{x} = [p_x, p_y, v_x, v_y, a_x, a_y]^T$$

6 维状态：位置、速度、加速度。

### 4.2 预测步

状态转移矩阵：

$$\mathbf{F} = \begin{bmatrix} 
1 & 0 & \Delta t & 0 & \frac{1}{2}\Delta t^2 & 0 \\
0 & 1 & 0 & \Delta t & 0 & \frac{1}{2}\Delta t^2 \\
0 & 0 & 1 & 0 & \Delta t & 0 \\
0 & 0 & 0 & 1 & 0 & \Delta t \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

$$\hat{\mathbf{x}}_{k|k-1} = \mathbf{F} \hat{\mathbf{x}}_{k-1}$$

$$\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1} \mathbf{F}^T + \mathbf{Q}$$

### 4.3 更新步

观测矩阵只测量位置：

$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}$$

$$\mathbf{y} = \mathbf{z} - \mathbf{H}\hat{\mathbf{x}} \quad \text{(innovation)}$$

$$\mathbf{S} = \mathbf{H}\mathbf{P}\mathbf{H}^T + \mathbf{R}$$

$$\mathbf{K} = \mathbf{P}\mathbf{H}^T\mathbf{S}^{-1}$$

$$\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K}\mathbf{y}$$

$$\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}(\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K}\mathbf{R}\mathbf{K}^T$$

### 4.4 自适应过程噪声

根据 innovation 的 EMA 动态调整 Q：

$$\text{innov\_ema} \leftarrow \begin{cases}
0.8 \cdot \text{innov\_ema} + 0.2 \cdot \|\mathbf{y}\| & \text{if } \|\mathbf{y}\| > \text{innov\_ema} \\
0.9 \cdot \text{innov\_ema} + 0.1 \cdot \|\mathbf{y}\| & \text{otherwise}
\end{cases}$$

$$q_{\text{scale}} = 0.5 + 1.5 \cdot \text{clip}\left(\frac{\text{innov\_ema}}{30}, 0, 1\right)$$

$$\mathbf{Q} = \mathbf{Q}_{\text{base}} \cdot q_{\text{scale}}$$

大 innovation → 高 Q → Kalman 更信任量测（快速追踪突变）；小 innovation → 低 Q → 更信任模型（平滑）。

### 4.5 协方差初值

| 分量 | 初值 | 说明 |
|------|------|------|
| $\sigma_{\text{pos}}^2$ | 2 | 等于 R_base |
| $\sigma_{\text{vel}}^2$ | 400 | ~5×Q_vel，2-3 帧内收敛 |
| $\sigma_{\text{acc}}^2$ | 800 | ~4×Q_acc |

Trace 初值 = 2404 → cov_penalty 最初压到 0.75。几帧后 trace < 600 → cov_penalty 恢复到 1.0。

### 4.6 config 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `R` | 1.726 | 量测噪声协方差（对 $I_2$ 缩放） |
| `Q_pos` | 10.46 | 位置过程噪声 |
| `Q_vel` | 49.85 | 速度过程噪声 |
| `Q_acc` | 252.07 | 加速度过程噪声 |

---

## 5. TrackManager — 多目标匈牙利 IoU 匹配

**文件**: `perception/track_manager.py`

### 5.1 架构

每个物理目标分配独立 `track_id`，拥有独立的 `SimpleKalman`（含自适应 innov_ema 和 Q）。使用 Hungarian-IoU 贪心级联匹配（SORT 风格）。

### 5.2 匹配算法

每帧执行：

1. 计算所有 track × detection 的 IoU 矩阵
2. 取 IoU > `track_iou_threshold` 的对，按 IoU 降序排列
3. 贪心分配：每个 track 和 detection 最多匹配一次
4. **已匹配**：Kalman predict + update
5. **未匹配 track**：纯 predict（coast_count++）
6. **未匹配 detection**：spawn 新轨迹
7. coast_count > `track_max_coast`：删除轨迹

### 5.3 目标选择策略

由 `track_selection_policy` 控制，**所有策略统一经过切换日志**：

| 策略 | 规则 |
|------|------|
| `closest_to_crosshair` | min(‖track.abs_position - ego_pos_px‖) |
| `highest_conf` | max(conf_ema + 0.001×matches) |
| `newest` | max(last_matched) |
| `largest` | max(bbox_width × bbox_height) |

### 5.4 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `track_iou_threshold` | 0.25 | IoU 匹配阈值 |
| `track_max_coast` | 8 | 最大 coast 帧数后删除 |
| `track_max_tracks` | 12 | 最大活跃轨迹数 |
| `track_selection_policy` | closest_to_crosshair | 目标选择策略 |

---

## 6. CIPHER v1.0 控制器

**文件**: `controllers/pro_controller.py`

CIPHER = **C**ognitive **I**mpedance & **P**roprioceptive **H**armonic **E**xecution **R**untime

### 6.1 五大支柱

| 缩写 | 全称 | 职责 |
|------|------|------|
| **MPE** | Motor Program Engine | Fitts' Law + min-jerk 开环弹道 |
| **AIC** | Adaptive Impedance Control | 连续增益调度闭环跟踪 |
| **SEC** | Sub-Movement Error Correction | BALLISTIC ↔ TRACKING 两相切换 |
| **OUN** | Ornstein-Uhlenbeck Noise | 两层生物噪声（手指震颤 + 姿态漂移） |
| **FBL** | Fitts-Bio Loop | 生理计时噪声 + 欠射偏差 |

### 6.2 两相状态机

```
           dist > thresh_high
    ┌──────────────┐           ┌──────────────┐
    │   TRACKING   │──────────►│  BALLISTIC   │
    │  (闭环跟踪)   │           │ (开环运动程序) │
    └──────────────┘◄──────────└──────────────┘
                         tau >= 1.0
                      (或 startle 重规划)
```

**BALLISTIC 触发条件**:
- TRACKING 阶段：`dist > cipher_thresh_high_px · px_to_ct` 且距上次 > `cipher_prog_interval`
- BALLISTIC 阶段（startle 重规划）：`dist > prog_remain + max(6×head_r, 0.65×remain, 30px)`

---

### 6.3 MPE — 运动程序引擎

#### 6.3.1 Fitts' Law 时间模型

人类瞄准时间遵循 Fitts' Law：

$$T = a + b \cdot \log_2\left(\frac{2D}{W}\right)$$

其中 D = 距离（counts），W = 目标宽度 = 2×head_r_counts。

**两种 duration 模型**：

| 模式 | duration | 说明 |
|------|----------|------|
| `human_flick` | $T \sim \mathcal{N}(a + b \cdot \log_2(2D/W),\ \sigma_T \cdot T)$ | 模拟人类生理规划时间 |
| `pure_ai` | $T = 1.875D / v_{\text{max\_ai}} \times 1.05$ | 物理 min-jerk 下限（无视觉反馈延迟） |

**pure_ai 物理下限原理**: min-jerk 峰值速度 = $1.875 \cdot D/T$，受 v_max_ai 限制，$T_{\min} = 1.875D / v_{\text{max\_ai}}$，1.05 安全裕度。

#### 6.3.2 Min-Jerk 速度剖面 (Flash & Hogan 1985)

$$v(\tau) = 30\tau^2(1-\tau)^2, \quad \tau \in [0, 1]$$

$$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$$

**关键性质**:
- ∫₀¹ v(τ)dτ = 1.0（单位位移，缩放后得到实际）
- 峰值 1.875 在 τ = 0.5
- τ = 0 和 τ = 1 处速度、加速度均为零 → jerk 连续

**实际 BALLISTIC 速度**:

$$v_{\text{ballistic}} = \vec{d} \cdot v(\tau) \cdot \frac{D}{T} \cdot \text{brake} + \vec{v}_{\text{ff}} \cdot \text{ff\_coef}$$

- $\vec{d}$ = 发射方向（τ>0.4 时逐渐混合当前误差方向）
- `brake`：τ>0.65 时根据 undershoot 剩余量缩放（0.20~1.0）

#### 6.3.3 Direction Blending

BALLISTIC 执行期间目标可能移动。方向混合：

$$\text{mixed} = (1 - 2.5\tau) \cdot \vec{d}_{\text{prog}} + 2.5\tau \cdot \vec{d}_{\text{cur}},\quad \tau < 0.4\text{ 时纯程序方向}$$

#### 6.3.4 Undershoot（欠射）

BALLISTIC 故意在目标前停住，留 2-8px 给 TRACKING 做"微调收尾"：

$$u \sim \mathcal{N}(\text{head\_r} \cdot \text{undershoot\_loc},\ \sigma_u \cdot \text{px\_to\_ct})$$

钳制在 [1.5·px_to_ct, min(12·px_to_ct, 0.3D)]

#### 6.3.5 BALLISTIC → TRACKING 尾段 Blend

τ > 0.80 时，最后 20% 窗口内混合 TRACKING impedance 输出：

$$v_{\text{cmd}} = (1 - \text{blend}) \cdot v_{\text{ballistic}} + \text{blend} \cdot v_{\text{tracking\_preview}}$$

消除相变处的速度跳变，避免 >5000 ct/s² 的加速度脉冲。

#### 6.3.6 TRACKING 着陆

- `human_flick`：激活 entry_ticks（12 ticks，死区放大到 2.5px），保留 1-3px 微调
- `pure_ai`：跳过 entry_ticks，arm_vel×0.35 软着陆

---

### 6.4 AIC — 自适应阻抗控制

#### 6.4.1 阻抗控制律

核心方程（**相对阻尼**变体，消除稳态滞后）：

$$v_{\text{des}} = \left(K \cdot \text{pos\_gain} \cdot \vec{e} - B \cdot (\vec{v}_{\text{arm}} - \vec{v}_{\text{ref}})\right) \cdot dg \cdot \text{power}$$

**Soft Deadzone**（二次缓入，避免硬切换）：

$$\text{pos\_gain} = \begin{cases}
0 & \text{dist} < dz\_r \\
\left(\frac{\text{dist} - dz\_r}{2 \cdot dz\_r}\right)^2 & dz\_r \leq \text{dist} < 3 \cdot dz\_r \\
1.0 & \text{dist} \geq 3 \cdot dz\_r
\end{cases}$$

**相对阻尼优势**：当 $v_{\text{arm}}=v_{\text{ref}}$ 时阻尼项归零，稳态误差仅由 ff_gain 残差决定（<5px），而不像绝对阻尼 $B\cdot v$ 那样 $e_{ss} \approx B\cdot v_{\text{target}}/K$（可达 25+px）。

#### 6.4.2 连续增益调度

全部按归一化距离平滑插值，**无硬切换**：

$$d_{\text{norm}} = \frac{\text{dist}}{\text{head\_r}}$$

$$\text{near\_w} = \text{clip}(1 - d_{\text{norm}},\ 0,\ 1) \quad \text{（头部内=1，远离=0）}$$

$$w_{\text{flick}} = \text{sigmoid}\left(\frac{\text{dist}}{\text{thresh\_high}} - 1\right)$$

**K 调度**：

$$K_{\text{mid}} = K_{\text{pursuit}}(1-w_{\text{flick}}) + K_{\text{flick}} \cdot w_{\text{flick}}$$
$$K_{\text{eff}} = K_{\text{correction}} \cdot \text{near\_w} + K_{\text{mid}} \cdot (1-\text{near\_w})$$

**B 调度**：

$$B_{\text{eff}} = B_{\text{correction}} \cdot \text{near\_w} + B_{\text{pursuit}} \cdot (1-\text{near\_w})$$

| 区域 | d_norm | K 倾向 | B 倾向 |
|------|--------|--------|--------|
| 头部内 | <1 | K_correction (86) | B_correction (0.35) |
| 近距追踪 | 1~3 | K_pursuit (212) | B_pursuit (0.48) |
| 远距 flick | ≫1 | → K_flick (511) | B_pursuit |

#### 6.4.3 Depth Gain（深度增益）

根据 bbox 大小估计目标距离，调整角速度：

$$dg = \text{clip}\left(\frac{\text{depth\_ref\_bbox}}{\max(\text{bbox\_w}, 10)},\ 0.45,\ 2.8\right)$$

远距小目标 → 更大 dg → 更快速度。

#### 6.4.4 前馈速度门控 (human_flick 模式)

近端按目标速度动态降权（防 Kalman 噪声），远端全开：

$$\text{ff\_scale\_near} = \begin{cases}
\text{ff\_scale\_min} & t_{\text{speed}} < \text{knee} \\
\text{线性爬升} & \text{knee} \leq t_{\text{speed}} < \text{knee}+\text{scale} \\
1.0 & t_{\text{speed}} \geq \text{knee}+\text{scale}
\end{cases}$$

$$\text{ff\_scale} = \text{ff\_scale\_near} \cdot \text{near\_w} + 1.0 \cdot (1-\text{near\_w})$$

pure_ai 模式：ff_scale 恒为 1.0。

#### 6.4.5 阻尼参考速度

| 模式 | 近端 (near_w→1) | 远端 (near_w→0) |
|------|-----------------|------------------|
| human_flick | vref=0（绝对阻尼，抗噪） | vref=ff_vel·power（相对阻尼） |
| pure_ai | vref=ff_vel·power | vref=ff_vel·power（全相对） |

---

### 6.5 速度前馈

$$\vec{v}_{\text{ff}} = \vec{v}_{\text{target}} \cdot \text{ff\_gain} + \vec{a}_{\text{target}} \cdot \text{ff\_acc\_gain\_sec}$$

加速度项 ≈ 11.6ms 二阶预瞄，应对 adad/slide。

---

### 6.6 后处理滤波器链

```
v_cmd ──[Arm LPF τ≈12ms]──► arm_vel ──[Anti-orbit]──► speed_cap ──► output
```

**Arm LPF**（一阶低通，~20Hz 截止）：

$$\alpha = 1 - e^{-\Delta t / \tau},\quad \tau = 12\text{ms}$$

$$v_{\text{arm}} \leftarrow (1-\alpha)v_{\text{arm}} + \alpha \cdot v_{\text{cmd}}$$

**Anti-Orbit Damping**：抑制近距圆周振荡。将 arm_vel 分解为径向+切向，切向分量乘以 $\sqrt{\text{dist}/\text{anti\_orbit\_dist}}$。作用范围 1e-3 < dist < 35·px_to_ct。

**Global Speed Cap**：

| 模式 | v_max (ct/s) | 等效 px/s |
|------|-------------|-----------|
| human_flick | cipher_max_speed (6611) | ~1970 |
| pure_ai | max(cipher_max_speed, 9500) | ~2840 |

---

### 6.7 get_expected_lead() — 收敛时间解析

**BALLISTIC**：直接返回剩余程序时长 `T_prog - t_elapsed`

**TRACKING**：二阶阻抗动力学解析

系统方程（归一化）：

$$\frac{d^2e}{dt^2} + \frac{1}{\tau}\frac{de}{dt} + \frac{K \cdot dg \cdot \text{power}}{\tau} \cdot e = 0$$

$$\omega_n = \sqrt{\frac{K \cdot dg \cdot \text{power}}{\tau}},\quad \zeta = \frac{1}{2\sqrt{K \cdot dg \cdot \text{power} \cdot \tau}}$$

分段求解（钳制在 [0, 250ms]）：

$$e_{\text{sat}} = v_{\max} / \text{gain}$$

$$t_s = \begin{cases}
\frac{4}{\zeta \cdot \omega_n} & \text{err} \leq e_{\text{sat}} \quad\text{(线性区 2% settling)} \\
\frac{\text{err} - e_{\text{sat}}}{v_{\max}} + \frac{4}{\zeta \cdot \omega_n} & \text{err} > e_{\text{sat}} \quad\text{(饱和+线性)}
\end{cases}$$

---

### 6.8 OUN — 生物噪声模型

#### Finger Tremor

OU 过程作为**速度扰动**注入 arm_vel：

$$dX_t = -\theta X_t dt + \sigma dW_t,\quad \theta = 20,\ \tau_{\text{corr}} \approx 50\text{ms}$$

$$\text{noise\_vel} = X_t \cdot \text{ou\_vel\_scale}$$

再经 wrist LPF（τ=8ms, ~20Hz），天然滤掉 >20Hz 分量。

**强度分级**：

| 阶段 | sigma 系数 | 场景 |
|------|-----------|------|
| BALLISTIC | 1.0×ou_sigma_ball (0.29) | 甩枪中噪声小 |
| TRACKING > head_r | 1.0×ou_sigma_track (0.24) | 追踪 |
| < head_r | 0.70×ou_sigma_track | 近距减噪 |
| < 0.5×head_r | 0.40×ou_sigma_track | 极近压噪 |

稳态量级：vel RMS ≈ 10.4 ct/s，pos RMS ≈ 0.15 px。

#### Postural Drift

OU 过程作为**位置扰动**：

$$dX_t = -\theta X_t dt + \sigma dW_t,\quad \theta = 0.5,\ \tau_{\text{corr}} \approx 2\text{s}$$

$$\text{drift\_pos} = X_t \cdot \text{drift\_pos\_scale}$$

天然低频谱（<0.5Hz），不污染加速度谱。

---

### 6.9 输出层 (tick_mouse)

1000Hz 物理时钟驱动。信号链：

```
arm_vel + tremor_vel ──► [Wrist LPF τ≈8ms] ──► wrist_vel
wrist_vel·dt + drift_pos + subpixel_residual ──► displacement
floor(displacement) ──► (mx, my) ──► SendInput
```

亚像素累加器保留 floor 残差，避免整数量化误差累积。

---

### 6.10 冷启动与接管

**冷启动斜坡** (human_flick，pure_ai 跳过)：

$$\text{age\_ramp} = \text{clip}\left(0.83 + 0.17 \cdot \frac{\text{age\_ms}}{32.7},\ 0.83,\ 1.0\right)$$

**Power 平滑** (human_flick)：EMA，τ=25ms。pure_ai 直通。

**热启动** `warm_start_from_velocity`：唤醒时预种 arm_vel = seed×0.7，spf 跳到 0.30，跳过冷启动。

**渐变冻结** `soft_freeze`：停手时 arm_vel 指数衰减（τ=12ms），避免瞬切。

---

### 6.11 Rust 原生引擎

当 `rust_enable = True` 且已编译时，`_minjerk_vel`、`_minjerk_pos`、`_impedance_vel` 三个核心函数替换为 Rust 原生实现。未编译则回退 Numba 并打印 WARNING。

### 6.12 参数降维（Functional Parameter Reduction）

**问题**：CIPHER 有 30+ 个独立 config 参数（K/B/τ/threshold/OU…），每个都是 CMA-ES 独立搜索出的 magic number。这些参数之间有明确的物理关系（如 K 和 B 应满足阻尼比约束），但独立优化破坏了这些关系，导致参数耦合严重、调参空间爆炸。

**解决**：引入 4 个可选派生参数，设为 >0 时函数化推导替代独立参数。设为 0 时保持向后兼容（使用显式值）。

#### 6.12.1 K→B 阻尼比推导

**物理关系**（近似一阶 LPF 后的闭环阻抗）：

$$B_{\text{eff}} = \max\left(0.05,\ 2\zeta\sqrt{K \cdot \tau_{\text{arm}}} - 1\right)$$

| 派生参数 | 默认值 | 推导目标 | 原独立参数 |
|----------|--------|----------|-----------|
| `cipher_zeta_correction` | 0 | `B_correction` | `cipher_b_correction` |
| `cipher_zeta_pursuit` | 0 | `B_pursuit` | `cipher_b_pursuit` |

启用后，3 个 K 值 + 2 个 ζ 值替代 5 个独立 K/B 参数。实际减少 1 个自由度，但建立了 B 跟随 K 的物理约束——调 K 时 B 自动联动。

#### 6.12.2 阈值滞回比率

| 派生参数 | 默认值 | 推导 |
|----------|--------|------|
| `cipher_thresh_hyst_ratio` | 0 | `thresh_low = thresh_high × ratio` |

两个独立阈值 → 1 个主阈值 + 1 个比率。比率 ≈0.95 意味着高/低阈值间有 5% 的滞回带。

#### 6.12.3 滤波器 τ 比率

| 派生参数 | 默认值 | 推导 |
|----------|--------|------|
| `cipher_tau_wrist_ratio` | 0 | `tau_wrist = tau_arm × ratio` |

两个独立时间常数 → 1 个主 τ + 1 个比率。比率 ≈0.5~0.55 保持 wrist 比 arm 快约 2 倍。

**净效果**：这些派生参数全部启用后，可调参数减少 ~5 个（K/B 耦合解除 + 阈值/τ 联动），且每次调 K 时 B 自动联动，降低调参出错率。

---

## 7. Agent 编排层

**文件**: `agent.py`

### 7.1 chase_mode 判定与切换

| 模式 | 首帧触发条件 | 行为特征 |
|------|-------------|----------|
| `pure_ai` | 人类近 100ms 速度 < 500 px/s（目标自行入框） | ff_scale=1.0、entry_ticks=0、v_max=9500 |
| `human_flick` | 人类近 100ms 速度 > 500 px/s（人拉枪） | entry_ticks=12、v_max=6611 |

**自动切换**：human_flick 下人类 80ms 速度 < 450 px/s 且锁定时长 > 150ms → 自动切 pure_ai（有 INFO 日志确认）。

**切换确认**：agent 设置 `reset_target_state(mode=X)` 后立即读回 `controller.chase_mode`，不一致则打 ERROR。

### 7.1.1 三帧确认计数（Lock Confirmation Counter）

**问题**：原逻辑中，只要单帧检测满足 `conf ≥ min_aim_conf` 且 `is_valid=True`，立即锁定目标（`Target acquired`）。这会带来两个问题：

1. **单帧误检**：YOLO 偶发的 False Positive（背景误识别为目标）会瞬间触发锁定 → 准星被拉到错误位置
2. **噪声锁假目标**：低置信度抖动导致 conf 在门限附近穿零 → 锁→丢→锁 循环，日志刷屏

**解决方案**：引入多帧确认计数器 `_lock_confirm_count`。

**状态机**：

```
                  conf < min_aim_conf
    ┌─────────────────────────────────────┐
    │           (reset counter)           │
    ▼                                      │
┌────────┐    conf≥min_aim     ┌────────────┐   count≥N   ┌──────────┐
│  IDLE  │ ──────────────────► │ CONFIRMING │ ──────────► │  LOCKED  │
│ count=0│                     │ count=1..N │             │ (ACQUIRE)│
└────────┘                     └────────────┘             └──────────┘
    ▲                              │                            │
    │         conf<min_aim         │                            │
    └──────────────────────────────┘                            │
                                                               │
    ┌──────────────────────────────────────────────────────────┘
    │         DROP (invalid streak ≥ n 或 conf < min_drop)
    ▼
┌────────┐
│  IDLE  │
│ count=0│
└────────┘
```

**配置参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `aim_lock_confirm_count` | 3 | 目标需连续 N 帧有效后才锁定。设为 1 可完全禁用（等同于立即锁定） |

**与已有机制的关系**：

| 机制 | 方向 | 默认值 | 作用 |
|------|------|--------|------|
| `aim_lock_confirm_count` | 锁定前 | 3 | 连续 N 帧有效才锁（防假阳性） |
| `aim_drop_invalid_frames` | 锁定后 | 2 | 连续 N 帧无效才拆（防短丢框） |

两者形成 **"进严出宽"** 的滞回门控：确认需要 3 帧才能锁定（过滤噪声），但拆锁允许短暂 2 帧容忍（避免频繁重锁）。

**计数器重置条件**：
- 任意帧 `conf < min_aim_conf`（未锁态）
- 任意帧 `is_valid == False` 或 `p_predict is None`（触发了 DROP）
- 已锁态 `conf < min_aim_conf_drop`（拆锁）
- COAST 阶段不影响计数器（coast 发生在已有轨迹 > 5 帧丢框时，不参与锁判定）

### 7.2 Power Factor（三层闸门）

$$\text{power\_factor} = \text{spatial} \times \text{reaction} \times \text{human\_override}$$

**Spatial（空间衰减）**：

$$\text{spatial} = \text{clip}\left(1 - (\text{error}/800)^2,\ 0.1,\ 1.0\right)$$

接近目标降速防过冲。

**Reaction**：

| 模式 | reaction_factor |
|------|-----------------|
| pure_ai | 1.0（无视觉反应延迟） |
| human_flick | clip((t - t_flick_end)/0.03, 0.30, 1.0) |

**Human Override（人机离合器）**：

近距(误差 < `human_clutch_near_err_px`，默认 90) 用 **低 s_min**（默认 55px/s 起就分权），避免旧版 1200 导致准星在头上时手一慢 AI 全功率。再向 150px 外线性过渡到远距 150/800。详见 `[General] human_clutch_near_*`。

由 `human_speed` 与上表 $s_{\min},s_{\max}$ 先算目标主导度 $h_{\text{target}}$（同旧式分段线性），再经 `utils/human_ai_envelope.step_human_dominance` 做不对称平滑得 $H$，最终 `human_override = 1 - H`（唯一路径，无 pure_ai 旁路）。**clutch_window** 仍自适应（flick_end 后 100ms 内从 20ms → 100ms）。

### 7.3 Flick End 检测

当 `_prev_human_flicking and not cur_human_flicking`（人类速度 < 450 px/s）：

1. `notify_flick_end()` → 清惯性
2. `warm_start_from_velocity(人速×0.7 + 目标速×0.3)`

### 7.4 ~~到点停手~~（已移除）

人机权仅 `human_ai_envelope` 包络 + `s_min/s_max` 带；不再做进带冻结 / 路径唤醒 / `set_block_aimbot_move`。

---

## 8. 附录：config.ini 参数速查

### [General]

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_aim_conf` | 0.32 | 首次锁目标需 conf ≥ 此值 |
| `min_aim_conf_drop` | 0.20 | 已锁后 conf 跌到此以下才丢 |
| `aim_drop_invalid_frames` | 2 | 连续无效帧数丢锁 |
| `aim_lock_confirm_count` | 3 | 连续有效帧数才锁（三帧确认） |
| `human_ai_envelope_*` | 见 runtime_defaults | 包络时间常数（attack/release/decel/sustain） |
| `human_input_backend` | pynput | 人手输入后端 |

### [Kalman]

| 参数 | 默认值 |
|------|--------|
| `R` | 1.726 |
| `Q_pos` | 10.46 |
| `Q_vel` | 49.85 |
| `Q_acc` | 252.07 |

### [WorldModel]

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_hardware_lag` | 0.019s | 硬件延迟 |
| `moonlight_latency_ms` | 0 | 串流延迟 |
| `predict_ahead` | True | 时间前视 |
| `use_total_delta` | True | 全量位移追踪 ego |
| `adaptive_latency_enable` | True | 自适应延迟 |
| `ctrl_lead_enable` | True | 控制器 lead |
| `sticky_target_enable` | True | 多目标粘滞 |
| `track_iou_threshold` | 0.25 | IoU 阈值 |
| `track_max_coast` | 8 | 最大 coast 帧 |
| `track_selection_policy` | closest_to_crosshair | 目标选择策略 |
| `wan_mode` | False | WAN/远程串流模式 |
| `wan_min_latency_ms` | 60 | WAN 最小有效延迟 |
| `wan_jitter_ms` | 25 | WAN 抖动幅度 |

### [AimStrategy]

| 参数 | 默认值 |
|------|--------|
| `k_factor_x` | 1 |
| `k_factor_y` | 1 |
| `bypass_strategy_mapping` | True |

### [Controller] — CIPHER

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rust_enable` | False | Rust 引擎开关 |
| `mode_threshold_high` | 50 px | WM flick→track |
| `mode_threshold_low` | 35 px | WM track→flick |

| CIPHER 参数 | 默认值 | 说明 |
|-------------|--------|------|
| `cipher_max_speed` | 6611 ct/s | 人类手速上限 |
| `cipher_k_correction` | 86.1 | Correction K |
| `cipher_k_pursuit` | 211.9 | Pursuit K |
| `cipher_k_flick` | 510.8 | Flick K |
| `cipher_b_correction` | 0.347 | Correction B |
| `cipher_b_pursuit` | 0.479 | Pursuit B |
| `cipher_ff_gain` | 0.921 | 速度前馈增益 |
| `cipher_ff_acc_sec` | 0.012s | 加速度前馈 |
| `cipher_deadzone_scale` | 0.054 | 死区缩放 |
| `cipher_thresh_high_px` | 43.6 | flick 触发 |
| `cipher_thresh_low_px` | 41.5 | track 回落 |
| `cipher_fitts_a` | 0.026s | Fitts 截距 |
| `cipher_fitts_b` | 0.043s/bit | Fitts 斜率 |
| `cipher_fitts_T_min` | 0.039s | 最小弹道时间 |
| `cipher_fitts_T_max` | 0.254s | 最大弹道时间 |
| `cipher_undershoot_loc` | 0.025 | 欠射均值 |
| `cipher_undershoot_sigma` | 2.29 | 欠射标准差 |
| `cipher_tau_arm` | 0.010s | Arm LPF τ |
| `cipher_tau_wrist` | 0.005s | Wrist LPF τ |
| `cipher_ou_sigma_ball` | 0.292 | BALLISTIC 噪声 |
| `cipher_ou_sigma_track` | 0.237 | TRACKING 噪声 |
| `cipher_ou_vel_scale` | 154.4 | 噪声速度缩放 |
| `cipher_drift_sigma` | 0.134 | 漂移噪声 |
| `cipher_drift_pos_scale` | 0.015 | 漂移位置缩放 |
| `cipher_ff_speed_knee` | 471.8 | 前馈膝点 |
| `cipher_ff_speed_scale` | 2605.2 | 前馈斜率 |
| `cipher_ff_scale_min` | 0.557 | 前馈最小 |
| `cipher_ramp_ms` | 32.7ms | 冷启动爬升 |
| `cipher_ramp_min` | 0.827 | 冷启动最低 |
| `cipher_prog_interval` | 0.072s | 弹道冷却 |
| `cipher_entry_ticks` | 12 | 微调 tick 数 |
| `cipher_entry_dz_px` | 2.5px | 微调死区 |
| `cipher_zeta_correction` | 0 (可选) | B_correction 阻尼比 ζ |
| `cipher_zeta_pursuit` | 0 (可选) | B_pursuit 阻尼比 ζ |
| `cipher_thresh_hyst_ratio` | 0 (可选) | thresh_low=high×ratio |
| `cipher_tau_wrist_ratio` | 0 (可选) | tau_wrist=arm×ratio |
