# YOLOv13 Aimbot — 系统架构文档

> **版本**: CIPHER v1.0 | **目标游戏**: Valorant (UE4/5) | **语言**: Python 3.12+ | **平台**: Windows 10/11

---

## 目录

1. [系统概述](#1-系统概述)
2. [数据流与线程拓扑](#2-数据流与线程拓扑)
3. [目录结构与文件职责](#3-目录结构与文件职责)
4. [核心模块详解](#4-核心模块详解)
5. [配置系统](#5-配置系统)
6. [管线阶段分解](#6-管线阶段分解)
7. [已知问题与 Bug](#7-已知问题与-bug)

---

## 1. 系统概述

本系统是一个基于 YOLO 视觉检测 + 卡尔曼滤波 + 生物力学控制模型的 **AI 辅助瞄准系统**（aimbot）。核心流程：

```
屏幕抓取 → YOLO/HSV 目标检测 → 卡尔曼滤波追踪 → 时空校正 → 力反馈控制器 → SendInput 鼠标输出
```

**关键特性**：

| 特性 | 实现 |
|---|---|
| 目标检测 | YOLO26 TensorRT 推理 (256×256)，或 HSV 色球 (Aim Lab) |
| 目标追踪 | 6 状态 Kalman 滤波器 (位置/速度/加速度)，Numba JIT 加速 |
| 时空对齐 | 拍照—现在 位移回退 + 自适应延迟引擎 + 复合预测前馈 |
| 控制模型 | CIPHER v1.0：Min-Jerk 弹道 + 自适应阻抗 + OU 神经噪声 |
| 人机分离 | RingBuffer 分通道记录人类 (RawInput) 与 AI (SendInput) 位移 |
| 到点停手 | 误差入带自动停手，人手移动或目标漂出带自动唤醒 |
| 拟人化 | 人类甩枪检测、反应延迟斜坡、人机动态离合器 |

---

## 2. 数据流与线程拓扑

```
┌──────────────────┐     FrameBus      ┌──────────────────┐
│  CaptureThread   │ ───────────────→  │ InferenceThread  │
│  (DXCam 240fps)  │    BGR frame      │  (YOLO / HSV)    │
│  ┌────────────┐  │                   │  ┌─────────────┐ │
│  │ frame_ready │──┼────────────────→ │  │ frame_ready  │ │
│  │   .set()    │  │    cap_event      │  │   .wait()   │ │
│  └────────────┘  │                   │  └─────────────┘ │
└──────────────────┘                   └────────┬─────────┘
                                                │ update_detections()
                                                │ + frame_ready.set()
                                                ▼
┌──────────────────┐                   ┌──────────────────┐
│ HumanMouseListener│                  │   WorldModel     │
│ (RawInput /pynput)│──RingBuffer────→│   + AIAgent      │
│    is_ai=False    │  (人/AI 分通道)  │   tick() 主循环   │
└──────────────────┘                   └────────┬─────────┘
                                                │ compute()
                                                ▼
┌──────────────────┐                   ┌──────────────────┐
│  TriggerWorker   │                  │   MouseWorker    │
│  (异步 click)     │                  │   (1000Hz 输出)   │
└──────────────────┘                   └────────┬─────────┘
                                                │ SendInput
                                                ▼
                                         ┌──────────────┐
                                         │  游戏客户端    │
                                         └──────────────┘
```

**六线程拓扑**：

| 线程 | 频率 | 职责 |
|---|---|---|
| `CaptureThread` | ~240 Hz | DXCam 抓屏，写 FrameBus，置 capture_event |
| `InferenceThread` | ~240 Hz | 消费 FrameBus，YOLO/HSV 推理，调 `update_detections()`，置 `frame_ready_event` |
| `MainThread` (tick) | ~200-300 Hz | 等 `frame_ready_event`，调 `world_model.step()` + `controller.compute()` |
| `MouseWorker` | 1000 Hz | 调 `controller.tick_mouse()`，通过 SendInput 下发 |
| `HumanMouseListener` | 阻塞 | RawInput / pynput 钩子，写入 RingBuffer（is_ai=False） |
| `TriggerWorker` | 按需 | 异步 click down → sleep → click up |

---

## 3. 目录结构与文件职责

```
yolov13-aimbot/
├── main.py                          # 入口：高精度定时器、信号处理、热键绑定、主循环
├── agent.py                         # AIAgent 主驱动：线程管理、tick()、扳机、到点停手
├── world_model.py                   # WorldModel：Kalman、ego 追踪、时空对齐、预测前馈
├── config.py                        # Config 单例：INI 解析、类型转换
├── config.ini                       # 所有可调参数（编辑此文件即可）
├── inference.py                     # InferenceThread：YOLO26 TensorRT 推理循环
├── inference_aimlab.py              # AimlabBallInferenceThread：HSV 色球检测
├── output.py                        # SystemMouse：SendInput 封装、AI 签名水印
├── requirements.txt                 # pip 依赖
│
├── perception/                      # —— 感知层 ——
│   ├── bus.py                       # FrameBus：发布/订阅单帧总线 (Drop-Oldest)
│   ├── capture.py                   # CaptureThread：DXCam 截屏 (BGR, touchscreen backend)
│   └── ring_buffer.py              # RingBuffer：人/AI 位移分离记录
│
├── controllers/                     # —— 控制层 ——
│   ├── base_controller.py           # BaseController 抽象基类：pos/vel/backlog
│   ├── controller_factory.py        # ControllerFactory：按 config 创建控制器
│   ├── simple_controller.py         # SimpleController：直通调试 (无 tick_mouse)
│   ├── pid_controller.py            # PIDController：PID 控制器
│   ├── pro_controller.py            # PROController (CIPHER v1.0)：主力控制器
│   └── humanize/
│       └── humanize.py              # Humanizer：生物噪声、惯性动量 (默认关)
│
├── aim_strategies/                  # —— 瞄准策略层 ——
│   ├── factory.py                   # create_aim_strategy()：按 config 创建策略实例
│   └── valorant/
│       └── strategy.py              # ValorantStrategy：像素↔counts FOV 非线性映射
│
├── utils/                           # —— 工具层 ——
│   ├── types.py                     # 核心数据结构：Detection, InferenceContext
│   ├── runtime_defaults.py          # 非炼丹常数：人机包络、Aimlab 色域
│   ├── human_ai_envelope.py         # 人手/AI 权不对称包络
│   ├── aim_class_filter.py          # 目标类别白名单过滤
│   ├── aim_diagnostics.py           # 瞄准摆振诊断：翻转计数、提示生成
│   ├── logging_bootstrap.py         # 日志初始化：从 config 读 level
│   ├── recorder.py                  # TraceRecorder：帧上下文序列化 (deque)
│   ├── helpers.py                   # 通用工具：clamp/lerp/smoothstep/letterbox
│   ├── debug_reader.py              # 离线读取 debug_trace.pkl
│   ├── test_hardware.py             # 硬件延迟/输入延迟诊断脚本
│   └── analyze_fail.py              # 失败案例离线分析
│
├── test/                            # —— 仿真与测试 ——
│   ├── simulation.py                # 仿真框架（离线验证控制器）
│   ├── sim_agent.py                 # 仿真 Agent（与实战 agent 接口一致）
│   ├── control.py                   # 仿真评分引擎 (Precision/TTK/BioBonus)
│   ├── animate_trajectory.py        # 轨迹可视化
│   ├── _bench.py                    # CMA-ES 参数优化基准
│   ├── _noise_diag.py              # 噪声诊断
│   ├── _score_breakdown.py          # 评分拆解
│   ├── _trace_movement.py           # 移动追踪回放
│   └── repair.py                    # 参数修复
│
└── md/                              # —— 文档 ——
    └── ARCHITECTURE.md              # 本文件
```

---

## 4. 核心模块详解

### 4.1 `main.py` — 入口

```
流程：
  1. setup_root_logging()    —— 从 config.ini 读日志等级
  2. timeBeginPeriod(1)      —— 强制 Windows 定时器 1ms 精度
  3. AIAgent()               —— 构造所有子系统
  4. agent.start()           —— 启动全部子线程
  5. agent.tick() 循环       —— 每帧执行瞄准/扳机（阻塞等 frame_ready）
  6. Ctrl+C → agent.stop()   —— 优雅退出
```

热键：
- `P` → 暂停/继续
- `Alt+F1` → 开关自瞄

### 4.2 `agent.py` — AIAgent 主驱动

`tick()` 方法 ≈ 整个系统的"一帧"逻辑：

```python
tick():
  1. wait(frame_ready_event)          # 阻塞等待推理结果
  2. world_model.step(ctx, ring_buffer)  # Kalman 更新 + 预测
  3. 目标有效性校验 (is_valid / conf / coast)
  4. chase_mode 判定 (human_flick vs pure_ai)
  5. 到点停手检测与唤醒
  6. aim_strategy.calculate_mouse_move()  # 像素 → counts
  7. 人机动态离合器 (human_override)
  8. controller.compute(target, dt, v_real, a_real, power)
  9. _check_and_trigger()             # 扳机判定
```

**chase_mode** 判定逻辑：
- 100ms 内人类速度 > 500 px/s → `human_flick`（人拉枪，AI 补枪微调）
- 否则 → `pure_ai`（AI 独立瞄准）
- 人放手后 80ms 速度 < 450 px/s → 自动切回 `pure_ai`

**到点停手 (On-target hands off)**：
- 误差 < 32px (Linf) 且臂速 < 300 ct/s → 停 AI
- 人手移动或误差 > 40px → 自动唤醒
- 唤醒后 30ms 内 power 线性爬升（避免跳变脉冲）

### 4.3 `world_model.py` — WorldModel

**Kalman 滤波器**：6 状态 (x, y, vx, vy, ax, ay)，Numba JIT 加速。

```
predict:  state = F·state        F = [1 0 dt 0 dt²/2 0  ]
         cov = F·cov·Fᵀ + Q          [0 1 0  dt 0  dt²/2]
                                      [0 0 1  0  dt  0   ]
                                      [0 0 0  1  0   dt  ]
                                      [0 0 0  0  1   0   ]
                                      [0 0 0  0  0   1   ]

update:   innovation = meas - H·state
          K = cov·Hᵀ·S⁻¹
          state += K·innovation
```

**ego_pos_px 物理时钟**：
- 准星始终在实时位置（含 AI SendInput 的相机旋转）
- `ego_at_capture = ego_pos_px − 拍照至今的全量位移回退`
- 配合动态 vh 延迟自适应微调

**时空对齐与预测前馈**：

```
pipeline_lead = (stream_ingress + software_lag + base_hardware_lag)
              × accel_penalty(1.0~1.4)
              × cov_penalty(0.75~1.0)

smart_lead = pipeline_lead + ctrl_lead  (ctrl_lead: 控制器预期执行时长)

p_predict = kalman_pos + kalman_vel × smart_lead + ½·kalman_acc × smart_lead²
```

**自适应延迟引擎**（可选，`adaptive_latency_enable`）：
- 当 ego 速度 > 80 px/s 时，用 `innovation[0]·vx + innovation[1]·vy` 投影估计延迟偏差
- 每帧 EMA 微调 `dynamic_vh_latency`（2~50ms 范围）

**多目标粘滞**（可选，`sticky_target_enable`）：
- 多人 conf 接近时，优先保持上一帧首选目标，避免框间跳变

### 4.4 `controllers/pro_controller.py` — CIPHER v1.0

核心控制器，模拟人类神经肌肉运动控制。

| 子系统 | 功能 |
|---|---|
| **MPE** (Motor Program Engine) | Min-Jerk 弹道：`v(τ)=30τ²(1-τ)²`，Fitts' Law 规划时长 |
| **AIC** (Adaptive Impedance) | 弹簧-阻尼追踪：`v_des = (K·e - B·v) × depth × power` |
| **SEC** (Sub-Movement Error Correction) | 三段修正：BALLISTIC → PURSUIT → CORRECTION |
| **OUN** | Ornstein-Uhlenbeck 神经噪声：手指震颤 (τ≈50ms) + 姿态漂移 (τ≈2s) |
| **FBL** (Fitts-Bio Loop) | Fitts 时长高斯扰动 + 欠冲采样 |

关键接口：

| 方法 | 调用方 | 功能 |
|---|---|---|
| `compute(target_x, target_y, dt, v_real, a_real, power_factor)` | agent.tick() | 设置目标，计算指令速度 |
| `tick_mouse()` | MouseWorker (1kHz) | 消费指令速度，返回 SendInput counts |
| `get_expected_lead()` | world_model | 返回 BALLISTIC 轨迹剩余时间 (用于复合 lead) |
| `reset_target_state(mode)` | agent tick | 目标切换时重置内部状态 |
| `notify_flick_end()` | agent tick | 人甩枪结束，清积分 + 热启动 |
| `warm_start_from_velocity(vx, vy)` | agent tick | 用目标速度预种 arm_vel |
| `soft_freeze(dt)` | agent tick | 到点停手渐隐输出 |

### 4.5 `perception/ring_buffer.py` — RingBuffer

人/AI 位移分离的核心基础设施：

```
HumanMouseListener  ──→ add_event(dx, dy, is_ai=False)
MouseWorker (output) ──→ add_event(dx, dy, is_ai=True)
```

对外 API：

| 方法 | 语义 | 使用者 |
|---|---|---|
| `get_cursor_delta_sum(t0, t1)` | 仅人类 RawInput 和 | 人速/离合器 |
| `get_total_delta_sum(t0, t1)` | 人+AI 全量和 | WorldModel ego 追踪 |
| `get_ai_delta_sum(t0, t1)` | 仅 AI SendInput 和 | 交叉校验 |
| `get_pure_human_delta_sum(t0, t1)` | = get_cursor_delta_sum | 人速/离合器 |
| `get_wake_fused_delta(t0, t1)` | 人/AI 通道融合启发式 | 诊断备用 |

### 4.6 `output.py` — SystemMouse

Windows SendInput 封装：

- 预分配 `INPUT` 结构体内存（减少每帧分配）
- AI 签名 `dwExtraInfo = 0xFFC0FFEE` 写入每个 SendInput
- 旧版「到点停手」曾用 `_AIM_MOVE_BLOCK`；现恒不拦 aim 相对移动
- 每次 `mouse_xy` 同步写 RingBuffer (`is_ai=True`)
- 单例模式，全局 `gHub` 实例

### 4.7 `aim_strategies/valorant/strategy.py` — ValorantStrategy

像素 ↔ 鼠标 counts 的正逆变换（UE FOV 非线性）：

```
正向 (pixel → counts):
  angle = atan(pixel / focal_length)
  fov_corrected = angle × focal_length
  counts = fov_corrected × k_factor

逆向 (counts → pixel):
  fov_corrected = counts / k_factor
  pixel = tan(fov_corrected / focal_length) × focal_length
```

`bypass_strategy_mapping = True` 时：1 pixel ≡ 1 count（桌面/图片调试）。

---

## 5. 配置系统

### 5.1 文件结构

- `config.ini` — 用户可编辑的 INI 文件，所有可调参数
- `config.py` — `Config` 单例类，提供 `get/getint/getfloat/getbool/getstr/getlist`
- `utils/runtime_defaults.py` — 非炼丹常数（到点停手门限、Aimlab 色域），需改源码

### 5.2 关键配置段

| 段 | 用途 |
|---|---|
| `[General]` | 屏幕尺寸、capture 参数、开关、热键行为 |
| `[Inference]` | 推理后端、模型路径、置信度/IoU 阈值 |
| `[Kalman]` | Q/R 过程/观测噪声矩阵 |
| `[WorldModel]` | 延迟参数、目标粘滞、类别过滤、预测开关 |
| `[AimStrategy]` | 策略包、FOV 映射、k_factor |
| `[Controller]` | 控制器类型、阈值、CIPHER 全部参数 |
| `[Triggerbot]` | 扳机 FOV、置信阈值 |
| `[Debug]` | 日志等级、诊断开关 |

---

## 6. 管线阶段分解

单帧处理管线的完整时间线：

```
T₀    拍照 (t_capture)：DXCam 抓取 256×256 BGR 帧
│
T₁    推理完成 (t_done)：YOLO TensorRT 输出 (N,6) 检测框
│     → world_model.update_detections()
│     → world_model.frame_ready_event.set()
│
T₂    tick() 唤醒 (now)：
│     Step 1: 物理时钟更新 ego_pos_px += 人+AI 位移
│     Step 2: 回退计算 ego_at_capture
│     Step 3: 检测框过滤 → 粘滞排序 → Kalman predict + update
│     Step 4: 自适应延迟微调 (可选)
│     Step 5: 时空前视预测 p_predict
│     Step 6: chase_mode 判定 + 到点停手检测
│     Step 7: 像素→ counts 映射
│     Step 8: 人机动态离合器 → power_factor
│     Step 9: CIPHER controller.compute()
│     Step 10: Triggerbot 判定
│
T₃    1kHz MouseWorker 消费 controller.tick_mouse() → SendInput
```

---

## 7. 已知问题与 Bug

### 7.1 ⚠️ `config.py` — `getstr("0")` 返回 `"False"`

**位置**: `config.py` 第 51–68 行 `Config.get()` 方法

**问题**: `get()` 在 bool 检查中将 `"0"` 判为 `False`、`"1"` 判为 `True`，且此检查**优先于** int 转换。

**影响**: 当配置值恰好为 `0` 或 `1` 时：
- `getstr()` 返回 `"False"` / `"True"` 而非 `"0"` / `"1"`
- 当前 `aim_target_classes = 0` → `getstr()` 返回 `"False"` → `int("False")` 抛出异常 → 白名单过滤**静默失效**

**现状影响范围**:
- `aim_target_classes = 0` 在 `get_aim_target_class_set()` 中：`"False"` 无法解析为整数，返回 `None`（不过滤）。用户**意图是只跟踪 class 0**，实际结果是**所有类别均被跟踪**。
- 其他 `getstr` 调用当前未使用 `"0"/"1"` 值，暂未受影响。

**建议修复**:

```python
def get(self, section: str, key: str, default: Any = None) -> Any:
    val = self.parser.get(section, key).strip()
    if not val:
        return default
    # 尝试 int → float → bool → str
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            if val.lower() in ('true', 'false'):
                return val.lower() == 'true'
            if val.lower() in ('yes', 'no'):
                return val.lower() == 'yes'
            return val
```

### 7.2 ⚠️ `SimpleController` 缺少 `tick_mouse()` 实现

**位置**: `controllers/simple_controller.py`

**问题**: `BaseController.tick_mouse()` 是 `@abstractmethod`，但 `SimpleController` 未实现。若 `controller_type = simple`，实例化会抛 `TypeError`。

**建议**: 为 SimpleController 添加简单直通 `tick_mouse`。

### 7.3 ⚠️ `SimpleController.compute()` 签名不兼容

**签名差异**:

| agent.py 调用 (关键字) | SimpleController 签名 (位置) |
|---|---|
| `target_x=`, `target_y=`, `v_real=`, `a_real=`, `power_factor=`, `bbox_w=` | `intent_dx`, `intent_dy`, `dt` |

若切换到 `simple` 控制器，agent.tick() 的 `compute()` 调用会因参数名不匹配而失败。

### 7.4 ⚠️ `capture.py:110` — 裸 `except:` 吞全部异常

```python
try:
    self.camera.stop()
except:
    pass
```

DXCam stop 的各种异常（包括 `AccessViolation`）被静默吞掉，可能在退出时遮盖真实错误。

**建议**: 改为 `except Exception:` 并加日志。

### 7.5 ⚠️ `agent.py` 多处 `except Exception: pass`

| 行号 | 场景 | 风险 |
|---|---|---|
| 795 | `warm_start_from_velocity` | 唤醒预热失败静默 |
| 812 | `warm_start_from_velocity` (drift_resume) | 同上 |
| 966 | `notify_flick_end` / `warm_start_from_velocity` | 甩枪结束处理失败静默 |

这些 `pass` 在正常运行时无害，但会给调试带来困难。

### 7.6 ℹ️ `PROController` 未继承 `BaseController`

`PROController` 是独立实现，不依赖基类的 `pos/vel/backlog`。此设计**非 Bug**，但需注意 `isinstance(controller, BaseController)` 检查会返回 `False`。

### 7.7 ℹ️ `controllers/beifen.py` (备份文件)

存在文件 `controllers/beifen.py`（拼音"备份"），可能是旧版控制器的备份，建议移入 `archive/` 目录或删除。

---

## 附录 A: 快速启动检查清单

1. 确认 `config.ini` 中 `[General] screen_width/screen_height` 与显示器一致
2. 确认 `[Inference] model_path` 指向正确的 `.engine` 文件
3. 如用 Moonlight/串流，设置 `[WorldModel] moonlight_latency_ms` 为 25–45
4. 如用 VirtualHere/USB 延长，设置 `[WorldModel] virtualhere_latency_ms`
5. 按下 `P` 暂停/继续，`Alt+F1` 开关自瞄
6. 观察日志中 `Tick fps` 是否稳定（应接近推理帧率）
7. 关注 `AimLock DROP` / `Target re-acquired` 频率，频繁时可能需调整 `min_aim_conf`

## 附录 B: 关键缩写

| 缩写 | 全称 |
|---|---|
| KF | Kalman Filter |
| MPE | Motor Program Engine |
| AIC | Adaptive Impedance Control |
| SEC | Sub-Movement Error Correction |
| OUN | Ornstein-Uhlenbeck Noise |
| FBL | Fitts-Bio Loop |
| EMA | Exponential Moving Average |
| ROI | Region of Interest |
| FOV | Field of View |
| WM | WorldModel |
| OT | On-Target (到点停手) |
