# YOLOv13 Aimbot + Triggerbot (Phase 3.5)
**高阶异步实时辅助框架 · 仿生规划与物理闭环版**  
（2025 年 12 月 · Phase 3.5 架构定稿）

当前架构阶段：**Phase 3.5 – 仿生规划与物理闭环 (Bio-Planning & Physical Loop)**

核心技术栈：**YOLOv13 + TensorRT + BetterCam + Python 3.12+ + Win32 API (Magic Watermark)**

## 一、最高法律与道德声明（必须阅读）

⚠️ **重要声明**  
本项目代码与文档仅用于以下用途：  
- 计算机视觉与运动控制算法研究  
- 实时系统的异步架构设计实验  
- 人机交互中的状态估计与卡尔曼滤波验证  
- 本地、离线、非联网环境下的技术测试  

**严禁**将本项目用于任何线上游戏、对抗性网络服务或商业用途。  
任何违规使用行为都可能导致永久封禁、账号清零及法律风险，责任由使用者自行承担。

## 二、核心设计理念 (Core Philosophy)

本项目已从早期的“PID 控制”进化为基于状态估计与仿生规划的高阶系统：

- **完全异步 + 状态驱动**：解耦帧率与控制频率，系统不再被动响应每一帧图像，而是基于时间戳驱动。
- **单一事实源 (RingBuffer)**：维护一个基于时间的“因果账本”，允许系统在处理 20ms 前的旧图像时，精确回溯并计算出目标的真实物理轨迹。
- **物理闭环 (Physical Loop via Watermark)**：  
  - 注入：Output 层在下发指令时注入 0xFFC0FFEE 水印。  
  - 过滤：Input Listener 识别水印，确保 RingBuffer 能精准区分“AI 操作”与“人类操作”，防止自激振荡。
- **仿生运动规划 (Sensory-Motor Planning)**：抛弃机械的 PID 修正，采用 Fitts 定律 与 Minimum Jerk（最小加加速度） 算法，规划出符合人类生物力学特征的平滑曲线。
- **认知决策模型 (Cognitive Decision)**：引入“感知滞后”、“机会浪费”和“拟人化偏移”，模拟人类的有限理性与生理缺陷。

## 三、系统架构与模块职责

### 3.1 目录结构 (Project Structure)
yolov13-aimbot/
├── main.py                     # 系统总控与线程调度
├── config.py / config.ini      # 全局配置（默认全熔断）
│
├── perception/
│   ├── capture.py              # 屏幕采集 + 生成 t_cap
│   ├── bus.py                  # 帧广播通道
│   └── ring_buffer.py          # 因果账本 (含水印过滤逻辑)
│
├── inference.py                # YOLOv13 TensorRT 异步推理
│
├── world_model.py              # [核心] 物理建模 + 认知决策 (Phase 4)
│
├── aim_strategies/
│   └── strategy.py             # 几何映射 + 自校准 (Self-Calibration)
│
├── controllers/
│   ├── base_controller.py
│   ├── simple_controller.py    # 调试用：无延迟直通
│   └── pro_controller.py       # [核心] 仿生规划器 (Fitts + Min-Jerk)
│
├── output.py                   # [核心] Win32 API + 水印注入 (取代 GHUB)
│
└── utils/
└── types.py                # 数据结构定义


### 3.2 模块详细职责与约束

1. **Perception Layer (感知层)**  
   - **capture.py (采样)**  
     职责：生成唯一的时间锚点 t_cap。  
     约束：只回答“这张图是什么时候截取的”，不关心后续处理。  
   - **inference.py (推理)**  
     职责：无状态地提取视觉特征，输出 (Detections, t_cap, t_done)。  
     约束：不读取内存，不读 RingBuffer，不知道鼠标动没动，只负责“看”。

2. **World Model (世界模型 - Phase 4)**  
   定位：系统的“大脑”。负责将视觉信号还原为物理事实，并注入认知特征。  
   - **物理层 (The Observer)**：  
     - 因果对冲：回溯 RingBuffer，剔除 is_ai=True 的位移，还原目标真实世界速度 (v_real)。  
     - Kalman 滤波：消除检测噪声，估计目标的状态向量。  
   - **认知层 (The Decider - WorldTarget)**：  
     - 感知门控：模拟 40-60ms 的生理反应延迟。  
     - 拟人化偏移：以 40% 概率将瞄准点下移至脖子/上胸 (Neck Offset)。  
     - 机会浪费：强制维持 5-15% 的“可打但不打”概率，模拟人类犹豫。  
     - 预测：输出 P_predict（可一键关闭退化为平滑追踪）。

3. **Aim Strategies (策略层)**  
   定位：像素世界到硬件世界的“汇率转换器”。  
   - 几何映射：将像素距离转换为鼠标计数 (Mickeys)。  
   - 自校准 (Self-Calibration)：  
     逻辑：对比“发出的指令总和”与“画面实际产生的位移”。  
     作用：自动修正 K_factor (灵敏度系数)，解决游戏内灵敏度设置偏差。

4. **Controllers (控制层 - PRO 版)**  
   定位：从“意图”到“动作”的规划师。取代旧版 PID。  
   - **pro_controller.py (仿生规划器)**：  
     - Fitts Law：根据目标距离 D 和目标大小 W，计算符合人类极限的运动时间 MT。  
     - Minimum Jerk：生成钟形速度曲线，彻底消除机器人的“匀速”或“线性减速”特征。  
     - 生物震颤：叠加 8-12Hz 的低频噪声，模拟肌肉紧张时的微颤。  
   - **simple_controller.py**：仅用于调试延迟和校准 K 值。

5. **Output (执行层 - Win32 水印版)**  
   定位：带敌我识别的硬件接口。取代旧版 output_ghub。  
   - 水印注入：使用 SendInput 下发指令时，在 dwExtraInfo 字段注入 Magic Number (0xFFC0FFEE)。  
   - 原子回写：指令发送瞬间，立即向 RingBuffer 写入记录 (is_ai=True)。

## 四、闭环执行流程 (The Closed Loop)

1. **采样 (Capture)**：获取图像，打上微秒级时间戳 t_cap。  
2. **感知 (Inference)**：YOLO 解算目标，输出 t_done。  
3. **世界仲裁 (World Model)**：读取 RingBuffer，扣除 t_cap 至今的所有 AI 移动（对冲）。Kalman 更新 v_real。WorldTarget 判定是否满足“感知延迟”门控，并叠加“脖子偏移”。输出最终预测意图 P_final。  
4. **仿生规划 (PRO Controller)**：基于 P_final 规划未来 N 毫秒的平滑轨迹。计算当前微小的步进量 u_k。  
5. **执行 (Output)**：下发 SendInput(dx, dy, dwExtraInfo=0xFFC0FFEE)。  
6. **反馈过滤 (Input Listener)**：  
   关键步骤：底层钩子检测到 0xFFC0FFEE，直接丢弃，不记入 RingBuffer 的 is_ai=False 通道（防止数据污染）。

## 五、关键工程警告 (Engineering Hazards)

⚠️ **1. Input Listener 必须修改 (CRITICAL)**  
由于采用了 Win32 水印方案，你必须修改底层的鼠标监听代码（如 pyWinhook 或 GetRawInputData）：

```python
# 在 Input Listener 线程中：
if event.dwExtraInfo == 0xFFC0FFEE:
    # 识别到这是 Output 模块发出的 AI 指令
    # 直接忽略，不要记录到 RingBuffer
    pass
else:
    # 这是真实的人类鼠标操作
    # 记录到 RingBuffer，标记 is_ai=False
    RingBuffer.add(event.dx, event.dy, is_ai=False)

六、参数调优指南
模块,参数,推荐值,说明
World,enable_prediction,True,是否开启延迟预测，新手建议开启
World,perception_lag_ms,40-55,模拟人眼看到目标的延迟
World,neck_offset_prob,0.40,40% 概率打脖子，降低爆头率以防封
Controller,fitts_a,0.10,反应时基数，越小越快
Strategy,k_factor,Auto,建议开启 self_calibration 自动计算
