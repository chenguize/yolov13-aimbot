# YOLOv13 Aimbot + Triggerbot
## 高阶异步实时辅助框架
**（2025 年 12 月 · Phase 3.5 架构修正版）**

> 架构阶段：**Phase 3.5 – 仿生规划与物理闭环**
> 技术栈：**YOLOv13 + TensorRT + BetterCam + Python 3.12+ + Win32 API (水印注入)**

---

## 一、核心设计理念（更新）

- **完全异步 + 状态驱动**
- **单一事实源 (RingBuffer)**：基于时间戳的因果账本。
- **物理闭环 (Physical Loop)**：
  - **水印注入**：Output 层在指令中注入 `Magic Number`。
  - **回环过滤**：Input Listener 识别水印，确保 RingBuffer 精确区分“人手操作”与“AI操作”。
- **仿生运动规划**：
  - 从简单的 PID 升级为 **PRO Controller (Fitts + Min-Jerk)**。
  - 模拟人类肌肉协同，而非机械误差修正。

---

## 二、Phase 3.5 闭环执行流程（修正版）

1.  **采样 (Capture)**: 生成唯一时间锚点 `t_cap`。
2.  **感知 (Inference)**: 
    - 纯视觉提取，不读内存。
    - 输出 `t_done` 标记推理完成时刻。
3.  **世界状态仲裁 (World Model - Phase 4 Ready)**:
    - **因果对冲**：剔除 `is_ai=True` 的位移。
    - **决策门控**：应用感知滞后 (Perception Lag) 和认知预热。
    - **随机决策**：计算“脖子偏移”与“机会浪费”概率。
    - **预测**：输出 $v_{real}$ 和 $P_{predict}$ (可一键关闭)。
4.  **仿生规划 (PRO Controller)**:
    - 接收意图，基于 Fitts 定律规划 $MT$ (运动时间)。
    - 生成 Minimum Jerk (最小加加速度) 速度曲线。
    - 叠加 8-12Hz 生物震颤噪声。
5.  **执行与反馈 (Output - Win32)**:
    - 使用 `SendInput` 下发带有 `dwExtraInfo=0xFFC0FFEE` 的指令。
    - **原子回写**：立即写入 RingBuffer (`is_ai=True`)。

---

## 三、模块职责更新说明

### 1. world_model.py —— 物理真相与认知决策
**变更点**：从单纯的 Kalman 滤波升级为“物理+认知”双层模型。
* **物理层**：负责位移对冲和速度估计。
* **认知层 (WorldTarget)**：负责模拟人类的反应延迟、瞄准部位偏好（Neck Offset）和非理性的放弃射击（Waste Chance）。

### 2. controllers/pro_controller.py —— 生物力学规划器
**变更点**：**取代原 PID Controller**。
* **核心**：不再回答“误差是多少”，而是回答“人类如何把手移过去”。
* **特性**：
    * **Sensory-Motor Planning**：主动规划平滑轨迹。
    * **Planned Overshoot**：模拟甩枪时的自然过冲与回扣。

### 3. output.py —— 带水印的执行层
**变更点**：**取代 output_ghub.py** (为保证闭环逻辑)。
* **职责**：将逻辑计数转换为操作系统事件。
* **关键机制**：必须注入 **Magic Signature (0xFFC0FFEE)**，以便 Input Listener 能够识别并丢弃这些事件，防止污染 RingBuffer 的“人类操作记录”。

### 4. aim_strategies.py —— 几何映射与自校准
**变更点**：增加了自校准接口。
* **职责**：像素 -> Mickeys 的映射。
* **自校准**：通过对比“发出指令总和”与“实际画面位移”，动态修正 $K_{factor}$ (灵敏度系数)。

---

## 四、关键配置警告

### ⚠️ 关于 Input Listener 的强制修改
由于采用了 `Win32 API + Watermark` 方案，**必须**修改底层的鼠标监听代码（如 `pyWinhook`）：
```python
if event.dwExtraInfo == 0xFFC0FFEE:
    return  # 这是 AI 自己发的，直接忽略，不要记入 RingBuffer
else:
    RingBuffer.add(event.dx, event.dy, is_ai=False) # 这是人发的，必须记录

## 五、项目结构更新
yolov13-aimbot/
├── main.py
├── config.ini
├── config.py
├── inference.py
├── config.py
│
├── perception/
│   ├── capture.py
│   ├── bus.py
│   └── ring_buffer.py          # 核心：支持水印过滤
│
├── world_model.py              # Phase 4 Practical: 物理+决策
│
├── aim_strategies/
│   └── strategy.py             # 含 Self-Calibration
│
├── controllers/
│   ├── base_controller.py
│   ├── simple_controller.py       # 一帧拉枪用来调试
│   └── pro_controller.py       # 默认推荐: Bio-Planning
│
└── output.py                   # Win32 + Watermark (控制鼠标)

