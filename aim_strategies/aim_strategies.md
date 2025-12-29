# Aim Strategies (Phase 3 Module)
## 几何映射与自校准层

**（2025.12 · Phase 3 架构组件）**

> **核心定位**：本模块是 **"像素世界"** 到 **"鼠标硬件世界"** 的汇率转换器。  
> **原则**：纯函数、无时间状态（除校准参数外）、确定性映射。

---

## 一、模块职责 (Module Responsibility)

在 Phase 3 架构中，`aim_strategies` 必须严格遵守单一职责。它不再负责“像不像人”或“什么时候动”，只负责“动多少”。

### 1.1 核心职能
1.  **几何映射**：将 `World Model` 输出的像素位移（Pixels）转换为鼠标硬件计数（Mickeys）。
2.  **自校准 (Self-Calibration)**：通过闭环反馈，动态修正“灵敏度/FOV”等配置参数的系统性偏差。

### 1.2 严格界限（不做所有事）
* ❌ **不做预测**：不计算提前量（这是 `World Model` 的工作）。
* ❌ **不做平滑**：不使用 PID 或曲线平滑（这是 `Controller` 的工作）。
* ❌ **不控制时间**：不包含 `sleep` 或冷却逻辑（这是 `Controller` 的工作）。

---

## 二、核心逻辑：自校准 (Self-Calibration)

为了解决游戏内灵敏度设置不准或鼠标 DPI 漂移的问题，本模块引入**闭环修正**。

### 2.1 映射公式
$$Counts = \Delta Pixel \times K_{factor}$$
其中 $K_{factor}$ 是核心参数，代表“每移动 1 像素需要发送多少鼠标计数”。

### 2.2 校准逻辑
系统会对比 **RingBuffer (历史 AI 操作)** 与 **World Model (实际画面反馈)**：

1.  **输入**：过去 N 帧 AI 发出的总指令 ($\sum Counts_{ai}$)
2.  **反馈**：对应时间段内画面产生的实际位移 ($\Delta Pixels_{real}$)
3.  **计算观测 K 值**：$K_{obs} = \frac{\sum Counts_{ai}}{\Delta Pixels_{real}}$
4.  **修正**：如果 $K_{obs}$ 长期偏离当前 $K_{factor}$，则进行微调更新。

---

## 三、接口定义

### 输入 (Input)
* `dx, dy` (float): 目标相对于准心的像素差 (经过 Kalman 预测后)。

### 输出 (Output)
* `Tuple[int, int]`: 理论上需要发送给鼠标的原始 X/Y 计数。

---