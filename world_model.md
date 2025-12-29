# World Model (Phase 3 Module)
## 世界状态仲裁与因果预测中心

**（2025.12 · Phase 3 架构核心）**

> **核心定位**：本模块是系统的“物理真相实验室”。  
> **职责**：将带有延迟、模糊、受鼠标移动干扰的“原始观察”，清洗还原为单一、物理真实的“实体轨迹”。

---

## 一、模块职责 (Module Responsibility)

在 Phase 3 架构中，World Model 是连接 **感知 (Inference)** 与 **决策 (Strategy)** 的唯一桥梁。

### 1.1 核心职能 (The Pipeline)
1.  **目标仲裁 (Arbitration)**：
    * 从 YOLO 输出的多个目标中，根据“距离优先 + 轨迹粘滞”原则选定唯一目标 (`selected_id`)。
    * 处理目标丢失与切换（ID Switch）逻辑，防止 Kalman 状态污染。
2.  **因果位移对冲 (Causal Hedging)**：
    *     * **问题**：摄像头截图时 ($t_{cap}$) 到现在推理完成，鼠标已经发生了移动。且上一帧到这一帧之间，鼠标也在移动。单纯的像素差 $\Delta P$ 混杂了“敌人的移动”和“准心的移动”。
    * **解法**：查询 `RingBuffer`，计算两个时间锚点之间的鼠标总位移，将其从画面像素变化中剔除，提取纯净的“敌人世界坐标位移”。
3.  **状态估计 (Kalman Filtering)**：
    * 维护目标的运动状态向量 $[x, y, v_x, v_y]^T$。
    * 利用对冲后的真实位移更新滤波器，平滑 YOLO 的抖动。
4.  **时延预测 (Latency Prediction)**：
    * 计算系统总时延：$T_{total} = (T_{now} - T_{cap}) + T_{input\_lag}$。
    * 推演未来位置：$P_{predict} = P_{current} + V_{kalman} \times T_{total}$。

---

## 二、核心数学逻辑

### 2.1 真实位移公式
$$\Delta P_{world} = (P_{now} - P_{last}) + \text{Map}(\sum_{t_{last}}^{t_{cap}} \Delta \text{Mickey})$$
* 若鼠标向右移动，画面整体向左平移，导致 $P_{now}$ 减小。
* 因此，必须将鼠标造成的位移“加回来”，才能得到物体在游戏世界中的真实移动。

### 2.2 状态有效性 (Validity Rules)
* **Init**：目标初次出现，`is_valid = False`（Kalman 需要至少两帧初始化）。
* **Tracking**：ID 匹配且速度稳定，`is_valid = True`。
* **Switch**：ID 发生突变，立即重置 Kalman，`is_valid = False`。
* **Coast**：目标短暂遮挡（<3帧），维持预测，线性外推。

---

## 三、输入与输出 (I/O)

### 输入 (Dependencies)
* **FrameInfo**: 包含 `t_cap` (时间锚点) 和 `bbox` (检测框)。
* **RingBuffer**: 提供 `get_cursor_delta(t_start, t_end)` 接口。

### 输出 (InferenceContext 填充)
World Model 不直接返回数据，而是就地修改传递进来的 `InferenceContext` 对象：
* `context.selected_id`: 锁定的目标 ID。
* `context.v_real`: 目标的真实屏幕速度 (px/s)。
* `context.p_predict`: 经过时延补偿后的最终击打点。
* `context.is_valid`: 本帧数据是否可信（Strategy 据此决定是否瞄准）。

---