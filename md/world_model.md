# World Model (Phase 4 Evolution)  
**世界状态仲裁与认知模拟中心**  
*(2025.12 · Phase 4 架构核心)*

**核心定位**：  
本模块是系统的“物理真相实验室”与“人类感知模拟器”。

**职责**：  
将原始视觉观察还原为物理真实的实体轨迹，并主动注入“感官延迟”与“认知预热”，使输出严格符合人类中枢神经系统的生理特性。

## 一、模块职责 (Module Responsibility)

在 Phase 4 架构中，World Model 已不再是单纯的数据清洗器，而是完整模拟人类从“看到”→“看清”→“大脑决策”的认知链路。

### 1.1 核心职能 (The Pipeline)

1. **因果位移对冲 (Causal Hedging)**  
   - **原理**：利用 RingBuffer 查账，剔除截图瞬时到当前时刻之间因鼠标移动导致的像素位移偏移量。  
   - **目的**：提取纯净的“敌人相对于游戏世界的位移”，作为后续所有预测的基础。

2. **状态估计 (Kalman Filtering)**  
   - 维护目标运动状态向量 $[x, y, v_x, v_y]^T$。  
   - 通过对冲后的坐标更新滤波器，平滑 YOLO 检测的噪声与跳变。

3. **感知滞后链 (Perception Lag Chain)** *[NEW]*  
   - **模拟逻辑**：建立环形队列，强制让输出信号落后于真实观测值 30–60ms（约 2–4 帧）。  
   - **拟人价值**：模拟视觉信号通过视神经传导至大脑皮层的生理延迟，杜绝准星产生“超感官”的瞬发反应。

4. **认知预热与置信度熔断** *[NEW]*  
   - **模拟逻辑**：引入 Confidence Warming 机制。目标初次出现时，置信度从 0.3 开始随帧数爬升至 1.0。  
   - **行为表现**：模拟人类“发现目标 → 确认敌我 → 锁定目标”的认知过程，导致移动初期呈现自然的“预热慢启动”。

5. **时延预测 (Latency Prediction)**  
   - **逻辑**：结合系统总时延 $T$ 与 PROController 提供的移动时间 $MT$（Fitts 计算）。  
   - **产出**：推演未来位置  
     $P_{predict} = P_{current} + V_{kalman} \times (T_{system} + T_{fitts\_mt})$

## 二、核心数学逻辑

### 2.1 认知级反应时计算

系统总反应延迟由物理延迟与模拟延迟共同构成：

$$
T_{response} = T_{sampling} + T_{inference} + T_{perception\_lag}
$$

- 物理采样 + 推理：约 15–40ms  
- 感知滞后链：固定注入 30–60ms  
- **最终效果**：整体反应时锁定在 110–150ms，处于人类职业电竞选手的巅峰极限区间

### 2.2 置信度对冲公式

最终下发给 Strategy 的位移意图将受到认知置信度的缩放：

$$
\vec{V}_{intent} = (P_{target} - P_{center}) \times \text{Confidence}
$$

- Confidence < 1.0 时 → 准星缓慢向目标靠拢（模拟确认过程）  
- Confidence = 1.0 时 → 释放全部动力进入爆发期（锁定完成）

## 三、输入与输出 (I/O)

### 输入 (Dependencies)

- Detection List：YOLO 原始检测框  
- RingBuffer：提供 `get_cursor_delta_sum(t_start, t_end)` 接口用于因果对冲  
- Fitts MT：来自 PROController 规划的预期移动时间

### 输出 (InferenceContext 填充)

World Model 实时修改 `InferenceContext` 对象：

- `context.p_predict`：经过感知滞后处理后的“视觉坐标”  
- `context.v_real`：目标的真实世界速度向量  
- `context.confidence`：当前目标的认知成熟度（0.3 → 1.0）  
- `context.is_valid`：仅当目标稳定追踪且置信度越过阈值时，才允许 Strategy 发起进攻

## 四、拟人化表现评估

- **启动阶段**：  
  由于“感知滞后”与“认知预热”，准星在目标出现的前几帧表现出明显的反应延迟与力量缓慢爬升。

- **追踪阶段**：  
  Kalman 滤波器结合因果对冲，确保在复杂位移（peek、闪现、变向）下轨迹依然丝滑且符合物理惯性。

- **鉴挂防御**：  
  这种“有层级的延迟”完美模拟人类生物特征，使得任何基于“反应时间过快”或“启动动量异常”的 AI 检测算法基本失效。

**一句话总结**：  
Phase 4 的 World Model 已从“数据处理器”进化为**人类认知链路的完整仿真器**，它不再追求“最快最准”，而是追求**“最像人、最难被机器分辨的真实”**。