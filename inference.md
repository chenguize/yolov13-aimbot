# Inference Layer (Phase 3 Module)
## 异步感知与目标提取层

**（2025.12 · Phase 3 架构组件）**

> **核心定位**：本模块是系统的 **"眼睛"**。  
> **职责**：从 TensorRT 引擎提取原始视觉数据，填充 **结构 B (Inference Context)** 的基础字段。

---

## 一、模块职责 (Module Responsibility)

在 Phase 3 架构中，`inference` 必须保持 **无状态 (Stateless)** 和 **无记忆 (Memoryless)**。

### 1.1 核心职能
1.  **模型加载**：加载 YOLOv13 TensorRT Engine (通过 Ultralytics 封装)。
2.  **异步推理**：从 `FrameBus` 获取最新图像，执行推理，不阻塞采集线程。
3.  **时序打标**：记录 `t_inference_done`，标志着“机器看清这张图”的时刻。
4.  **数据推送**：将 `(Detections, t_cap, t_done)` 打包推送到 `WorldModel`。

### 1.2 严格界限（不做所有事）
* ❌ **不读 RingBuffer**：推理层不知道鼠标动没动，只负责看图。
* ❌ **不做目标筛选**：不判断哪个是敌人，哪个是队友，全部扔给 WorldModel。
* ❌ **不做预测**：不计算提前量。

---

## 二、输入与输出 (I/O)

### 输入 (From FrameBus)
* `FrameInfo`: 包含 `frame` (图像) 和 `t_cap` (采集时间锚点)。

### 输出 (To WorldModel)
向 World Model 传递构建 **Inference Context** 所需的原材料：

| 参数 | 类型 | 说明 | Phase 3 对应字段 |
| :--- | :--- | :--- | :--- |
| `detections` | `List` | `[x1, y1, x2, y2, conf, cls]` 原始框 | `targets` |
| `frame_id` | `int` | 帧序列号 | - |
| `t_cap` | `float` | 截图时刻 (来自 FrameInfo) | `t_cap` |
| `t_done` | `float` | **[新增]** 推理完成时刻 | `t_inference_done` |

---