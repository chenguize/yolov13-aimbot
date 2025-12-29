# Perception (Phase 3 Module)
## 感知采样与因果账本层

**（2025.12 · Phase 3 架构组件）**

> **核心定位**：本模块负责将物理世界的“光子”转化为数字世界的“张量”，并记录所有“因果历史”。  
> **关键词**：`t_cap` (时间锚点), `RingBuffer` (事实源), `FrameBus` (数据总线)。

---

## 一、模块职责 (Module Responsibility)

### 1.1 采样 (Capture)
* **固定视界**：不再跟随鼠标动态截屏，而是锁定屏幕中心（如 256x256）。
* **时间锚定**：在截图发生的纳秒级瞬间打上 `t_cap`，这是后续所有推理和预测的唯一时间基准。
* **零拷贝**：直接将 GPU 显存中的数据传递给 TensorRT（如支持），或以最快速度转为 Numpy。

### 1.2 传输 (Bus)
* **FrameBus**：极简的生产者-消费者通道。只保留“最新的一帧”，丢弃旧帧，保证系统永远处理最新鲜的视觉信息。

### 1.3 记忆 (RingBuffer)
* **因果账本**：记录过去 N 秒内所有的鼠标移动事件。
* **成分区分**：明确区分位移是来自 `Human` (手) 还是 `AI` (程序)。
* **时空对齐**：提供接口，允许 `WorldModel` 查询“从截图时刻 ($t_{cap}$) 到现在，准星到底漂移了多少像素”。

---

## 二、核心结构：RingBuffer (Structure A)

这是 Phase 3 最重要的数据结构之一。

```python
@dataclass
class InputEvent:
    timestamp: float  # 事件发生的时间
    dx: int           # X轴位移 (Mickeys)
    dy: int           # Y轴位移
    is_ai: bool       # True=AI控制, False=人类手搓