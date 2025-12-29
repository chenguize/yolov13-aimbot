# Output (Phase 3 Module)
## 硬件执行与闭环反馈层 (Win32 API 版)

**（2025.12 · Phase 3 架构组件）**

> **核心定位**：本模块是 **"数字指令"** 到 **"操作系统事件"** 的执行者。
> **关键机制**：使用 `SendInput` 注入带有 **数字水印 (Magic Signature)** 的事件，实现敌我识别。

---

## 一、模块职责 (Module Responsibility)

在 Phase 3 架构中，`output` 承担两项关键任务：

### 1.1 注入与水印 (Injection & Watermarking)
* **API**：使用 Windows `SendInput` API 进行鼠标模拟。
* **水印 (`dwExtraInfo`)**：在每一个 AI 生成的鼠标事件中，注入特定的 **Magic Number (0xFFC0FFEE)**。
    * **目的**：让系统的其他部分（如 Input Listener）能够识别出“这是 AI 动的，不是人动的”，从而避免数据污染。

### 1.2 闭环反馈 (Feedback Loop)
* **回写账本**：执行移动后，主动调用 `RingBuffer.add_event(dx, dy, is_ai=True)`。
* **原子性**：确保“物理层发送”与“逻辑层记录”在同一微秒级时间片内完成。

---

## 二、关键风险提示

### ⚠️ 关于 Input Listener (监听器) 的配合
由于 `SendInput` 模拟的信号会被系统的全局钩子（Hook）捕获，**你必须修改你的输入监听代码**（如 `pyWinhook` 或 `GetRawInputData` 监听线程）：

1.  监听器接收到鼠标移动事件。
2.  检查 `event.dwExtraInfo`。
3.  **如果等于 `0xFFC0FFEE`**：直接丢弃，**不要**记录到 `RingBuffer`（因为 Output 模块已经主动记录了 `is_ai=True`）。
4.  **如果不等于**：视为人类操作，记录到 `RingBuffer` (`is_ai=False`)。

**如果不做这一步过滤，RingBuffer 会记录双倍位移，导致 WorldModel 瞬间崩溃。**

---