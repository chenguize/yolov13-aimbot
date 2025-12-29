# YOLOv13 Aimbot + Triggerbot
## 高阶异步实时辅助框架
**（2025 年 12 月 · Phase 3 架构定稿版）**
> 架构阶段：**Phase 3 – 世界模型与意图仲裁（已进入）**  
> 技术栈：**YOLOv13 + TensorRT + BetterCam (CUDA) + GHUB + Python 3.12+**

---

## 一、最高法律与道德声明（必须阅读）

⚠️ **重要声明**  
本项目仅用于以下用途：  
- 计算机视觉算法研究  
- 实时控制系统与异步架构设计  
- 世界建模、状态估计与预测控制的工程验证  
- 本地、离线、非联网环境下的技术实验  

**严禁**将本项目用于任何线上游戏、对抗性网络服务或商业用途。  
任何违规使用行为都可能导致**永久封禁、账号清零及法律风险**，责任自负。

---

## 二、核心设计理念（Phase 3 对齐）

- **完全异步 + 状态驱动（非帧驱动）**  
- **单一事实源（World Model 作为状态仲裁者）**  
- **时间可逆的因果建模**  
  - RingBuffer + 时间锚点 `t_cap`  
  - 允许系统在处理“过时图像”时回溯真实物理轨迹  
- **强时间一致性**  
  - 亚毫秒级时间戳  
  - 感知 → 建模 → 控制全链路对齐  
- **控制反馈闭环**  
  - 所有控制量 `u_k` 必须回写系统账本  
- **策略 / 控制 / 执行严格解耦**  
  - Strategy：只做几何与规则映射  
  - Controller：只负责控制与节流  
- **全配置驱动**  
  - 高危功能全部默认关闭，可一键熔断  
- **向“意图仲裁 + 全链路可回放”演进**

---

## 三、三阶段架构规划（已修正）

| 阶段                          | 核心目标               | 关键特性                                    | 当前状态     | 里程碑                  |
|-------------------------------|------------------------|---------------------------------------------|--------------|-------------------------|
| **Phase 1 – 纯粹执行**        | 极简、极速、极准       | 几何映射 + 死区                             | ✅ 已完成     | 固定中心 256×256 捕获   |
| **Phase 2 – 行为扰动**        | 打破机械轨迹           | 延迟 / 噪声 / 过冲                          | ⚠️ 可选开启   | humanize 管道           |
| **Phase 3 – 世界模型与意图仲裁** | 从“指令”到“意图”    | 因果账本 + Kalman + 延迟预测 + 控制闭环     | ✅ 已进入     | 真实物理闭环            |

> ⚠️ 说明  
> FSM / 行为树 / 犹豫机制属于 **Phase 3 的上层策略表达**，  
> **不再是 Phase 3 的定义条件本身**。

---

## 四、Phase 3 核心系统抽象

### 4.1 结构 A：RingBuffer Entry（因果账本）
**维护线程**：Input / Controller

```text
timestamp_us : 写入时的微秒级时间戳
mickey_dx/dy : 原始鼠标增量
is_ai        : 是否由 Controller 下发（True = AI 行为）
作用：
系统唯一的因果事实源
区分：人为输入、AI 控制、敌人导致的画面变化
支持按时间戳回溯与位移对冲
4.2 结构 B：Inference Context（单帧任务单）
维护线程：Inference / World Model
text复制t_cap             : 截图瞬间时间锚点
t_inference_done  : 推理完成时间
targets           : YOLO 输出目标列表
selected_id       : WorldTarget 选中目标
v_real            : Kalman 输出的真实速度
p_predict         : 延迟补偿后的预测终点
is_valid          : 本帧是否允许执行控制
原则：
一帧 = 一个完整的、可回溯、可解释的决策单元

五、Phase 3 五阶段闭环执行流程

第一阶段：采样（Capture Thread）
固定中心 256×256 截图
立即生成 Inference Context
写入 t_cap = getCurrentTimeUs()

第二阶段：推理与位移回溯（Inference Thread）
YOLO 推理填充 targets
访问 RingBuffer：回溯 [t_last_cap, t_cap] 之间的鼠标位移
剔除 is_ai = True 的控制量
构造纯净位移：
$\Delta P_{real} = (P_{now} - P_{last}) + Map(\sum \Delta Mickey)$

第三阶段：世界状态仲裁（World Model）
WorldTarget 筛选目标
若目标未切换：使用 $\Delta P_{real}$ 更新对应 Kalman
输出平滑后的真实速度 v_real
若目标切换：本帧 is_valid = False，给 Kalman 一帧重新收敛

第四阶段：延迟感知预测（Controller Preprocess）
计算画面年龄：$age = now - t_{cap}$
计算总预测时间：$T_{future} = age + system_{latency}$
预测终点：$p_{predict} = P_{now} + v_{real} \cdot T_{future}$

第五阶段：执行与反馈（Controller → Output）
Controller 下发控制量 $u_k$
立即回写 RingBuffer，is_ai = True
防止下一帧将自身行为误判为外界扰动
→ 闭环完成



六、项目结构（2025.12 · Phase 3）
text复制yolov13-aimbot/
├── main.py                     # 系统总控（P 键暂停）
├── config.py / config.ini      # 全局配置（默认全关闭）
│
├── perception/
│   ├── capture.py              # BetterCam 采集 + t_cap
|   ├── bus.py                  # 帧广播（FrameInfo）
|   └── ring_buffer.py          # 因果账本（唯一事实源）
│
├── inference.py                # YOLOv13 TensorRT 异步推理
│
├── world_model.py              # 世界状态中枢（Phase 3 核心）
│
├── aim_strategies/
│   └── valorant/strategy.py    # 纯几何映射（无预测）
│
├── controllers/
│   ├── base_controller.py
│   ├── simple_controller.py
│   ├── pid_controller.py
│   └── advanced_controller.py
│
├── controllers/humanize/       # 拟人化（默认全关）
│
├── output_ghub.py              # 执行层 + u_k 回写
│
├── utils/
│   └── types.py                # FrameInfo / Detection / Context
│
└── models/
    └── best256.engine

七、模块职责总览（Phase 3 对齐）

| 模块             | 职责   | 关键说明            |
| -------------- | ---- | --------------- |
| capture.py     | 采样   | 生成唯一时间锚点        |
| bus.py         | 配合帧采样 | 帧广播（FrameInfo） |
| ring_buffer.py | 因果账本   | 结构 A 存储与时间回溯查询       |
| inference.py   | 感知   | 填充 结构 B 的原始检测数据           |
| world_model.py | 状态仲裁 | Kalman + 目标筛选   |
| strategy.py    | 映射   | 像素 → 计数         |
| controller     | 控制   | 执行 + 节流         |
| output_ghub    | 执行   | u_k 回写          |

## 模块职责关键说明（Phase 3 设计约束）

### capture.py —— 采样与时间锚点生成

**职责定位：**
capture.py 是系统中唯一允许“感知真实世界时间起点”的模块。

**核心职责：**
- 执行固定中心区域的屏幕采样（如 256×256）
- 在截图瞬间生成唯一时间锚点 `t_cap`
- 将图像数据与 `t_cap` 绑定为不可拆分的 FrameInfo

**设计约束：**
- `t_cap` 必须在截图完成的同一时刻生成
- 严禁在推理、控制等后续阶段补打时间戳
- capture 不做任何推理、不做任何预测、不关心控制结果

> capture.py 只回答一个问题：  
> **“这张画面，真实发生在什么时候？”**

---

### bus.py（FrameBus）—— 帧通道而非因果来源

**职责定位：**
bus.py 是“当前帧”的广播通道，而不是历史事实存储。

**核心职责：**
- 接收 capture.py 生成的 FrameInfo
- 向 inference / world_model 提供“最新可用帧”
- 不保留跨帧的因果语义

**设计约束：**
- FrameBus 中的数据只代表“现在看到的画面”
- 禁止在 bus 中存储或推断历史运动
- 禁止将 bus 作为世界状态或控制依据的来源

> FrameBus 传递的是**观察结果**，不是**世界事实**。

---

### RingBuffer（因果账本）—— 真实世界事实源

**职责定位：**
RingBuffer 是系统唯一可信的**因果账本**，记录一切真实发生过的输入与控制行为。

**核心职责：**
- 记录所有鼠标位移（dx / dy）
- 精确标记位移来源（`is_ai`）
- 支持按时间戳区间回溯历史位移
- 为 world_model 提供“可对冲”的真实位移数据

**设计约束：**
- RingBuffer 只允许追加写入，不允许修改历史
- world_model 不得直接信任画面跳变，必须通过 RingBuffer 解释
- controller / output 层写入 RingBuffer 是**合法且必须的**

> RingBuffer 不是“日志”，  
> 它是系统对“现实世界发生了什么”的唯一共识。

---

### inference.py —— 感知，不参与世界解释

**职责定位：**
inference.py 只负责“看到什么”，不负责“为什么会这样”。

**核心职责：**
- 从 FrameBus 获取最新 FrameInfo
- 执行 YOLOv13 推理
- 输出目标检测结果（bbox / conf / cls）

**设计约束：**
- inference 不读取 RingBuffer
- inference 不关心鼠标是否移动
- inference 不参与目标仲裁、不参与预测

> inference 只回答一个问题：  
> **“这张图里，有哪些目标？”**

---
### ### world_model.py —— 世界状态仲裁中心 (Phase 3 核心)

**职责定位**：  
系统的“物理真相实验室”。负责将带有延迟和干扰的原始观察，清洗还原为单一、干净、物理真实的实体轨迹。可以开关是否预测。

#### **1. 核心职责 (顺序执行)**

* **目标仲裁与锁定**：根据“距离中心 + 轨迹平稳度”选定唯一 `selected_id`。若 ID 切换，立即标记 `is_valid = False` 并重置 Kalman。
* **因果位移对冲**：回溯 `RingBuffer`，剔除 `is_ai == True` 的增量。计算目标在世界坐标系下的纯净位移：
    $$\Delta P_{real} = (P_{now} - P_{last\_frame}) + \Delta P_{human\_px}$$
* **线性建模 (Kalman)**：喂入 $\Delta P_{real}$，维护约 10 帧的滑动窗口，平滑输出敌人的真实物理速度 $v_{real}$。
* **时延感知预测**：
    * 画面过期时间：$age = Now - t_{cap}$
    * 总预测时间：$T_{future} = age + system\_latency$
    * 最终预瞄点：$P_{predict} = P_{now} + v_{real} \times T_{future}$

#### **2. 设计铁律**

1.  **无对冲，不更新**：Kalman 输入必须经过 `RingBuffer` 对冲，严禁直接使用原始像素差。
2.  **切换即重置**：ID 切换首帧禁止预测，Kalman 状态空间立即重置。
3.  **稳定性熔断**：若 $v_{real}$ 连续 3 帧剧烈抖动，判定数据不可信，立即熔断 `is_valid`。

**一句话总结**：  
通过**因果对冲**与**时延补偿**，将“杂乱观察”清洗成“物理意图”。
### strategy.py —— 游戏规则映射层

**职责定位：**
strategy.py 是“游戏语义 → 控制意图”的映射层，而非控制器。

**核心职责：**
- 将目标相对位置映射为游戏内几何量
- 完成像素 → 角度 → 鼠标计数的确定性映射
- 不做平滑、不做预测、不引入时间状态

**设计约束：**
- strategy 必须是纯函数或弱状态函数
- 严禁在 strategy 中引入 Kalman 或历史依赖
- 不允许直接访问 RingBuffer
关于 Strategy 的鲁棒性：目前的 Strategy 定义为“纯几何映射”。建议在 Phase 3 中增加一个自校准逻辑。逻辑：对比 RingBuffer 记录的 is_ai=True 的总位移与下一帧画面中实际产生的位移差。如果长期存在 5% 的系统性偏差（例如游戏内开启了灵敏度缩放），Strategy 应该能自动修正那个像素映射系数 $K$。
> strategy 只负责回答：  
> **“如果我要指向这个目标，规则上该怎么动？”**

---

### controller —— 控制与节流执行层

**职责定位：**
controller 是“意图”到“可执行控制量”的转换器。

**核心职责：**
- 接收 world_model 给出的预测终点或控制意图
- 执行 PID / 比例 / 限幅等控制算法
- 进行拟人化相关可开关操作
- 决定是否实际下发控制

**设计约束：**
- controller 不做目标选择
- controller 不修改世界状态
- 所有输出都必须可回写、可追溯

> controller 回答的是：  
> **“现在，该不该动？动多少？”**

---

output_ghub.py —— 执行与闭环反馈

职责定位：
output_ghub.py 是系统中 AI 控制行为 与 物理输入设备 之间的唯一执行接口。

核心职责：

异步下发鼠标移动 / 点击等 AI 控制行为

合并微小位移，进行硬件级节流与安全限制

将实际已执行的 AI 控制量 u_k 写回 RingBuffer

写入时标记 is_ai = True，供 world_model 在下一帧进行位移对冲

设计约束（Phase 3 严格版）：

output_ghub 只负责 AI 行为的执行与记录

仅当控制量真实作用于物理设备后，才允许写入 RingBuffer

output_ghub 不监听、不记录人为鼠标输入

RingBuffer 的其他合法写入来源仅包括：

系统级输入监听（is_ai = False）

RingBuffer 记录的是世界中真实发生过的输入事实，
而不是“系统计划做什么”。

---
main.py —— 系统总控与线程协调

职责定位：
main.py 是整个 Phase 3 系统的入口，负责管理异步线程和全局状态。

核心职责：

启动各核心线程：

capture（屏幕采集 + 时间锚点）

inference（YOLO 推理 + FrameInfo 读取）

world_model（目标筛选 + Kalman + 预测）

controller（意图转换 + 控制输出）

管理 FrameBus 与 RingBuffer 的生命周期

统一读取全局配置（config.py / config.ini）

响应用户控制事件（如 P 键暂停 / 恢复）

提供系统级异常捕获与安全关闭机制

可选：触发 Trace Replay / 全链路回放

设计约束：

main.py 本身不参与感知、推理、映射或控制算法

保证线程启动顺序、资源安全和时间一致性

仅作为系统调度与入口，不生成任何业务决策或控制量

main.py 回答的是：
“系统如何组织、启动、协调、保证闭环顺利运行？”


八、系统定位与演进宣言
当前：
世界模型已成型的异步实时控制系统
决策追踪回放 (Trace Replay System) [可开关]
逻辑：将 结构 B 序列化导出。

功能：通过回放工具分析“为何没打中”。可一眼看出是由于 Strategy 映射误差，还是 RingBuffer 位移对冲失败。
下一步：
WorldTarget 引入稳定度 / 置信度仲裁
最终目标：
行为意图驱动 + 全链路可观测回放
（准工业级实时系统）

九、最终声明（再次强调）
本项目仅用于
算法研究 / 系统架构设计 / 学术讨论
严禁用于任何线上或对抗性环境。
