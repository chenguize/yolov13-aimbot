# YOLOv13 Aimbot + Triggerbot - 高阶异步工程版（2025年12月）

**基于 YOLOv13 + BetterCam(CUDA) + GHUB 的实时辅助框架**  
**核心理念**：完全异步、状态驱动、世界模型、独立高频控制环  
**所有高级功能均可通过 config.ini 开/关，方便调试、适配不同硬件和风险控制**

**⚠️ 郑重声明**  
本项目**仅供技术学习、算法研究、计算机视觉讨论与本地测试**。  
**任何形式用于线上游戏的行为均严重违反游戏服务条款，会导致永久封禁**，请勿尝试。

## 项目功能一览（2025.12）

| 功能模块              | 描述                                                                 | config 主要开关                     | 默认 | 风险等级 |
|-----------------------|----------------------------------------------------------------------|-------------------------------------|------|----------|
| **Aimbot（自瞄）**    | 检测到目标后平滑移动鼠标至目标（支持多种控制算法与拟人化）             | enable_aimbot                       | false| ★★★★     |
| **Triggerbot（自动扳机）** | 准星在敌人身上一定范围内时自动模拟鼠标左键点击                       | enable_triggerbot                   | false| ★★★★★    |
| **画面采集**          | 以当前鼠标位置为中心捕获 256×256 区域（与模型训练尺寸一致）           | —                                   | —    | —        |
| **YOLOv13 推理**      | TensorRT 异步推理（best256.engine）                                  | model_path, conf_threshold          | —    | —        |
| **世界模型**          | 目标跟踪、坐标转换、状态管理、可选多目标/卡尔曼预测                   | enable_multitrack, ekf_extrapolation| false| —        |
| **控制器**            | 多种控制算法（比例/PID/期望速度/Jerk限制）+ 拟人化管道                | controller_type, humanize_*         | simple| —        |
| **输出层**            | GHUB 鼠标移动 + 点击 + 微移动合并 + 死区过滤                         | enable_micro_move_merge, deadzone   | false| —        |

## 项目结构
yolov13-aimbot/
├── main.py                     # 系统入口、线程管理、监控、优雅退出
├── config.py                   # 全局配置单例（读取 config.ini）
├── config.ini                  # 所有功能开关与参数
│
├── perception/                 # 画面采集与分发层（高内聚）
│   ├── init.py
│   ├── capture.py              # BetterCam 256×256 鼠标中心捕获
│   ├── bus.py                  # FrameBus（最新帧 + 历史 + 鼠标位置）
│   └── frame_preprocessor.py   # 可选：亮度/对比度/去噪等预处理
│
├── inference.py                # YOLOv13 TensorRT 异步推理线程
├── world_model.py              # 目标跟踪、坐标转换、状态机、预测
│
├── controllers/                # 决策与执行引擎
│   ├── init.py
│   ├── base_controller.py
│   ├── simple_controller.py
│   ├── pid_controller.py
│   ├── controller_factory.py   # 根据配置创建控制器 + 拟人化管道
│   └── humanize/               # 拟人化处理（全部可开关）
│       ├── reaction_delay.py
│       ├── fatigue.py
│       ├── overshoot.py
│       ├── noise.py
│       └── curve.py
│
├── output_ghub.py              # GHUB 输出（移动 + 点击 + 队列 + 合并 + 死区）
├── utils/                      # 通用工具
│   ├── init.py
│   ├── helpers.py              # bbox 处理、坐标转换、时间工具等
│   └── types.py                # 常用数据结构（FrameInfo, Detection 等）
│
├── models/
│   └── best256.engine          # YOLOv13 导出模型（256×256 输入）
├── requirements.txt
└── README.md
text## 主要模块/文件夹 输入输出（逻辑单元视角）

| 模块/文件夹            | 主要职责                              | 主要输入来源                              | 主要输出提供给谁                          | 关键备注                              |
|------------------------|---------------------------------------|-------------------------------------------|-------------------------------------------|---------------------------------------|
| **perception/**        | 画面采集 + 预处理 + 状态广播           | 系统启动（无显式输入）                     | 最新 256×256 帧 + timestamp + 鼠标位置     | 高内聚，未来可扩展多源                |
| **inference.py**       | YOLOv13 TensorRT 异步推理              | perception 最新帧                          | 原始/过滤后的 detections                  | 永不阻塞，性能监控可选                |
| **world_model.py**     | 目标跟踪、坐标转换、状态管理、预测     | inference detections + 时间戳 + 鼠标位置   | 可信目标（屏幕绝对坐标）+ 状态 + 预测位置  | 多目标/卡尔曼/状态机均可开关          |
| **controllers/**       | 决策算法 + 拟人化处理                  | world_model 目标 + 当前鼠标位置 + dt       | 最终移动量（像素） + 是否触发扳机信号      | 多种算法 + 完整拟人化管道（可插拔）    |
| **output_ghub.py**     | 鼠标移动 & 点击执行层                  | controllers 移动量 + 扳机信号              | GHUB mouse_xy() + mouse_down/up()         | 微移动合并、死区、频率限制均可开关    |
| **main.py**            | 系统协调、线程生命周期、监控           | 无（启动所有）                             | 启动/停止所有线程，处理全局退出            | Supervisor 角色                       |

## 用户故事：遇到一个敌人靶标时，代码的典型流程（2025.12）

**场景**：玩家正在玩游戏，突然出现一个敌人（假设置信度足够）

1. **CaptureThread**（perception/capture.py）  
   → 每 ~2-3ms 抓取一次以**当前鼠标位置为中心**的 256×256 区域  
   → 通过 FrameBus 广播最新帧 + timestamp + 当前鼠标屏幕坐标

2. **InferenceThread**（inference.py）  
   → 拿到最新帧，送入 TensorRT engine (best256.engine)  
   → 执行推理 → 后处理 → 得到 detections 列表  
   → 传递给 world_model（包含原始坐标、置信度、类别）

3. **WorldModel**（world_model.py）  
   → 接收 detections  
   → 进行坐标转换：256×256 相对坐标 → 屏幕绝对坐标  
     （screen_x = mouse_x - 128 + det_center_x）  
   → （可选）多目标跟踪、卡尔曼预测、状态机过滤  
   → 输出：当前最佳目标（屏幕绝对坐标）+ 置信度 + 是否可见等状态

4. **主循环**（main.py）  
   → 读取 world_model 最新目标  
   → 如果 enable_aimbot == true：  
     → 调用 controller.compute() → 得到本次移动量 dx,dy  
     → 经过 humanize 管道（可选反应延迟、噪声、曲线等）  
     → 送入 output_ghub 发送移动指令

   → 如果 enable_triggerbot == true：  
     → 判断当前鼠标位置（屏幕中心）是否落在某个目标框内（一定 FOV 内）  
     → 满足 conf 阈值 → 等待随机延迟 → 发出点击指令（mouse_down → sleep → mouse_up）

5. **output_ghub.py**  
   → 接收移动指令（dx,dy）与点击指令  
   → （可选）微移动累积、死区过滤  
   → 通过 GHUB 接口发送 mouse_xy() / mouse_down() / mouse_up()

**完整链路时间线**（理想情况下）  
抓图 → 推理 (~8-15ms) → 世界模型处理 (~1ms) → 控制器计算 (~0.5ms) → 输出 (~2-4ms)  
→ 整个闭环延迟通常在 **15-35ms** 内（视硬件而定）

## 安全与开发建议（再次强调）

- **永远不要**同时开启 aimbot + triggerbot
- 所有高危行为（快速移动、无延迟点击）都应**先关闭**，逐项开启测试
- 微移动合并 + 死区 + 反应延迟 + 噪声 是目前最有效的降低检测概率手段
- 建议开发流程：先实现**纯 triggerbot** → 再加**慢速 aimbot** → 最后才考虑预测/多目标

**项目仅限本地研究使用，严禁用于任何线上环境。**