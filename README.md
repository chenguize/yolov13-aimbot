# YOLOv13 Aimbot + Triggerbot  
高阶异步实时辅助框架（2025年12月26日架构迭代）

当前工程成熟度：92 → 94/100（Phase 1 后期）  
目标成熟度：96–98/100（Phase 2 - 行为驱动 + 可仲裁 + 强时间一致性系统）

技术栈：YOLOv13 + TensorRT + BetterCam(CUDA) + GHUB + Python 3.11+

核心设计理念（持续演进中）
• 完全异步 + 状态驱动（非帧驱动）
• 单一事实源（World Model 作为状态仲裁者）
• 游戏专属逻辑与通用控制完全解耦（aim_strategies 只做映射，无预测）
• 全配置驱动 + 高危功能可一键切断
• 强调亚毫秒级时间一致性 + 控制反馈闭环（u_k 回传）
• 向行为意图仲裁 + 可观测回放演进

⚠️ 最高法律与道德声明  
本项目仅用于计算机视觉、实时控制系统、算法研究与本地技术验证。  
任何形式用于线上游戏均为严重违规行为，将导致永久封禁并可能承担法律后果。  
严禁在任何联网环境、生产环境或实际游戏中使用。

## 当前功能与工程状态（2025.12.26）

| 层级              | 模块/功能                              | 主要职责                                       | 当前实现度 | 关键 config 开关                        | 风险/稳定性等级 |
|-------------------|----------------------------------------|------------------------------------------------|------------|------------------------------------------|-----------------|
| 感知              | 画面采集 + 时间戳精确记录              | 256×256 鼠标中心捕获 + 采集时刻 mouse_pos 记录 | ★★★★★     | —                                        | —               |
| 推理              | YOLOv13 TensorRT 异步推理              | 高性能检测（Ultralytics 封装）                 | ★★★★      | model_path, conf_threshold               | —               |
| 状态中枢          | World Model（含控制反馈闭环）          | 坐标转换 / 跟踪 / 预测 / 自运动补偿（u_k 输入）| ★★★★      | ego_motion_compensation                  | ★★★             |
| 游戏策略          | Aim Strategies（仅映射）               | Valorant 专用鼠标移动映射 + 死区               | ★★★★      | sensitivity, deadzone_pixels             | —               |
| 控制算法          | Controllers + Humanize                 | 通用控制 + 拟人化管道（反应/噪声/疲劳/过冲/曲线）| ★★★       | controller_type, humanize_*              | ★★★             |
| 执行层            | Output GHUB                            | 鼠标移动/点击 + 微移动合并 + 死区 + 限频 + u_k 反馈 | ★★★★      | enable_micro_move_merge, deadzone        | ★★★★            |
| 行为控制          | Aimbot / Triggerbot                    | 功能级开关                                     | ★★★       | enable_aimbot, enable_triggerbot         | ★★★★ / ★★★★★   |
| 可观测性          | 状态快照 / 时间线回放                  | 全链路事件记录与复现                           | ★☆        | —（规划中）                              | —               |

## 项目结构（2025.12.26）
yolov13-aimbot/
├── main.py                       # 系统总控 + 主循环（已优化：屏幕中心参考 + P 键暂停）
├── config.py                     # 配置单例
├── config.ini                    # 所有开关与调参
│
├── perception/                   # 感知链
│   ├── capture.py                # BetterCam 256×256 采集 + 采集时刻 mouse_pos 环形缓冲
│   ├── bus.py                    # FrameBus + 鼠标历史环形缓冲
│
├── inference.py                  # YOLOv13 Ultralytics + TensorRT 异步推理（极简封装）
├── world_model.py                # 状态中枢 + Kalman + u_k 反馈 + 采集时刻坐标转换，多目标优先选择最近的
│
├── aim_strategies/               # 游戏专用映射（当前仅 Valorant，无预测）
│   ├── init.py
│   ├── valorant/strategy.py      # 高速比例映射 + 死区
│   └── factory.py                # 策略工厂
│
├── controllers/                  # 通用控制 + 拟人化管道
│   ├── init.py
│   ├── base_controller.py
│   ├── simple_controller.py
│   ├── pid_controller.py
│   ├── controller_factory.py
│   └── humanize/                 # 拟人化模块（全部可开关）
│       ├── init.py
│       ├── reaction_delay.py
│       ├── noise.py
│       ├── fatigue.py
│       ├── overshoot.py
│       └── curve.py
│
├── output_ghub.py                # GHUB 输出（异步 + 微移动合并 + 死区 + u_k 反馈）
├── utils/                        # 通用工具
│   ├── helpers.py                # letterbox、clamp、smoothstep 等
│   └── types.py                  # FrameInfo、Detection 等数据结构
│
├── models/
│   └── best256.engine
└── requirements.txt
text## 重要改动记录（同步至 2025.12.26）

1. **射击游戏特性适配**（2025.12.24）  
   - 鼠标位置恒等于屏幕中心（十字准星位置）  
   - 删除主循环中实时 `get_current_mouse_pos()`，统一使用固定 `screen_center`  
   - 瞄准 & Triggerbot 均以屏幕中心为参考点  
   - 理由：避免采集→推理延迟（15~35ms）导致的坐标误差  
   - 影响：瞄准更稳定、更准确，逻辑更简洁

2. **预测功能精简**（2025.12.26）  
   - 删除所有预测逻辑（速度外推、历史平均、衰减等）  
   - aim_strategies/valorant/strategy.py 只保留**鼠标移动映射**（比例 + 死区）  
   - 理由：用户明确要求“只要无畏契约的鼠标移动映射就行了，预测什么的不需要”

3. **快捷键支持**（2025.12.26）  
   - 新增全局热键：P 键 = 暂停/恢复整个项目  
   - 使用 keyboard 库实现（需 pip install keyboard 并管理员运行）

4. **输出层增强**（2025.12.26）  
   - output_ghub.py 增加异步队列 + 微移动合并 + 死区 + 频率限制  
   - 实际发送的 u_k 实时反馈给 world_model（ego-motion 闭环）

## 安全铁律（必须遵守）

1. 所有高危开关**默认关闭**，逐项开启且从小强度开始
2. **永不**同时开启 aimbot + triggerbot
3. **最有效的行为伪装组合**（优先级顺序）：
   1. 微移动合并 + 死区
   2. 采集时刻时间戳同步 + 控制反馈 ego-motion
   3. 反应延迟 + 随机性 + 疲劳累积
   4. 行为噪声 + 曲线形状 + 过冲回正
   5. 意图仲裁层（FSM/BT）+ 状态犹豫机制（Phase 2）

## 项目定位与演进宣言

**当前**：功能驱动的高质量异步实时系统（94分工程水准）  
**下一阶段（Phase 1.5）**：强时间一致性 + 控制反馈闭环（目标 95+）  
**最终目标（Phase 2）**：行为意图驱动 + 仲裁 + 全链路可观测回放（准工业级）

**再次郑重声明**  
**本项目仅限本地技术研究、算法验证与学术讨论。**  
**严禁用于任何线上游戏环境或商业用途。**