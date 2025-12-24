# YOLOv13 Aimbot + Triggerbot  
高阶异步实时辅助框架（2025年12月26日架构迭代）

当前工程成熟度：92 → 94/100（Phase 1 后期）  
目标成熟度：96–98/100（Phase 2 - 行为驱动 + 可仲裁 + 强时间一致性系统）

技术栈：YOLOv13 + TensorRT + BetterCam(CUDA) + GHUB + Python 3.11+

核心设计理念（持续演进中）
• 完全异步 + 状态驱动（非帧驱动）
• 单一事实源（World Model 作为状态仲裁者）
• 游戏专属逻辑与通用控制完全解耦（aim_strategies）
• 全配置驱动 + 高危功能可一键切断
• 强调亚毫秒级时间一致性 + 控制反馈闭环
• 向行为意图仲裁 + 可观测回放演进

⚠️ 最高法律与道德声明  
本项目仅用于计算机视觉、实时控制系统、算法研究与本地技术验证。  
任何形式用于线上游戏均为严重违规行为，将导致永久封禁并可能承担法律后果。  
严禁在任何联网环境、生产环境或实际游戏中使用。

## 当前功能与工程状态（2025.12.26）

| 层级              | 模块/功能                              | 主要职责                                       | 当前实现度 | 关键 config 开关                        | 风险/稳定性等级 |
|-------------------|----------------------------------------|------------------------------------------------|------------|------------------------------------------|-----------------|
| 感知              | 画面采集 + 时间戳精确记录              | 256×256 鼠标中心捕获 + 采集时刻 mouse_pos 记录 | ★★★★★     | —                                        | —               |
| 推理              | YOLOv13 TensorRT 异步推理              | 高性能检测                                     | ★★★★      | model_path, conf_threshold               | —               |
| 状态中枢          | World Model（含控制反馈闭环）          | 坐标转换 / 跟踪 / 预测 / 自运动补偿（u_k输入） | ★★★☆      | enable_multitrack, ekf_extrapolation, ego_motion_compensation | ★★★             |
| 游戏策略          | Aim Strategies                         | 游戏专用映射 + 预测（Valorant 风格自适应衰减） | ★★★★      | current_game, strategy_package           | —               |
| 控制算法          | Controllers + Humanize                 | 通用控制 + 拟人化（延迟/噪声/曲线/疲劳）       | ★★★       | controller_type, humanize_*              | ★★★             |
| 执行层            | Output GHUB                            | 鼠标移动/点击 + 微移动合并 + 死区 + 限频       | ★★★★      | enable_micro_move_merge, deadzone        | ★★★★            |
| 行为控制（Phase1）| Aimbot / Triggerbot                    | 功能级开关                                     | ★★★       | enable_aimbot, enable_triggerbot         | ★★★★ / ★★★★★   |
| 行为控制（Phase2规划）| Intention Arbiter + FSM/Behavior Tree | 统一意图仲裁（AIM_TRACK / HOLD / FIRE / RELEASE）| ☆         | —（规划中）                              | —               |
| 可观测性          | 状态快照 / 时间线回放                  | 全链路事件记录与复现                           | ★☆        | —（规划中）                              | —               |

## 项目结构（2025.12.26）
yolov13-aimbot/
├── main.py
│   职责：系统总控（Supervisor）
│   - 启动/停止/监控所有线程
│   - 主控制循环（读取world_model最新状态 → 决策aimbot/triggerbot → 调用controller → 发送到output）
│   - 全局异常捕获、优雅退出、信号处理
│   - 性能监控（可选：fps、延迟、丢帧统计）

├── config.py
│   职责：全局配置单例 + 类型安全读取器
│   - 从config.ini加载所有配置
│   - 提供get()/getbool()/getfloat()等方法，支持智能类型转换
│   - 所有模块都通过此单例访问配置

├── config.ini
│   职责：项目唯一配置入口
│   - 包含所有开关（enable_aimbot / enable_triggerbot / ego_motion_compensation 等）
│   - 调参（灵敏度、延迟范围、阈值、Kalman参数、策略包选择等）
│   - 游戏选择（current_game = valorant）

├── perception/
│   ├── capture.py
│   │   职责：画面采集 + 采集时刻鼠标位置精确记录
│   │   - 使用BetterCam以当前鼠标为中心采集256×256区域
│   │   - **关键**：每帧记录**采集瞬间**的鼠标位置（mouse_pos_at_capture）
│   │   - 将最近N帧（建议100~200ms）的mouse_pos存入环形缓冲区（供后期时间同步使用）
│   │   - 输出：FrameUpdate（frame + timestamp_capture + mouse_pos_at_capture + frame_id）

│   ├── bus.py
│   │   职责：状态总线（FrameBus + MousePos历史）
│   │   - 存储最新帧 + 最近若干帧的鼠标位置历史（环形缓冲区）
│   │   - 提供get_latest_frame() / get_mouse_pos_at_timestamp(ts)等接口
│   │   - 实现轻量锁 + 条件变量（避免轮询）

│   └── frame_preprocessor.py
│       职责：可选帧预处理（当前可为空或简单实现）
│       - 亮度/对比度/伽马校正
│       - 轻度去噪/锐化（如果模型对噪声敏感）
│       - 未来可加抗锯齿或色彩归一化

├── inference.py
│   职责：YOLOv13 TensorRT 异步推理线程
│   - 从bus取最新帧（使用采集时刻数据）
│   - 执行letterbox预处理 → TensorRT推理 → 简单后处理（过滤低conf）
│   - 输出detections（256×256相对坐标）+ 推理完成时间戳

├── world_model.py
│   职责：**唯一事实源 + 状态仲裁者**（当前最核心演进点）
│   - 接收inference的detections + 采集时刻的mouse_pos（通过bus同步）
│   - 坐标转换：相对 → 屏幕绝对（使用**采集时刻**的mouse_pos）
│   - 目标跟踪（IOU/简单匈牙利/未来Kalman）
│   - **关键升级**：Ego-motion补偿闭环
│   │   - 接收output_ghub实际下发的控制量u_k（dx,dy）
│   │   - 将u_k作为Kalman滤波器的控制输入，剥离自身运动
│   │   - 输出世界坐标系下更稳定的目标位置/速度
│   - 提供get_best_target() / get_state() / get_history()接口

├── aim_strategies/
│   ├── base_strategy.py
│   │   职责：抽象基类（接口定义）
│   │   - calculate_mouse_move()
│   │   - calculate_prediction()

│   ├── valorant/
│   │   ├── strategy.py
│   └── factory.py
│       职责：策略工厂（根据config.current_game / strategy_package创建实例）

├── controllers/
│   ├── base_controller.py
│   │   职责：控制器抽象基类

│   ├── pid_controller.py
│   │   职责：PID控制实现（主力控制器）

│   ├── humanize/
│   │   职责：拟人化处理管道（全部可开关）
│   │   - reaction_delay / fatigue / overshoot / noise / curve

│   └── controller_factory.py
│       职责：组合工厂
│       - 根据配置选择strategy + controller + humanize链路
│       - 最终输出平滑后的移动量 + trigger信号

├── output_ghub.py
│   职责：物理执行层 + 控制反馈闭环
│   - 接收移动量(dx,dy) + trigger信号
│   - 微移动累积 + 死区丢弃 + 频率限制
│   - 通过GHUB发送mouse_xy / mouse_down/up
│   - **关键**：将实际发送成功的控制量u_k（dx,dy）反馈给world_model（用于ego-motion补偿）

├── utils/
│   职责：通用工具集（不放业务逻辑）
│   - helpers.py：bbox处理、letterbox、scale_boxes、时间工具、坐标转换等
│   - types.py：数据结构（FrameInfo, Detection, MouseHistory等）

└── models/best256.engine
    职责：YOLOv13 256×256输入模型文件（TensorRT engine）



text## 当前最关键的三个工程痛点与解法方向（2025.12.26）

1. **自身位移补偿闭环（Ego-motion Compensation）**  
   痛点：甩枪/快速转动时，背景运动严重干扰目标检测速度  
   当前方案：仅使用历史 mouse_pos 做简单补偿  
   **推荐升级**：在 World Model 中引入**控制反馈闭环**  
   → 将 output_ghub 下发的实际移动量（u_k）作为 Kalman 滤波器的控制输入  
   → 实现世界坐标系下的目标速度剥离（真正意义上的 ego-motion compensation）

2. **亚毫秒级时间戳对齐**  
   痛点：推理结果出来时，鼠标位置已移动 10~30 像素（15~35ms 延迟）  
   **当前最佳实践**（强烈推荐立即实现）：  
   - perception 层建立**环形缓冲区**记录最近 100ms 鼠标位置 + 时间戳  
   - inference 输出 detections 时，使用**该帧采集时刻**对应的 mouse_pos 进行坐标转换  
   → 可将坐标误差从 20~30px 降至 <5px

3. **意图仲裁的自然性与安全性**  
   当前：简单的 enable 开关 + 阈值判断  
   **Phase 2 目标**：  
   - 引入**有限状态机（FSM）** 或 **行为树（Behavior Tree）**  
   - 典型状态转移示例：
IDLE → CONFIRMING(conf>0.45 & jerk<阈值) → AIM_TRACK → HOLD → FIRE → RELEASE
CONFIRMING → IDLE（jerk过大或conf突降）
text- 增加模糊逻辑权重：低 conf + 高 jerk → 自动进入犹豫/抑制状态

## 安全铁律（必须遵守）

1. 所有高危开关**默认关闭**，逐项开启且从小强度开始
2. **永不**同时开启 aimbot + triggerbot
3. **最有效的行为伪装组合**（优先级顺序）：
1. 微移动合并 + 死区
2. 采集时刻时间戳同步 + 控制反馈 ego-motion
3. 反应延迟 + 随机性 + 疲劳累积
4. 行为噪声 + 曲线形状 + 过冲回正
5. 意图仲裁层（FSM/BT）+ 状态犹豫机制

## 项目定位与演进宣言

**当前**：功能驱动的高质量异步实时系统（92→94分工程水准）  
**下一阶段（Phase 1.5）**：强时间一致性 + 控制反馈闭环（目标 95+）  
**最终目标（Phase 2）**：行为意图驱动 + 仲裁 + 全链路可观测回放（准工业级）

**再次郑重声明**  
**本项目仅限本地技术研究、算法验证与学术讨论。**  
**严禁用于任何线上游戏环境或商业用途。**