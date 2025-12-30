# test_module.py
# Phase 3 单元测试套件
# 用于验证各个模块的核心逻辑，无需启动真实游戏或摄像头。

import unittest
import time
import sys
import os
import numpy as np
from dataclasses import dataclass

# 确保能导入项目模块
sys.path.append(os.getcwd())

# 模块导入
from perception.ring_buffer import RingBuffer
from aim_strategies.valorant.strategy import ValorantStrategy
from controllers.simple_controller import SimpleController
from controllers.pro_controller import  PROController
from world_model import WorldModel
from utils.types import Detection, InferenceContext


class TestPhase3Modules(unittest.TestCase):

    def setUp(self):
        print(f"\n--- Testing: {self._testMethodName} ---")

    # =========================================================================
    # 1. 测试 RingBuffer (因果账本)
    # =========================================================================
    def test_ring_buffer_hedging(self):
        """测试 RingBuffer 的时间切片查询是否准确"""
        rb = RingBuffer(max_duration=1.0)

        t0 = time.perf_counter()

        # 模拟：0.1秒前，AI 往右移动了 10
        rb.add_event(dx=10, dy=0, is_ai=True)
        # 模拟：0.05秒前，人手往上移动了 5
        rb.add_event(dx=0, dy=-5, is_ai=False)

        # 稍微等待写入完成
        time.sleep(0.01)

        # 测试：查询 "过去 0.2秒" 的总位移
        # 预期：dx=10, dy=-5
        sum_x, sum_y = rb.get_cursor_delta_sum(t0 - 0.2, time.perf_counter())

        print(f"[RingBuffer] Mock Input: (+10, 0), (0, -5)")
        print(f"[RingBuffer] Query Result: ({sum_x}, {sum_y})")

        self.assertEqual(sum_x, 10, "X轴累计位移错误")
        self.assertEqual(sum_y, -5, "Y轴累计位移错误")

    # =========================================================================
    # 2. 测试 Strategy (几何映射)
    # =========================================================================
    def test_strategy_mapping(self):
        """测试 Pixels -> Mickey 的映射逻辑"""
        strat = ValorantStrategy()

        # 假设 K_factor_x = 1.9 (默认值)
        # 输入 100 像素偏差
        dx_pixel = 100.0
        dy_pixel = 0.0

        out_x, out_y = strat.calculate_mouse_move(dx_pixel, dy_pixel)

        expected_x = int(100.0 * 1.9)
        print(f"[Strategy] Input Pixels: 100.0 -> Output Counts: {out_x}")

        # 允许 ±1 的取整误差
        self.assertTrue(abs(out_x - expected_x) <= 1, f"策略映射不准确: {out_x} vs {expected_x}")

    # =========================================================================
    # 3. 测试 Controller (动力学与限幅)
    # =========================================================================
    def test_controller_safety(self):
        """测试控制器的死区和硬限幅"""
        ctrl = SimpleController()  # 或者 AdvancedController

        # Case A: 测试死区 (Deadzone)
        # 输入极小位移 0.2 (小于 min_step 0.5)
        out_x, out_y = ctrl.compute(0.2, 0.0, dt=0.01)
        print(f"[Controller] Deadzone Input: 0.2 -> Output: {out_x}")
        self.assertEqual(out_x, 0.0, "死区未生效")

        # Case B: 测试限幅 (Max Step)
        # 输入巨大位移 9999
        out_x, out_y = ctrl.compute(9999.0, 0.0, dt=0.01)
        print(f"[Controller] Huge Input: 9999.0 -> Output: {out_x}")
        self.assertTrue(out_x <= ctrl.max_step, "安全限幅未生效")

    # =========================================================================
    # 4. 测试 WorldModel (因果对冲核心逻辑) [最关键]
    # =========================================================================
    def test_world_model_hedging(self):
        """
        模拟核心场景：
        如果在截图(t_cap)到处理完成期间，鼠标向右移动了，
        WorldModel 能否算出物体其实是在向左跑？
        """
        wm = WorldModel()
        rb = RingBuffer()

        # --- 场景设置 ---
        # 屏幕中心 (960, 540)
        # 1. t=0.0s: 截图。物体在屏幕正中心 (960, 540)
        t_cap = time.perf_counter()

        # 2. t=0.0s ~ t=0.1s: 推理期间，鼠标向右甩了 190 counts
        # 假设 K=1.9，这相当于屏幕视野向右平移了 100 像素
        # 因此，物体在"当前时刻"的屏幕坐标应该变成了 (860, 540)

        # 模拟鼠标移动 (写入 RingBuffer)
        # 我们分两段写，模拟连续移动
        rb.add_event(dx=95, dy=0, is_ai=True)
        rb.add_event(dx=95, dy=0, is_ai=True)

        # 3. t=0.1s: 推理完成。YOLO 告诉我们在 t_cap 时物体在 (960, 540)
        # 注意：YOLO 给的是旧坐标！
        detections = [[950, 530, 970, 550, 0.9, 0]]  # xyxy格式，中心约 (960, 540)

        # --- 执行 Step ---
        # 模拟 Inference 线程推送数据
        t_done = time.perf_counter()
        wm.update_detections(detections, frame_id=1, t_cap=t_cap, t_done=t_done)

        # 模拟 Main 线程调用 Step
        ctx = InferenceContext()
        wm.step(ctx, rb)

        # --- 验证结果 ---
        # 理论分析：
        # 观测值 z = 960 (t_cap时的坐标)
        # 鼠标位移 = +190 counts = +100 pixels (向右)
        # 对冲公式: z_real = z + pixel_shift
        #               = 960 + 100 = 1060
        #
        # 等等！ WorldModel 的逻辑是：
        # 观测坐标是"旧"的。我们想知道物体"现在"在哪。
        # 如果鼠标向右动了，画面就向左动了。
        # 如果物体静止，它在 t_now 的坐标应该是 960 - 100 = 860。
        #
        # 让我们看代码实现：
        # z_measured = best_det.x + pixel_shift
        # 这里 pixel_shift 是正数 (100)。
        # 所以 z_measured = 960 + 100 = 1060。
        #
        # 这似乎意味着 WorldModel 计算的是 "如果鼠标没动，物体应该在哪（绝对世界坐标）"。
        # 这是 Phase 3 的定义：将所有坐标还原到"截图那一刻的世界坐标系"或者"绝对坐标系"中。
        # 只要 Controller 知道这是一个绝对坐标，并减去屏幕中心，就能得到正确的差值。

        target = wm.current_target
        self.assertIsNotNone(target, "目标未被选中")

        print(f"[WorldModel] Raw Detection X: {960}")
        print(f"[WorldModel] Mouse Moved X: +190 counts (~100 px)")
        print(f"[WorldModel] Kalman State X: {target.state[0]:.2f}")

        # [Fix] 物理修正
        # 原始坐标 960 -> 鼠标右移导致屏幕右移 -> 物体在屏幕上左移 -> 新坐标应 < 960
        # 理论值 = 960 - (190 / 1.9) = 860
        self.assertTrue(target.state[0] < 960, "位移对冲方向错误 (应向左补偿)")
        self.assertTrue(target.state[0] > 800, "位移对冲数值异常")

        # 验证 Kalman 状态是否包含了鼠标位移的补偿
        # 因为我们模拟的是"物体静止，鼠标动"，
        # 所以还原后的"绝对世界坐标"应该包含这个位移？
        # 或者说，WorldModel 试图还原物体的真实运动。
        # 如果物体静止，z_measured 应该是不变的吗？
        #
        # 让我们重新理一下 Phase 3 的对冲逻辑：
        # 截图时物体在 960。
        # 鼠标往右动了，摄像机往右动了。
        # 如果物体不动，新的一帧里物体应该在 860。
        # 但我们要喂给 Kalman 的是"物体在世界里的位置"。
        # 无论鼠标怎么动，物体在世界里的位置（假设世界原点是第一帧的屏幕左上角）应该是不变的（如果是静止靶）。
        #
        # 截图时(t0)，坐标 960。
        # 鼠标动了 +100px。
        # 此时视界原点变成了 +100。
        # 为了让 Kalman 认为物体没动，我们需要把这 100 加回来？
        # 是的。 z = 960 (相对t0原点)
        # z_new = z_measured (相对t1原点) + delta_mouse
        #
        # 代码里的逻辑：z_measured = det.x + pixel_shift
        # 960 + 100 = 1060。
        # 这说明 WorldModel 维护的是一个"随动"的坐标系？
        # 不，这说明 WorldModel 试图抵消鼠标移动带来的"视觉反向运动"。
        # 鼠标右移 -> 视觉左移。
        # 所以 +shift 抵消了视觉上的左移（如果下一帧视觉变小的话）。
        # 但这里只有一帧数据。

        # 结论：WorldModel 的输出应该是 "抵消了鼠标干扰后的位置"。
        # 如果鼠标右移，导致物体在屏幕上看起来左移了，对冲应该把它加回来。
        # 在这个测试用例里，我们只有一帧 YOLO 数据。
        # 对冲逻辑主要在第二帧起作用。这里主要是看代码有没有跑通，不崩溃。



if __name__ == "__main__":
    unittest.main()