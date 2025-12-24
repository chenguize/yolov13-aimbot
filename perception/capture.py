# perception/capture.py - 画面采集模块
#
# 核心职责：
# 1. BetterCam CUDA 抓取线程（高性能屏幕捕获）
# 2. 256x256 鼠标中心区域捕获（精确目标区域）
# 3. 采集时刻鼠标位置记录（时间一致性核心）
# 4. 环形缓冲区管理（历史鼠标位置存储）
#
# 架构特点：记录采集瞬间鼠标位置 → 存入环形缓冲
#

import time
import threading
import numpy as np
import win32api
from collections import deque
from typing import Tuple
import bettercam  # 需要 pip install bettercam
from config import config
from perception.bus import FrameBus, FrameInfo


class CaptureThread(threading.Thread):
    """
    真实 BetterCam 采集线程 - 256x256 鼠标中心捕获
    核心：记录采集瞬间鼠标位置 → 存入环形缓冲
    """

    def __init__(self, bus: FrameBus, shutdown_event: threading.Event):
        """初始化采集线程"""
        super().__init__(name="CaptureThread", daemon=True)
        self.bus = bus  # 帧总线引用，用于发布采集结果
        self.shutdown_event = shutdown_event  # 关闭事件，用于优雅退出

        # 从配置中读取采集参数
        self.capture_size = config.getint("General", "capture_size", 256)  # 采集区域大小
        self.half_size = self.capture_size // 2  # 中心偏移量
        self.target_fps = config.getint("General", "capture_fps_target", 360)  # 目标采集帧率
        self.frame_interval = 1.0 / self.target_fps  # 每帧间隔时间

        # BetterCam 初始化 - 高性能 CUDA 加速捕获
        self.camera = bettercam.create(
            device_idx=0,  # 默认主显卡
            output_idx=0,  # 主显示器（0）
            region=None,  # 先不设置，run() 中动态覆盖
            output_color="BGR",  # OpenCV 兼容格式
            nvidia_gpu=True,  # 强制用 GPU 加速（推荐）
            torch_cuda=False,  # 不转 torch.Tensor（我们用 TensorRT，不需要）
            max_buffer_len=8  # 缓冲区大小，建议 4~16
        )

        # 环形缓冲区：最近 200ms 鼠标位置历史
        # 用于时间一致性，解决推理延迟导致的坐标不匹配问题
        self.mouse_history = deque(maxlen=int(self.target_fps * 0.2))

    def run(self):
        """采集线程主循环"""
        print("[Capture] BetterCam 256x256 鼠标中心采集启动")
        frame_id = 0  # 帧 ID，用于标识每帧

        while not self.shutdown_event.is_set():
            loop_start = time.perf_counter()

            # 采集时刻鼠标位置（最关键！）
            # 这是时间一致性的核心：记录捕获帧时的准确鼠标位置
            mx, my = win32api.GetCursorPos()
            capture_ts = time.perf_counter()  # 精确的时间戳

            # 将鼠标位置和时间戳存入历史记录
            self.mouse_history.append((capture_ts, mx, my))

            # 计算采集区域 - 以当前鼠标位置为中心的 256x256 区域
            region = (
                max(0, mx - self.half_size),  # 左边界，不能小于0
                max(0, my - self.half_size),  # 上边界，不能小于0
                min(config.getint("General", "screen_width", 1920), mx + self.half_size),  # 右边界，不能超过屏幕宽度
                min(config.getint("General", "screen_height", 1080), my + self.half_size)  # 下边界，不能超过屏幕高度
            )

            try:
                # 执行屏幕捕获
                frame = self.camera.grab(region=region)

                if frame is not None:
                    # 如果区域不是 256x256，进行中心裁剪以确保尺寸一致
                    if frame.shape[:2] != (self.capture_size, self.capture_size):
                        h, w = frame.shape[:2]
                        frame = frame[
                            (h - self.capture_size) // 2 : (h + self.capture_size) // 2,
                            (w - self.capture_size) // 2 : (w + self.capture_size) // 2
                        ]

                    # 通过总线发布采集结果
                    self.bus.publish_frame(
                        frame=frame,  # 捕获的图像帧
                        frame_id=frame_id,  # 帧 ID
                        timestamp=capture_ts,  # 捕获时间戳
                        mouse_pos_at_capture=(mx, my)  # 捕获时刻的鼠标位置
                    )
                    frame_id += 1  # 帧 ID 递增

            except Exception as e:
                print(f"[Capture] 抓图异常: {e}")

            # 控制采集帧率，确保不超过目标帧率
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, self.frame_interval - elapsed))

        print("[Capture] 线程退出")