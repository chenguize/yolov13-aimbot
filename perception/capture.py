# perception/capture.py
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
        super().__init__(name="CaptureThread", daemon=True)
        self.bus = bus
        self.shutdown_event = shutdown_event

        self.capture_size = config.getint("General", "capture_size", 256)
        self.half_size = self.capture_size // 2
        self.target_fps = config.getint("General", "capture_fps_target", 360)
        self.frame_interval = 1.0 / self.target_fps

        # BetterCam 初始化
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
        self.mouse_history = deque(maxlen=int(self.target_fps * 0.2))

    def run(self):
        print("[Capture] BetterCam 256x256 鼠标中心采集启动")
        frame_id = 0

        while not self.shutdown_event.is_set():
            loop_start = time.perf_counter()

            # 采集时刻鼠标位置（最关键！）
            mx, my = win32api.GetCursorPos()
            capture_ts = time.perf_counter()

            self.mouse_history.append((capture_ts, mx, my))

            # 动态区域
            region = (
                max(0, mx - self.half_size),
                max(0, my - self.half_size),
                min(config.getint("General", "screen_width", 1920), mx + self.half_size),
                min(config.getint("General", "screen_height", 1080), my + self.half_size)
            )

            try:
                frame = self.camera.grab(region=region)

                if frame is not None:
                    # 如果区域不是 256x256，进行中心裁剪
                    if frame.shape[:2] != (self.capture_size, self.capture_size):
                        h, w = frame.shape[:2]
                        frame = frame[
                            (h - self.capture_size) // 2 : (h + self.capture_size) // 2,
                            (w - self.capture_size) // 2 : (w + self.capture_size) // 2
                        ]

                    self.bus.publish_frame(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp=capture_ts,
                        mouse_pos_at_capture=(mx, my)
                    )
                    frame_id += 1

            except Exception as e:
                print(f"[Capture] 抓图异常: {e}")

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, self.frame_interval - elapsed))

        print("[Capture] 线程退出")