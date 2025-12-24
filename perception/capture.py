# perception/capture.py
import time
import threading
import numpy as np
import bettercam
from config import config
from perception.bus import FrameBus

class CaptureThread(threading.Thread):
    """
    针对 FPS 优化的 BetterCam 采集线程
    特点：固定屏幕中心 256x256 采样，不再实时获取鼠标坐标
    """
    def __init__(self, bus: FrameBus, shutdown_event: threading.Event):
        super().__init__(name="CaptureThread", daemon=True)
        self.bus = bus
        self.shutdown_event = shutdown_event

        # 基础配置
        self.capture_size = config.getint("General", "capture_size", 256)
        self.target_fps = config.getint("General", "capture_fps_target", 360)
        self.frame_interval = 1.0 / self.target_fps

        # 获取屏幕中心
        sw = config.getint("General", "screen_width", 1920)
        sh = config.getint("General", "screen_height", 1080)
        self.center_x, self.center_y = sw // 2, sh // 2

        # 预先计算固定的采集区域 (left, top, right, bottom)
        half = self.capture_size // 2
        self.fixed_region = (
            self.center_x - half,
            self.center_y - half,
            self.center_x + half,
            self.center_y + half
        )

        # BetterCam 初始化 - 强制使用 GPU 加速
        self.camera = bettercam.create(
            device_idx=0,
            output_idx=0,
            region=self.fixed_region, # 直接绑定固定区域
            output_color="BGR",
            nvidia_gpu=True,
            max_buffer_len=8
        )

    def run(self):
        print(f"[Capture] FPS 模式启动: 固定中心 {self.capture_size}x{self.capture_size}")
        frame_id = 0

        while not self.shutdown_event.is_set():
            loop_start = time.perf_counter()

            try:
                # 执行高速捕获
                frame = self.camera.grab() # 无需再传入 region

                if frame is not None:
                    # 发布到总线，mouse_pos 直接传中心点
                    self.bus.publish_frame(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp=loop_start,
                        mouse_pos_at_capture=(self.center_x, self.center_y)
                    )
                    frame_id += 1

            except Exception as e:
                print(f"[Capture] 异常: {e}")

            # 频率控制
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, self.frame_interval - elapsed))

        print("[Capture] 线程退出")