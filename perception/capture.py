import time
import threading
import numpy as np
import bettercam
from config import config
from perception.bus import FrameBus, FrameInfo


class CaptureThread(threading.Thread):
    """
    Phase 3 专用采集线程
    特点：
    1. 极速：固定 256x256 中心区域，无动态计算开销。
    2. 精准：在 camera.grab() 返回的瞬间记录 t_cap。
    """

    def __init__(self, bus: FrameBus, shutdown_event: threading.Event):
        super().__init__(name="CaptureThread", daemon=True)
        self.bus = bus
        self.shutdown_event = shutdown_event

        # 配置参数
        self.target_fps = config.getint("General", "capture_fps_target", 200)
        self.capture_size = config.getint("General", "capture_size", 320)
        self.frame_interval = 1.0 / self.target_fps

        # 屏幕几何
        sw = config.getint("General", "screen_width", 1920)
        sh = config.getint("General", "screen_height", 1080)
        self.center_x, self.center_y = sw // 2, sh // 2

        # 计算固定的 ROI (Region of Interest)
        # left, top, right, bottom
        half = self.capture_size // 2
        self.roi = (
            self.center_x - half,
            self.center_y - half,
            self.center_x + half,
            self.center_y + half
        )

        # BetterCam 初始化 (强制 GPU)
        self.camera = bettercam.create(
            device_idx=0,
            output_idx=0,
            region=self.roi,  # 固定区域
            output_color="BGR",
            nvidia_gpu=True,
            max_buffer_len=4  # 缓冲不用太大，我们要的是实时性
        )

    def run(self):
        print(f"[Capture] Started. Region: {self.capture_size}x{self.capture_size} @ Center")
        frame_count = 0

        while not self.shutdown_event.is_set():
            t_start = time.perf_counter()

            try:
                # 1. 抓取 (阻塞直到有帧)
                frame = self.camera.grab()

                # 2. 关键：立即打上时间戳
                # 这是这一帧"光子到达传感器"的最接近时间估算
                t_cap = time.perf_counter()

                if frame is not None:
                    # 3. 封装并广播
                    info = FrameInfo(
                        frame=frame,
                        frame_id=frame_count,
                        t_cap=t_cap,
                        center_pos=(self.center_x, self.center_y)
                    )
                    self.bus.publish(info)
                    frame_count += 1

            except Exception as e:
                print(f"[Capture] Error: {e}")
                time.sleep(0.1)

            # 4. 软限频 (虽然 BetterCam 内部有限制，这里做个双保险)
            # 避免在显卡负载极低时跑出 1000FPS 烧 CPU
            elapsed = time.perf_counter() - t_start
            wait = self.frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)

        # 清理资源
        self.camera.stop()
        print("[Capture] Stopped.")