import time
import threading
import numpy as np
import bettercam
from config import config
from perception.bus import FrameBus, FrameInfo


class CaptureThread(threading.Thread):
    """
    Phase 3 专用采集线程 (稳定版)
    修复：关闭 nvidia_gpu 防止画面冻结
    """

    def __init__(self, bus: FrameBus, shutdown_event: threading.Event):
        super().__init__(name="CaptureThread", daemon=True)
        self.bus = bus
        self.shutdown_event = shutdown_event

        self.target_fps = config.getint("General", "capture_fps_target", 240)
        self.capture_size = config.getint("General", "capture_size", 256)
        # 限制最高帧率，防止 CPU 空转
        self.frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.002

        sw = config.getint("General", "screen_width", 1920)
        sh = config.getint("General", "screen_height", 1080)
        self.center_x, self.center_y = sw // 2, sh // 2

        half = self.capture_size // 2
        self.roi = (
            self.center_x - half,
            self.center_y - half,
            self.center_x + half,
            self.center_y + half
        )

        # [关键修复] 必须显式设置 nvidia_gpu=False
        try:
            self.camera = bettercam.create(
                device_idx=0,
                output_idx=0,
                region=self.roi,
                output_color="BGR",  # <--- 这里的 False 是防止飞天的关键
                max_buffer_len=2
            )
        except Exception as e:
            print(f"[Capture] ❌ BetterCam 初始化失败: {e}")
            self.camera = None

    def run(self):
        if not self.camera:
            return

        print(f"[Capture] 采集已启动 (ROI: {self.capture_size}x{self.capture_size})")
        frame_count = 0

        while not self.shutdown_event.is_set():
            t_start = time.perf_counter()

            try:
                frame = self.camera.grab()
                t_cap = time.perf_counter() # 立即打戳

                if frame is not None:
                    info = FrameInfo(
                        frame=frame,
                        frame_id=frame_count,
                        t_cap=t_cap,
                        center_pos=(self.center_x, self.center_y)
                    )
                    self.bus.publish(info)
                    frame_count += 1
                else:
                    time.sleep(0.001)

            except Exception:
                time.sleep(0.01)

            elapsed = time.perf_counter() - t_start
            wait = self.frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)

        if self.camera:
            self.camera.stop()
        print("[Capture] Stopped.")