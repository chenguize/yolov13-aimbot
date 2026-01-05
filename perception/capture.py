import time
import threading
import numpy as np
import dxcam  # 替换为 dxcam
from config import config
from perception.bus import FrameBus, FrameInfo


class CaptureThread(threading.Thread):
    """
    Phase 5 专用采集线程 (dxcam 极速版)
    优势：原生 DirectX 支持，采集延迟更低且稳定。
    """

    def __init__(self, bus: FrameBus, shutdown_event: threading.Event):
        super().__init__(name="CaptureThread", daemon=True)
        self.bus = bus
        self.shutdown_event = shutdown_event

        self.target_fps = config.getint("General", "capture_fps_target", 240)
        self.capture_size = config.getint("General", "capture_size", 256)

        sw = config.getint("General", "screen_width", 1920)
        sh = config.getint("General", "screen_height", 1080)
        self.center_x, self.center_y = sw // 2, sh // 2

        half = self.capture_size // 2
        # dxcam 的 region 格式为 (left, top, right, bottom)
        self.roi = (
            self.center_x - half,
            self.center_y - half,
            self.center_x + half,
            self.center_y + half
        )

        try:
            # 初始化 dxcam 实例
            # output_color="BGR" 确保与 InferenceThread 兼容
            self.camera = dxcam.create(device_idx=0, output_idx=0, output_color="BGR")
            # 开启后台线程模式抓取，fps 参数直接控制采集频率
            self.camera.start(region=self.roi, target_fps=self.target_fps)
        except Exception as e:
            print(f"[Capture] ❌ DXCAM 初始化失败: {e}")
            self.camera = None

    def run(self):
        if not self.camera:
            return

        print(f"[Capture] DXCAM 采集已启动 (ROI: {self.capture_size}x{self.capture_size})")
        frame_count = 0

        while not self.shutdown_event.is_set():
            try:
                # get_latest_frame 会直接返回后台线程捕获的最新一帧
                # 这比同步 grab() 响应更快，消除了等待时间
                frame = self.camera.get_latest_frame()
                t_cap = time.perf_counter()

                if frame is not None:
                    # 封装为 FrameInfo 发布
                    info = FrameInfo(
                        frame=frame,
                        frame_id=frame_count,
                        t_cap=t_cap,
                        center_pos=(self.center_x, self.center_y)
                    )
                    self.bus.publish(info)
                    frame_count += 1

                # 由于 camera.start 已经在后台控制了 FPS，
                # 我们这里只需极短休眠，降低 CPU 空转压力
                time.sleep(0.0005)

            except Exception as e:
                print(f"[Capture] Loop Error: {e}")
                time.sleep(0.01)

        if self.camera:
            self.camera.stop()
        print("[Capture] Stopped.")