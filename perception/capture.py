import time
import threading
import numpy as np
import dxcam
from config import config
from perception.bus import FrameBus, FrameInfo


class CaptureThread(threading.Thread):
    def __init__(self, bus: FrameBus, shutdown_event: threading.Event, frame_ready_event: threading.Event):
        super().__init__(name="CaptureThread", daemon=True)
        self.bus = bus
        self.shutdown_event = shutdown_event
        self.frame_ready_event = frame_ready_event

        self.target_fps = config.getint("General", "capture_fps_target", 240)
        self.capture_size = config.getint("General", "capture_size", 256)

        sw = config.getint("General", "screen_width", 1920)
        sh = config.getint("General", "screen_height", 1080)
        self.center_x, self.center_y = sw // 2, sh // 2

        half = self.capture_size // 2
        # ROI 格式: (left, top, right, bottom)
        self.roi = (self.center_x - half, self.center_y - half, self.center_x + half, self.center_y + half)

        self.camera = None
        try:
            # 🌟 核心修复：在 create 阶段就强制传入 region，让 dxcam 按照 256x256 分配底层 Buffer
            self.camera = dxcam.create(
                device_idx=0,
                output_idx=0,
                output_color="RGB",
                max_buffer_len=64,
                region=self.roi  # <--- 就是加了这一行
            )

            if self.camera:
                # start 里也可以保留，双保险
                self.camera.start(region=self.roi, target_fps=self.target_fps)
        except Exception as e:
            print(f"[Capture] ❌ DXCAM Init Failed: {e}")

    # ... 剩下的 run 方法保持不变 ...
    def run(self):
        if not self.camera:
            print("[Capture] ❌ No camera instance, thread exiting.")
            return

        print(f"[Capture] Started (Event-Driven Mode) | ROI: {self.roi}")

        frame_count = 0

        while not self.shutdown_event.is_set():
            try:
                frame = self.camera.get_latest_frame()

                if frame is not None:
                    # 验证 shape
                    if frame.shape[0] != self.capture_size or frame.shape[1] != self.capture_size:
                        continue

                    t_cap = time.perf_counter()
                    info = FrameInfo(
                        frame=frame,
                        frame_id=frame_count,
                        t_cap=t_cap,
                        center_pos=(self.center_x, self.center_y)
                    )
                    self.bus.publish(info)

                    # 通知推理线程
                    self.frame_ready_event.set()
                    frame_count += 1
                else:
                    time.sleep(0.001)

            except Exception as e:
                # 如果还是有报错，不再暴力 stop，避免线程锁死
                print(f"[Capture] ⚠️ Runtime Error: {e}")
                time.sleep(0.01)

        try:
            self.camera.stop()
        except:
            pass
        print(f"[Capture] Stopped. Total frames: {frame_count}")