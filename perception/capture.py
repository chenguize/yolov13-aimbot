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
        # dxcam 直出 BGR（YOLO/TensorRT 的原生输入格式），省掉 inference
        # 每帧 256×256×3 的 RGB→BGR 翻转 copy（240fps × 196KB ≈ 47MB/s）。
        # 旧版 dxcam 若不识别 BGR，回退 RGB + 上层翻转。
        try:
            self.camera = dxcam.create(
                device_idx=0,
                output_idx=0,
                output_color="BGR",
                max_buffer_len=64,
                region=self.roi,
            )
            self.output_is_bgr = True
        except (TypeError, ValueError) as e:
            print(f"[Capture] ⚠️ DXCAM BGR mode not supported, falling back to RGB: {e}")
            try:
                self.camera = dxcam.create(
                    device_idx=0,
                    output_idx=0,
                    output_color="RGB",
                    max_buffer_len=64,
                    region=self.roi,
                )
                self.output_is_bgr = False
            except Exception as e2:
                print(f"[Capture] ❌ DXCAM Init Failed: {e2}")
                return
        except Exception as e:
            print(f"[Capture] ❌ DXCAM Init Failed: {e}")
            return

        if self.camera:
            try:
                self.camera.start(region=self.roi, target_fps=self.target_fps)
            except Exception as e:
                print(f"[Capture] ❌ DXCAM Start Failed: {e}")
                self.camera = None

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
                    if frame.shape[0] != self.capture_size or frame.shape[1] != self.capture_size:
                        continue

                    # 旧版 dxcam 若不识别 BGR，这里翻转一次；新版直出 BGR
                    if not self.output_is_bgr:
                        frame = frame[..., ::-1]

                    t_cap = time.perf_counter()
                    info = FrameInfo(
                        frame=frame,
                        frame_id=frame_count,
                        t_cap=t_cap,
                        center_pos=(self.center_x, self.center_y)
                    )
                    self.bus.publish(info)
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