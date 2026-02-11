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
        self.frame_ready_event = frame_ready_event  # [新增] 同步信号

        self.target_fps = config.getint("General", "capture_fps_target", 240)
        self.capture_size = config.getint("General", "capture_size", 256)

        sw = config.getint("General", "screen_width", 1920)
        sh = config.getint("General", "screen_height", 1080)
        self.center_x, self.center_y = sw // 2, sh // 2

        half = self.capture_size // 2
        self.roi = (self.center_x - half, self.center_y - half, self.center_x + half, self.center_y + half)

        try:
            self.camera = dxcam.create(device_idx=0, output_idx=0, output_color="BGR")
            self.camera.start(region=self.roi, target_fps=self.target_fps)
        except Exception as e:
            print(f"[Capture] ❌ DXCAM Init Failed: {e}")
            self.camera = None

    def run(self):
        if not self.camera: return
        print(f"[Capture] Started (Event-Driven Mode)")

        frame_count = 0

        # [优化] 预分配 FrameInfo 对象，避免循环内重复 malloc (虽然 Python GC 很快，但高频下能省则省)
        # 注意：这里主要为了代码结构清晰，极致优化可以用对象池，但这里先不做过度优化

        while not self.shutdown_event.is_set():
            try:
                # get_latest_frame 依然是主动获取，但配合后面的 Event，我们可以让消费者完全被动
                frame = self.camera.get_latest_frame()

                if frame is not None:
                    t_cap = time.perf_counter()
                    info = FrameInfo(
                        frame=frame,
                        frame_id=frame_count,
                        t_cap=t_cap,
                        center_pos=(self.center_x, self.center_y)
                    )
                    self.bus.publish(info)

                    # [关键算法改变] 生产一帧，立即通知消费者
                    self.frame_ready_event.set()

                    frame_count += 1

                # [优化] 这里的 sleep 仅用于释放 GIL 给 DXCam 的后台线程一点喘息空间
                # 由于是生产者，这里保留微小 sleep 是安全的
                time.sleep(0.0001)

            except Exception:
                pass

        if self.camera: self.camera.stop()