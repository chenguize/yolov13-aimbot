# perception/capture.py
import time
import threading
import numpy as np
from typing import Optional
from perception.bus import FrameBus

class CaptureThread(threading.Thread):
    """画面采集线程 - 使用 BetterCam (占位)"""

    def __init__(self, bus: FrameBus, shutdown_event: threading.Event):
        super().__init__(name="CaptureThread", daemon=False)
        self.bus = bus
        self.shutdown_event = shutdown_event
        # self.camera = bettercam.create(...)  # 实际使用时打开

    def run(self):
        print("[Capture] 画面采集线程启动")
        frame_counter = 0

        while not self.shutdown_event.is_set():
            start = time.perf_counter()

            # frame = self.camera.grab()          # 实际调用
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 占位

            if frame is not None:
                ts = time.perf_counter()
                self.bus.publish_frame(frame, frame_counter, ts)
                frame_counter += 1

            # 控制采集帧率 ≈240-360fps
            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, 0.0028 - elapsed))  # 目标 ~357fps

        print("[Capture] 画面采集线程退出")