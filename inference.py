import logging
import threading
import time

import numpy as np
from ultralytics import YOLO

from config import config
from perception.bus import FrameBus
from world_model import WorldModel

logger = logging.getLogger("Inference")


class InferenceThread(threading.Thread):
    """
    YOLO26 推理循环。DXCam 现已直出 BGR（capture.py 中 output_color="BGR"），
    所以这里不再需要每帧 RGB→BGR 的翻转 copy；若 dxcam 回退到 RGB 则
    由 capture 层做一次翻转。
    """

    def __init__(
        self,
        bus: FrameBus,
        world_model: WorldModel,
        shutdown_event: threading.Event,
        frame_ready_event: threading.Event,
    ):
        super().__init__(name="InferenceThread", daemon=True)
        self.bus = bus
        self.world_model = world_model
        self.shutdown_event = shutdown_event
        self.frame_ready_event = frame_ready_event
        self.last_processed_id = -1
        self._first_push_logged = False

        model_path = config.getstr("Inference", "model_path", "models/yolo26n.engine")
        img_size = config.getint("General", "capture_size", 256)

        try:
            logger.info("Loading YOLO26 Model (NMS-Free)...")
            self.model = YOLO(model_path, task='detect')
            dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            self.model.predict(source=dummy, imgsz=img_size, device=0, half=True, verbose=False)
            logger.info("YOLO ready")
        except Exception as e:
            logger.error("YOLO init failed: %s", e)
            self.model = None

    def run(self):
        if self.model is None:
            logger.error("InferenceThread not started: YOLO model is None (check model_path / TensorRT)")
            return

        conf_thres = config.getfloat("Inference", "conf_threshold", 0.40)
        max_det = config.getint("Inference", "max_det", 20)
        img_size = config.getint("General", "capture_size", 256)

        last_print_time = 0.0
        last_inference_ms = 0.0

        while not self.shutdown_event.is_set():
            if not self.frame_ready_event.wait(timeout=0.005):
                continue
            self.frame_ready_event.clear()

            frame_info = self.bus.get_latest()
            if frame_info is None or frame_info.frame_id <= self.last_processed_id:
                continue

            try:
                # dxcam 出来的帧通常已 C-contiguous；防御性兜底一次
                frame = frame_info.frame
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)

                t_inference_start = time.perf_counter()
                results = self.model.predict(
                    source=frame,
                    imgsz=img_size,
                    conf=conf_thres,
                    max_det=max_det,
                    verbose=False,
                    device=0,
                    half=True,
                    stream=True,
                )

                detections = np.empty((0, 6), dtype=np.float32)
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        detections = r.boxes.data.cpu().numpy()
                    break

                t_inference_done = time.perf_counter()
                last_inference_ms = (t_inference_done - t_inference_start) * 1000.0

                now = t_inference_done
                if now - last_print_time > 1.0:
                    if len(detections) > 0:
                        best_conf = detections[0][4]
                        best_cls = int(detections[0][5])
                        logger.info(
                            "%d target(s) | best: cls=%d conf=%.2f | infer=%.1fms",
                            len(detections), best_cls, best_conf, last_inference_ms,
                        )
                    else:
                        logger.debug("No targets in FOV | infer=%.1fms", last_inference_ms)
                    last_print_time = now

                self.world_model.update_detections(
                    detections=detections,
                    frame_id=frame_info.frame_id,
                    t_capture=frame_info.t_cap,
                    t_done=t_inference_done,
                )
                if not self._first_push_logged:
                    self._first_push_logged = True
                    logger.info(
                        "Pipeline: first YOLO → world_model.update_detections (frame_id=%d, dets=%d, infer=%.1fms). Main.tick should wake next.",
                        frame_info.frame_id,
                        len(detections),
                        last_inference_ms,
                    )
                self.last_processed_id = frame_info.frame_id

            except Exception as e:
                logger.error("Inference loop error: %s", e)
