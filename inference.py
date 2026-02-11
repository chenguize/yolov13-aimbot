import time
import threading
import numpy as np
from ultralytics import YOLO
from config import config
from perception.bus import FrameBus, FrameInfo
from world_model import WorldModel


class InferenceThread(threading.Thread):
    def __init__(self, bus: FrameBus, world_model: WorldModel, shutdown_event: threading.Event,
                 frame_ready_event: threading.Event):
        super().__init__(name="InferenceThread", daemon=True)
        self.bus = bus
        self.world_model = world_model
        self.shutdown_event = shutdown_event
        self.frame_ready_event = frame_ready_event
        self.last_processed_id = -1

        # [YOLO26] 确保这里加载的是 YOLO26 的模型文件 (pt 或 engine)
        model_path = config.getstr("Inference", "model_path", "models/yolo26n.engine")
        img_size = config.getint("General", "capture_size", 256)

        try:
            print(f"[Inference] Loading YOLO26 Model (NMS-Free)...")
            self.model = YOLO(model_path, task='detect')

            # Warmup
            dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            # YOLO26 不需要 NMS 预热，直接跑一次即可
            self.model.predict(source=dummy, imgsz=img_size, device=0, half=True, verbose=False)
            print(f"[Inference] ✅ Ready")
        except Exception as e:
            print(f"[Inference] ❌ Failed: {e}")
            self.model = None

    def run(self):
        if self.model is None: return

        conf_thres = config.getfloat("Inference", "conf_threshold", 0.40)
        # [YOLO26] iou_threshold 已不再需要 (无 NMS)
        max_det = config.getint("Inference", "max_det", 20)
        img_size = config.getint("General", "capture_size", 256)

        # [算法优化] 确定要检测的目标 ID。假设 7 是敌人。
        target_classes = [7]

        while not self.shutdown_event.is_set():
            # 阻塞等待新帧信号，超时 5ms 防止死锁
            if not self.frame_ready_event.wait(timeout=0.005):
                continue

            self.frame_ready_event.clear()

            # 从总线拿数据
            frame_info = self.bus.get_latest()

            if frame_info is None or frame_info.frame_id <= self.last_processed_id:
                continue

            try:
                # [针对 YOLO26 的修改]
                # 1. 移除了 agnostic_nms (无 NMS 架构)
                # 2. 移除了 iou (不需要 IOU 抑制)
                results = self.model.predict(
                    source=frame_info.frame,
                    imgsz=img_size,
                    conf=conf_thres,     # 仍然需要 conf 过滤
                    max_det=max_det,     # 仍然可以限制最大输出数量
                    classes=target_classes, # GPU 层面直接过滤类别
                    verbose=False,
                    device=0,
                    half=True,
                    stream=True          # 流式推理
                )

                t_inference_done = time.perf_counter()

                detections = np.empty((0, 6), dtype=np.float32)
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        # YOLO26 输出直接就是最终框
                        # 数据格式依然是标准 [x1, y1, x2, y2, conf, cls]
                        detections = r.boxes.data.cpu().numpy()
                    break

                self.world_model.update_detections(
                    detections=detections,
                    frame_id=frame_info.frame_id,
                    t_capture=frame_info.t_cap,  # [修复 2] 这里必须叫 t_capture，与 world_model 保持一致
                    t_done=t_inference_done
                )

                self.last_processed_id = frame_info.frame_id

            except Exception:
                pass