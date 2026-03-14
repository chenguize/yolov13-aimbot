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
        max_det = config.getint("Inference", "max_det", 20)
        img_size = config.getint("General", "capture_size", 256)

        # ⚠️ 记得把这里的 [7] 注释掉或者改成你的目标类别，否则依然会被强行过滤！
        # target_classes = [7]

        # 1️⃣ 新增：用于控制台打印的计时器
        last_print_time = 0.0

        while not self.shutdown_event.is_set():
            if not self.frame_ready_event.wait(timeout=0.005):
                continue

            self.frame_ready_event.clear()
            frame_info = self.bus.get_latest()

            if frame_info is None or frame_info.frame_id <= self.last_processed_id:
                continue

            try:
                # 🌟 修复点 1：将 DXCam 的 RGB 转换为 YOLO 认识的 BGR，并强制内存连续以满足 TensorRT 胃口
                bgr_frame = np.ascontiguousarray(frame_info.frame[..., ::-1])

                # ⚠️ 将 source 替换为我们转换好的 bgr_frame
                results = self.model.predict(
                    source=bgr_frame,
                    imgsz=img_size,
                    conf=conf_thres,
                    max_det=max_det,
                    verbose=False,
                    device=0,
                    half=True,
                    stream=True
                )

                t_inference_done = time.perf_counter()

                detections = np.empty((0, 6), dtype=np.float32)
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        detections = r.boxes.data.cpu().numpy()
                    break

                now = time.perf_counter()
                if now - last_print_time > 1.0:
                    if len(detections) > 0:
                        best_conf = detections[0][4]
                        best_cls = int(detections[0][5])
                        print(
                            f"[Inference 🔍] 状态: 识别中 | 视野(256x256)内发现 {len(detections)} 个目标 | 最优: 类别 {best_cls}, 置信度 {best_conf:.2f}")
                    else:
                        print(f"[Inference 🔍] 状态: 识别中 | 视野(256x256)内没有任何符合的目标...")
                    last_print_time = now

                self.world_model.update_detections(
                    detections=detections,
                    frame_id=frame_info.frame_id,
                    t_capture=frame_info.t_cap,
                    t_done=t_inference_done
                )

                self.last_processed_id = frame_info.frame_id

            except Exception as e:
                # 🌟 修复点 2：绝不吞掉报错，大声喊出来！
                print(f"[Inference 🔍] ⚠️ 识别线程崩溃: {e}")