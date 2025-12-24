# inference.py - YOLOv13 异步推理线程（使用 Ultralytics 官方封装 + TensorRT engine）
#
# 核心职责：
# 1. 加载导出的 TensorRT engine（best256.engine）
# 2. 异步从 bus 取帧，进行推理
# 3. 直接输出 NMS 后的 detections 给 world_model
# 4. 极简、高性能、无底层手动操作

import time
import threading
from typing import List, Optional
from ultralytics import YOLO
from config import config
from perception.bus import FrameBus, FrameInfo
from world_model import WorldModel


class InferenceThread(threading.Thread):
    """
    YOLOv13 异步推理线程 - Ultralytics 官方封装版
    - 直接使用 YOLO('best256.engine') 进行推理
    - 自动完成预处理、NMS、后处理
    - 输出 detections 给 world_model
    """

    def __init__(self, bus: FrameBus, world_model: WorldModel, shutdown_event: threading.Event):
        super().__init__(name="InferenceThread", daemon=True)
        self.bus = bus
        self.world_model = world_model
        self.shutdown_event = shutdown_event
        self.last_processed_id = -1

        # 加载导出的 TensorRT engine
        model_path = config.getstr("Inference", "model_path", "models/best256.engine")
        try:
            self.model = YOLO(model_path)
            print(f"[Inference] YOLOv13 TensorRT engine 加载成功: {model_path}")
        except Exception as e:
            print(f"[Inference] 模型加载失败: {e}")
            self.model = None

    def run(self):
        if self.model is None:
            print("[Inference] 模型不可用，线程空转")
            while not self.shutdown_event.is_set():
                time.sleep(1.0)
            return

        print("[Inference] YOLOv13 Ultralytics 推理线程启动")

        while not self.shutdown_event.is_set():
            frame_info: Optional[FrameInfo] = self.bus.get_latest()

            if frame_info is None or frame_info.frame_id <= self.last_processed_id:
                time.sleep(0.001)
                continue

            try:
                start = time.perf_counter()

                # Ultralytics 一行完成所有：预处理 + 推理 + NMS + 后处理
                results = self.model.predict(
                    source=frame_info.frame,               # 直接传原始帧
                    imgsz=256,                             # 固定 256x256 输入
                    conf=config.getfloat("Inference", "conf_threshold", 0.38),
                    iou=config.getfloat("Inference", "iou_threshold", 0.45),
                    max_det=config.getint("Inference", "max_det", 20),
                    verbose=False                          # 关闭冗长日志
                )

                # 提取 detections（[x1,y1,x2,y2,conf,cls] 格式）
                detections = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes.data.cpu().numpy()  # [N, 6]
                    detections = boxes.tolist()

                inference_time = time.perf_counter() - start

                # 传递给 world_model（包含采集时刻信息）
                self.world_model.update_detections(
                    detections=detections,
                    frame_id=frame_info.frame_id,
                    timestamp=frame_info.timestamp
                )

                self.last_processed_id = frame_info.frame_id

                # 性能监控（每 60 帧打印一次）
                if frame_info.frame_id % 60 == 0:
                    print(f"[Inference] Frame {frame_info.frame_id} | {inference_time*1000:.1f}ms")

            except Exception as e:
                print(f"[Inference] 单帧异常: {e}")

            time.sleep(0.0005)  # 轻微休眠，平衡 CPU

        print("[Inference] 线程退出")