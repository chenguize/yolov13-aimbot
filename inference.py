# inference.py - Phase 3 异步推理
#
# 核心职责：
# 1. 加载 TensorRT Engine
# 2. 生产结构 B (Inference Context) 的原始数据
# 3. 将 (Detections + t_cap + t_done) 推送给 WorldModel

import time
import threading
from typing import List, Optional
from ultralytics import YOLO
from config import config
from perception.bus import FrameBus, FrameInfo
from world_model import WorldModel  # 类型提示用


class InferenceThread(threading.Thread):
    """
    Phase 3 推理线程
    只负责 '看' (Perception)，不负责 '想' (Cognition)。

    流程：
    FrameBus -> YOLO -> Raw Detections -> WorldModel.update()
    """

    def __init__(self, bus: FrameBus, world_model: WorldModel, shutdown_event: threading.Event):
        super().__init__(name="InferenceThread", daemon=True)
        self.bus = bus
        self.world_model = world_model
        self.shutdown_event = shutdown_event
        self.last_processed_id = -1

        # 加载 TensorRT 导出模型 (best256.engine)
        # Ultralytics 会自动识别 .engine 后缀并调用 TensorRT 后端
        model_path = config.getstr("Inference", "model_path", "models/best256.engine")
        try:
            print(f"[Inference] 正在加载模型: {model_path} ...")
            self.model = YOLO(model_path, task='detect')
            print(f"[Inference] 模型加载成功，准备热身...")
            # 预热一次，避免首帧延迟
            # self.model.predict(source="models/warmup.jpg", imgsz=256, verbose=False)
            # (如果没有图片可跳过，第一次推理会自动预热)
        except Exception as e:
            print(f"[Inference] ❌ 模型加载严重失败: {e}")
            self.model = None

    def run(self):
        if self.model is None:
            print("[Inference] 模型未就绪，线程挂起")
            while not self.shutdown_event.is_set():
                time.sleep(1.0)
            return

        print("[Inference] 🚀 异步推理循环已启动")

        # 读取配置 (避免在循环中重复 get)
        conf_thres = config.getfloat("Inference", "conf_threshold", 0.40)
        iou_thres = config.getfloat("Inference", "iou_threshold", 0.45)
        max_det = config.getint("Inference", "max_det", 20)

        while not self.shutdown_event.is_set():
            # 1. 从总线获取最新帧 (Non-blocking check usually handled by get_latest logic)
            frame_info: Optional[FrameInfo] = self.bus.get_latest()

            # 过滤旧帧或空帧
            if frame_info is None or frame_info.frame_id <= self.last_processed_id:
                time.sleep(0.0005)  # 极短休眠让出 CPU
                continue

            try:
                # 2. 执行推理
                # 注意：FrameInfo 携带了 t_cap (截图时间)，这是最重要的时间锚点

                # Ultralytics Predict
                results = self.model.predict(
                    source=frame_info.frame,
                    imgsz=256,
                    conf=conf_thres,
                    iou=iou_thres,
                    max_det=max_det,
                    verbose=False,
                    device=0,  # 强制 GPU
                    half=True  # 开启 FP16 (如果 Engine 支持)
                )

                # 3. 记录推理完成时间 (用于 Structure B)
                # 这代表了"感知结束，准备进入认知"的时刻
                t_inference_done = time.perf_counter()

                # 4. 提取原始检测框
                detections = []
                if results and results[0].boxes is not None:
                    # boxes.data 格式: [x1, y1, x2, y2, conf, cls]
                    detections = results[0].boxes.data.cpu().numpy().tolist()

                # 5. 推送给 WorldModel (状态仲裁者)
                # Inference 层只管传数据，不管 RingBuffer 对冲，那是 WorldModel 的事
                self.world_model.update_detections(
                    detections=detections,
                    frame_id=frame_info.frame_id,
                    t_cap=frame_info.t_cap,  # 截图时间 (用于计算 Age)
                    t_done=t_inference_done  # 推理结束时间 (用于 Debug/Latency)
                )

                self.last_processed_id = frame_info.frame_id

                # 性能监控 (每 100 帧显示一次)
                if frame_info.frame_id % 100 == 0:
                    latency_ms = (t_inference_done - frame_info.t_cap) * 1000
                    print(
                        f"[Inference] FPS Monitor | Frame ID: {frame_info.frame_id} | Total Latency: {latency_ms:.2f}ms")

            except Exception as e:
                print(f"[Inference] Loop Error: {e}")
                time.sleep(0.01)

        print("[Inference] 线程安全退出")