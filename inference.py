# inference.py - Phase 5 DXGI 适配版
#
# 核心职责：
# 1. 消费 FrameBus 中的 Numpy BGR 数组 (来自 dxcam)
# 2. 调用 TensorRT Engine 进行推理
# 3. 将结果推送到 WorldModel

import time
import threading
import numpy as np
from typing import Optional
from ultralytics import YOLO
from config import config
from perception.bus import FrameBus, FrameInfo
from world_model import WorldModel


class InferenceThread(threading.Thread):
    """
    Phase 5 推理线程
    完美适配 DXGI 采集的数据格式 (BGR + Contiguous)。
    """

    def __init__(self, bus: FrameBus, world_model: WorldModel, shutdown_event: threading.Event):
        super().__init__(name="InferenceThread", daemon=True)
        self.bus = bus
        self.world_model = world_model
        self.shutdown_event = shutdown_event
        self.last_processed_id = -1

        # 加载 TensorRT 模型
        model_path = config.getstr("Inference", "model_path", "models/best256.engine")
        img_size = config.getint("General", "capture_size", 256)

        try:
            print(f"[Inference] 正在加载模型: {model_path} ...")
            self.model = YOLO(model_path, task='detect')

            print(f"[Inference] 模型加载成功，正在进行内存热身...")
            # [优化] 无文件热身：直接生成一个全黑的 dummy frame
            # 这样不需要磁盘上有 'warmup.jpg' 文件，且速度更快
            dummy_frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            self.model.predict(
                source=dummy_frame,
                imgsz=img_size,
                device=0,
                half=True,
                verbose=False
            )
            print(f"[Inference] ✅ 热身完成，随时可以开火")

        except Exception as e:
            print(f"[Inference] ❌ 模型加载严重失败: {e}")
            self.model = None

    def run(self):
        if self.model is None:
            print("[Inference] 模型未就绪，线程挂起")
            while not self.shutdown_event.is_set():
                time.sleep(1.0)
            return

        print("[Inference] 🚀 异步推理循环已启动 (DXGI Ready)")

        # 缓存配置，减少循环内的字典查找
        conf_thres = config.getfloat("Inference", "conf_threshold", 0.40)
        iou_thres = config.getfloat("Inference", "iou_threshold", 0.45)
        max_det = config.getint("Inference", "max_det", 20)
        img_size = config.getint("General", "capture_size", 256)

        while not self.shutdown_event.is_set():
            # 1. 获取最新帧
            frame_info: Optional[FrameInfo] = self.bus.get_latest()

            # 过滤旧帧
            if frame_info is None or frame_info.frame_id <= self.last_processed_id:
                # 极短休眠，让出 CPU 给 Capture 线程
                # 因为 DXGI 很快，这里的休眠要尽可能短
                time.sleep(0.0001)
                continue

            try:
                # 2. 执行推理
                # 输入: frame_info.frame 已经是 (256, 256, 3) 的 BGR Numpy 数组
                # 且已经是内存连续的 (Contiguous)，直接喂给 TensorRT 最快

                results = self.model.predict(
                    source=frame_info.frame,
                    imgsz=img_size,
                    conf=conf_thres,
                    iou=iou_thres,
                    max_det=max_det,
                    verbose=False,
                    device=0,  # 强制 GPU
                    half=True  # 开启 FP16 (TensorRT 必须)
                )

                # 3. 记录时间 (Cognition Start)
                t_inference_done = time.perf_counter()

                # 4. 解析结果
                detections = []
                if results and results[0].boxes is not None:
                    # boxes.data: [x1, y1, x2, y2, conf, cls]
                    # 保持 Numpy 格式，拒绝 tolist() 的序列化开销
                    detections = results[0].boxes.data.cpu().numpy()

                # 5. 更新世界模型
                self.world_model.update_detections(
                    detections=detections,
                    frame_id=frame_info.frame_id,
                    t_cap=frame_info.t_cap,
                    t_done=t_inference_done
                )

                self.last_processed_id = frame_info.frame_id

                # 性能监控 (DXGI 模式下这个 Latency 应该非常低)
                if frame_info.frame_id % 100 == 0:
                    latency_ms = (t_inference_done - frame_info.t_cap) * 1000
                    print(
                        f"[Inference] FPS Monitor | Frame ID: {frame_info.frame_id} | Total Latency: {latency_ms:.2f}ms")

            except Exception as e:
                print(f"[Inference] Loop Error: {e}")
                time.sleep(0.01)

        print("[Inference] 线程安全退出")