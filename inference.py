# inference.py
import time
import threading
import numpy as np
from typing import List, Optional, Tuple
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from config import config
from perception.bus import FrameBus, FrameInfo
from world_model import WorldModel
from utils.helpers import letterbox  # 假设已实现 letterbox 函数


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTEngine:
    """TensorRT 引擎封装 - 专门用于 best256.engine"""

    def __init__(self):
        model_path = config.getstr("Inference", "model_path", "models/best256.engine")
        self.logger = TRT_LOGGER

        # 加载 engine
        with open(model_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {model_path}")

        self.context = self.engine.create_execution_context()

        # 输入输出绑定
        self.input_shape = self.engine.get_tensor_shape(0)  # 假设第一个是 input
        self.output_shape = self.engine.get_tensor_shape(1)  # 假设第二个是 output

        self.input_size = trt.volume(self.input_shape) * self.engine.max_batch_size
        self.output_size = trt.volume(self.output_shape) * self.engine.max_batch_size

        self.host_in = cuda.pagelocked_empty(self.input_size, dtype=np.float32)
        self.host_out = cuda.pagelocked_empty(self.output_size, dtype=np.float32)
        self.device_in = cuda.mem_alloc(self.host_in.nbytes)
        self.device_out = cuda.mem_alloc(self.host_out.nbytes)

        self.bindings = [int(self.device_in), int(self.device_out)]

        print(f"[TensorRT] Engine loaded: {model_path}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Output shape: {self.output_shape}")

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """单帧推理"""
        # 预处理：letterbox 到 256x256
        img, ratio, pad = letterbox(frame, (256, 256))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img[None])  # [1,3,256,256]

        # 拷贝到 device
        np.copyto(self.host_in, img.ravel())
        cuda.memcpy_htod(self.device_in, self.host_in)

        # 执行推理
        self.context.execute_v2(self.bindings)

        # 获取输出
        cuda.memcpy_dtoh(self.host_out, self.device_out)
        return self.host_out.reshape(self.output_shape)


class InferenceThread(threading.Thread):
    """
    YOLOv13 TensorRT 异步推理线程
    输入：bus 中的最新帧（采集时刻数据）
    输出：detections → world_model
    """

    def __init__(self, bus: FrameBus, world_model: WorldModel, shutdown_event: threading.Event):
        super().__init__(name="InferenceThread", daemon=True)
        self.bus = bus
        self.world_model = world_model
        self.shutdown_event = shutdown_event
        self.last_processed_id = -1

        try:
            self.engine = TensorRTEngine()
        except Exception as e:
            print(f"[Inference] 引擎加载失败: {e}")
            self.engine = None

    def run(self):
        if self.engine is None:
            print("[Inference] TensorRT 引擎不可用，线程空转")
            while not self.shutdown_event.is_set():
                time.sleep(1.0)
            return

        print("[Inference] YOLOv13 TensorRT 推理线程启动")

        while not self.shutdown_event.is_set():
            frame_info: Optional[FrameInfo] = self.bus.get_latest()

            if frame_info is None or frame_info.frame_id <= self.last_processed_id:
                time.sleep(0.001)
                continue

            try:
                start = time.perf_counter()

                # 推理
                raw_output = self.engine.infer(frame_info.frame)

                # 简单后处理（实际项目建议把 NMS 移到 world_model）
                detections = self._postprocess(raw_output)

                inference_time = time.perf_counter() - start

                # 传递给 world_model（包含采集时刻信息）
                self.world_model.update_detections(
                    detections=detections,
                    frame_id=frame_info.frame_id,
                    timestamp=frame_info.timestamp
                )

                self.last_processed_id = frame_info.frame_id

                if frame_info.frame_id % 60 == 0:
                    print(f"[Inference] #{frame_info.frame_id} | {inference_time*1000:.1f}ms")

            except Exception as e:
                print(f"[Inference] 单帧异常: {e}")

            time.sleep(0.0005)

        print("[Inference] 线程退出")

    def _postprocess(self, output: np.ndarray) -> List[List[float]]:
        """完整后处理：置信度过滤 + NMS"""
        boxes = []
        scores = []
        classes = []

        conf_thres = config.getfloat("Inference", "conf_threshold", 0.38)
        iou_thres = config.getfloat("Inference", "iou_threshold", 0.45)
        max_det = config.getint("Inference", "max_det", 20)

        for pred in output[0]:
            conf = pred[4]
            if conf < conf_thres:
                continue
            x1, y1, x2, y2 = pred[:4]
            cls_scores = pred[5:]
            cls_id = int(np.argmax(cls_scores))
            cls_conf = cls_scores[cls_id]

            boxes.append([x1, y1, x2, y2])
            scores.append(conf * cls_conf)  # 类别置信度 × 目标置信度
            classes.append(cls_id)

        if not boxes:
            return []

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        keep = nms(boxes_tensor, scores_tensor, iou_threshold=iou_thres)

        detections = []
        for idx in keep[:max_det]:
            box = boxes[idx]
            detections.append([
                float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                float(scores[idx]), int(classes[idx])
            ])

        return detections