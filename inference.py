# inference.py
"""
YOLOv13 / YOLO系列 TensorRT 异步推理线程
使用 models/best256.engine （256x256 输入模型）
"""

import time
import threading
import numpy as np
from typing import List, Optional, Tuple, Any

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from perception.bus import FrameBus, FrameInfo
from world_model import WorldModel
from config import config
from utils.helpers import letterbox, scale_boxes  # 假设你有这些辅助函数

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTEngine:
    """简单的 TensorRT 推理引擎封装"""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.logger = TRT_LOGGER

        # 加载 engine
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # 获取输入/输出绑定信息
        self.input_names = []
        self.output_names = []
        self.bindings = []
        self.host_buffers = []
        self.device_buffers = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            host_buf = cuda.pagelocked_empty(size, dtype)
            device_buf = cuda.mem_alloc(host_buf.nbytes)

            self.bindings.append(int(device_buf))
            self.host_buffers.append(host_buf)
            self.device_buffers.append(device_buf)

            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_names.append(binding)
            else:
                self.output_names.append(binding)

        print(f"[TRT] Engine loaded: {engine_path}")
        print(f"  Inputs: {self.input_names}")
        print(f"  Outputs: {self.output_names}")

    def infer(self, img: np.ndarray) -> np.ndarray:
        """执行单张图像推理"""
        # 预处理：letterbox resize to 256x256
        img_processed, ratio, pad = letterbox(img, new_shape=(256, 256), auto=False, scaleFill=True)
        img_processed = img_processed.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img_processed = np.ascontiguousarray(img_processed[None])  # add batch dim

        # 拷贝到 device
        np.copyto(self.host_buffers[0], img_processed.ravel())
        cuda.memcpy_htod(self.device_buffers[0], self.host_buffers[0])

        # 执行推理
        self.context.execute_v2(self.bindings)

        # 获取输出
        output = np.empty(trt.volume(self.engine.get_tensor_shape(self.output_names[0])), dtype=np.float32)
        cuda.memcpy_dtoh(output, self.device_buffers[1])  # 假设第二个 binding 是输出

        return output.reshape(self.engine.get_tensor_shape(self.output_names[0]))


class InferenceThread(threading.Thread):
    """
    YOLO 异步推理线程 - 使用 TensorRT engine (best256.engine)
    输入：256×256 BGR图像
    输出：直接给 world_model 的原始预测结果
    """

    def __init__(self, bus: FrameBus, world_model: WorldModel, shutdown_event: threading.Event):
        super().__init__(name="YOLOv13-Inference", daemon=False)
        self.bus = bus
        self.world_model = world_model
        self.shutdown_event = shutdown_event

        self.last_processed_frame_id = -1
        self.engine = None

        # 加载模型
        try:
            model_path = str(config.get("Inference", "model_path", "models/best256.engine"))
            self.engine = TRTEngine(model_path)
        except Exception as e:
            print(f"[Inference] 严重错误：无法加载 TensorRT engine\n{e}")
            self.engine = None

    def run(self):
        if self.engine is None:
            print("[Inference] 推理引擎加载失败，线程将空转")
            while not self.shutdown_event.is_set():
                time.sleep(0.5)
            return

        print("[Inference] YOLOv13 (TensorRT) 推理线程启动 - 模型: best256.engine")

        while not self.shutdown_event.is_set():
            frame_info: Optional[FrameInfo] = self.bus.get_latest()

            if frame_info is None or frame_info.id <= self.last_processed_frame_id:
                time.sleep(0.0012)
                continue

            try:
                start_time = time.perf_counter()

                # 执行推理
                raw_output = self.engine.infer(frame_info.frame)

                # 后处理（根据你的模型输出格式自行调整）
                # 假设是 YOLOv5/v8/v10/v13 常见的 [batch, num_boxes, 6+(classes)] 格式
                # 这里只做最基础的过滤，实际建议移到 world_model 里做 NMS/过滤
                detections = self.postprocess(raw_output)

                inference_time = (time.perf_counter() - start_time) * 1000

                self.world_model.update_detections(
                    detections=detections,
                    frame_id=frame_info.id,
                    timestamp=frame_info.timestamp,
                    inference_ms=inference_time,
                    mouse_center=getattr(frame_info, 'mouse_center', None)
                )

                self.last_processed_frame_id = frame_info.id

                # 可选：打印性能
                if frame_info.id % 30 == 0:
                    print(f"[Inference] #{frame_info.id} | {inference_time:.1f}ms")

            except Exception as e:
                print(f"[Inference] 单帧处理异常: {e}")

            time.sleep(0.0004)  # 尽量跑满，但留点余量

        print("[Inference] YOLOv13 推理线程正常退出")

    def postprocess(self, output: np.ndarray) -> List[Tuple[float, float, float, float, float, int]]:
        """最简后处理 - 实际项目建议把 NMS 等移到 world_model"""
        # 假设输出是 [1, 8400, 85] 之类的 (根据你的模型调整)
        # 这里只做极简示例
        detections = []

        conf_thres = config.getfloat("Inference", "conf_threshold", 0.38)

        for pred in output[0]:  # 遍历所有预测
            conf = pred[4]  # 置信度位置根据模型不同可能不同
            if conf < conf_thres:
                continue

            x1, y1, x2, y2 = pred[:4]
            cls_id = int(np.argmax(pred[5:]))  # 多分类情况

            detections.append((float(x1), float(y1), float(x2), float(y2), float(conf), cls_id))

        return detections