# utils/recorder.py
import threading
import pickle
import copy
from collections import deque  # <--- 引入双端队列
from typing import List, Optional
from utils.types import InferenceContext


class TraceRecorder:
    """
    Phase 3 决策追踪回放录制器 (循环覆盖版)
    职责：
    1. 在内存中缓存 **最近 N 帧** 的推理上下文 (InferenceContext)
    2. 系统退出时将其序列化到磁盘 (.pkl)

    修改：使用 deque(maxlen=N) 代替 List，自动丢弃旧数据，永远保留最新数据。
    """

    def __init__(self, save_path="debug_trace.pkl", max_frames=5000):
        # 使用 deque 实现固定长度的循环缓冲
        self.buffer = deque(maxlen=max_frames)
        self.save_path = save_path
        self.max_frames = max_frames
        self.enabled = False
        self._lock = threading.Lock()

    def enable(self):
        self.enabled = True
        print(f"[Trace] 🔧 追踪回放系统已开启 (保留最近 {self.max_frames} 帧)")

    def record_frame(self, ctx: InferenceContext):
        """
        深拷贝当前帧的 Context 并存入缓冲区
        """
        if not self.enabled:
            return

        try:
            # 使用 copy.deepcopy 确保数据独立性
            # 注意：Context 中不能包含 ThreadLock 等不可序列化对象
            snapshot = copy.deepcopy(ctx)

            with self._lock:
                # deque 会自动处理溢出，挤掉最旧的一帧
                self.buffer.append(snapshot)
        except Exception as e:
            # 录制不应影响主流程
            pass

    def save_to_disk(self):
        with self._lock:
            if not self.buffer:
                return
            data_to_save = list(self.buffer)  # 转回 list 以便 pickle 兼容

        print(f"[Trace] 正在导出最近 {len(data_to_save)} 帧数据到 {self.save_path}...")
        try:
            with open(self.save_path, "wb") as f:
                pickle.dump(data_to_save, f)
            print("[Trace] ✅ 导出完成")
        except Exception as e:
            print(f"[Trace] ❌ 导出失败: {e}")