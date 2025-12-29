# utils/recorder.py
import threading
import pickle
import copy
from typing import List, Optional
from utils.types import InferenceContext


class TraceRecorder:
    """
    Phase 3 决策追踪回放录制器
    职责：
    1. 在内存中缓存最近 N 帧的推理上下文 (InferenceContext)
    2. 系统退出时将其序列化到磁盘 (.pkl)
    """

    def __init__(self, save_path="debug_trace.pkl", max_frames=5000):
        self.buffer: List[InferenceContext] = []
        self.save_path = save_path
        self.max_frames = max_frames
        self.enabled = False
        self._lock = threading.Lock()

    def enable(self):
        self.enabled = True
        print(f"[Trace] 🔧 追踪回放系统已开启 (Buffer: {self.max_frames} frames)")

    def record_frame(self, ctx: InferenceContext):
        """
        深拷贝当前帧的 Context 并存入缓冲区
        注意：必须深拷贝，因为 main loop 会复用 ctx 对象
        """
        if not self.enabled:
            return

        # 简单的内存保护
        if len(self.buffer) >= self.max_frames:
            return

        try:
            # 使用 copy.deepcopy 确保数据独立性
            # 注意：Context 中不能包含 ThreadLock 等不可序列化对象
            snapshot = copy.deepcopy(ctx)

            with self._lock:
                self.buffer.append(snapshot)
        except Exception as e:
            # 录制不应影响主流程，静默失败或打印
            # print(f"[Trace] Record error: {e}")
            pass

    def save_to_disk(self):
        if not self.buffer:
            return

        print(f"[Trace] 正在导出 {len(self.buffer)} 帧数据到 {self.save_path}...")
        try:
            with open(self.save_path, "wb") as f:
                pickle.dump(self.buffer, f)
            print("[Trace] ✅ 导出完成")
        except Exception as e:
            print(f"[Trace] ❌ 导出失败: {e}")