import math
from typing import Tuple


class CalibrationState:
    """
    自校准状态容器：管理 '像素 -> 鼠标计数' 的转换比率
    """

    def __init__(self, init_k_x: float = 1.9, init_k_y: float = 1.2):
        self.k_x = init_k_x  # X轴系数 (Counts per Pixel)
        self.k_y = init_k_y  # Y轴系数
        self.learning_rate = 0.05  # 学习率：每次修正 5%
        self.min_k = 0.5  # 熔断下限
        self.max_k = 5.0  # 熔断上限

    def update(self, ai_counts: float, real_pixels: float, axis: str):
        """
        根据反馈更新 K 值
        axis: 'x' or 'y'
        """
        if abs(real_pixels) < 1.0 or abs(ai_counts) < 5.0:
            return

        # 计算这一帧观测到的真实 K 值
        observed_k = ai_counts / real_pixels

        # 异常值过滤 (防止因瞬间甩枪导致的计算错误)
        if not (self.min_k < observed_k < self.max_k):
            return

        # 指数加权移动平均 (EWMA) 更新
        if axis == 'x':
            self.k_x = (1 - self.learning_rate) * self.k_x + self.learning_rate * observed_k
            # print(f"[Strategy] Auto-Tune X: {self.k_x:.3f} (Obs: {observed_k:.3f})")
        else:
            self.k_y = (1 - self.learning_rate) * self.k_y + self.learning_rate * observed_k
            # print(f"[Strategy] Auto-Tune Y: {self.k_y:.3f} (Obs: {observed_k:.3f})")


class ValorantStrategy:
    def __init__(self):
        # 初始化校准状态
        # 初始值沿用你之前的经验参数: X=1.9, Y=1.2
        self.calib = CalibrationState(init_k_x=1.9, init_k_y=1.2)

        # 记录上一帧的计算意图，用于 feedback 匹配（可选）
        self.last_intent_x = 0
        self.last_intent_y = 0

    def calculate_mouse_move(self, dx: float, dy: float) -> Tuple[int, int]:
        """
        Phase 3 核心：纯几何映射
        输入：预测后的像素差 (dx, dy)
        输出：硬件鼠标计数 (counts_x, counts_y)
        """

        # 1. 应用自适应 K 值
        # Counts = Pixels * K
        raw_x = dx * self.calib.k_x
        raw_y = dy * self.calib.k_y

        # 2. 取整 (硬件限制)
        out_x = int(raw_x)
        out_y = int(raw_y)

        self.last_intent_x = out_x
        self.last_intent_y = out_y

        return out_x, out_y

    def feedback_update(self,
                        history_ai_counts_x: int, actual_pixel_dx: float,
                        history_ai_counts_y: int, actual_pixel_dy: float):
        """
        Phase 3 自校准接口
        由 World Model 或 Main Loop 调用

        参数:
          history_ai_counts: RingBuffer 中记录的过去几帧 AI 发出的总指令
          actual_pixel_dx:   画面上实际发生的像素位移
        """
        # 分轴校准
        self.calib.update(history_ai_counts_x, actual_pixel_dx, 'x')
        self.calib.update(history_ai_counts_y, actual_pixel_dy, 'y')