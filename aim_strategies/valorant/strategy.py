import time
import math
from typing import Tuple


class ValorantStrategy:
    def __init__(self):
        # =================================================================
        # [V32 逻辑修复版]
        # 修复了在静止画面下因去重逻辑导致的“假死”不瞄准问题
        # =================================================================

        # [自适应参数]
        self.scale_x = 1.9
        self.scale_y = 1.2
        self.learning_rate = 0.25

        # [延迟抑制]
        self.latency_wait = 0.15

        # 阈值
        self.stop_threshold = 2.0
        self.max_step = 800.0

        # =================================================================

        # 状态记录
        self.prev_raw_x = -1.0
        self.prev_raw_y = -1.0

        # 回合制状态
        self.suppress_until = 0.0
        self.last_cmd_x = 0
        self.last_cmd_y = 0
        self.start_error_x = 0.0
        self.start_error_y = 0.0
        self.is_waiting = False

        # 统计
        self.is_locking = False
        self.lock_start_time = 0.0
        self.lock_frame_count = 0

    def reset(self):
        self.prev_raw_x = -1.0
        self.prev_raw_y = -1.0
        self.is_waiting = False
        self.suppress_until = 0.0
        self.is_locking = False

    def calculate_mouse_move(
            self,
            target_x: float,
            target_y: float,
            center_x: float,
            center_y: float,
            dt: float
    ) -> Tuple[int, int]:

        current_time = time.time()

        # ---------------------------------------------------------
        # 1. [最高优先级] 检查是否需要从休眠中唤醒
        # ---------------------------------------------------------
        if self.is_waiting:
            # 如果还在冷却期，直接返回0，不往下走
            if current_time < self.suppress_until:
                return 0, 0

            # 时间到了！强制唤醒
            self.is_waiting = False

            # [关键修复] 唤醒时，强制让去重逻辑失效
            # 这样即使画面完全没变，我们也能根据“刚才那一枪没打准”的事实继续修正
            self.prev_raw_x = -9999.0

            # 验收成果 & 调参
            curr_error_x = target_x - center_x
            curr_error_y = target_y - center_y
            actual_move_x = self.start_error_x - curr_error_x
            actual_move_y = self.start_error_y - curr_error_y
            self._tune_parameter(actual_move_x, actual_move_y)

        # ---------------------------------------------------------
        # 2. 帧去重 (防止同一帧重复计算)
        # ---------------------------------------------------------
        if (abs(target_x - self.prev_raw_x) < 1e-5 and
                abs(target_y - self.prev_raw_y) < 1e-5):
            return 0, 0

        self.prev_raw_x = target_x
        self.prev_raw_y = target_y

        # ---------------------------------------------------------
        # 3. 正常计算逻辑
        # ---------------------------------------------------------
        curr_error_x = target_x - center_x
        curr_error_y = target_y - center_y
        dist = math.hypot(curr_error_x, curr_error_y)

        # 统计
        if dist > 10.0 and not self.is_locking:
            self.is_locking = True
            self.lock_start_time = current_time
            self.lock_frame_count = 0

        if self.is_locking:
            self.lock_frame_count += 1

        # 停止判定
        if abs(curr_error_x) < self.stop_threshold and abs(curr_error_y) < self.stop_threshold:
            if self.is_locking:
                duration = (current_time - self.lock_start_time) * 1000
                print(
                    f"✅ [锁定] 耗时:{duration:.0f}ms | 帧数:{self.lock_frame_count} | 参数:({self.scale_x:.2f}, {self.scale_y:.2f})")
                self.is_locking = False
            return 0, 0

        # 计算移动
        move_x = curr_error_x * self.scale_x
        move_y = curr_error_y * self.scale_y

        int_move_x = int(move_x)
        int_move_y = int(move_y)

        # 动力保底
        if int_move_x == 0 and abs(curr_error_x) > self.stop_threshold:
            int_move_x = 1 if curr_error_x > 0 else -1
        if int_move_y == 0 and abs(curr_error_y) > self.stop_threshold:
            int_move_y = 1 if curr_error_y > 0 else -1

        # 安全限制
        int_move_x = max(-self.max_step, min(self.max_step, int_move_x))
        int_move_y = max(-self.max_step, min(self.max_step, int_move_y))

        # 发送指令并进入休眠
        if int_move_x != 0 or int_move_y != 0:
            self.start_error_x = curr_error_x
            self.start_error_y = curr_error_y
            self.last_cmd_x = int_move_x
            self.last_cmd_y = int_move_y

            # 大移动才休眠，小移动(微调)不休眠以免卡顿
            if abs(int_move_x) > 5 or abs(int_move_y) > 5:
                self.is_waiting = True
                self.suppress_until = current_time + self.latency_wait

        return int_move_x, int_move_y

    def _tune_parameter(self, actual_x, actual_y):
        # 调参逻辑 (同前)
        if abs(self.last_cmd_x) > 10 and abs(actual_x) > 2:
            real_ratio = self.last_cmd_x / actual_x
            if 0.5 < real_ratio < 5.0:
                old = self.scale_x
                self.scale_x = (1 - self.learning_rate) * self.scale_x + self.learning_rate * real_ratio
                print(f"📊 [调参X] 预期{self.last_cmd_x} 实移{actual_x:.1f} | 参数: {old:.2f} -> {self.scale_x:.2f}")

        if abs(self.last_cmd_y) > 10 and abs(actual_y) > 2:
            real_ratio = self.last_cmd_y / actual_y
            if 0.5 < real_ratio < 5.0:
                old = self.scale_y
                self.scale_y = (1 - self.learning_rate) * self.scale_y + self.learning_rate * real_ratio
                print(f"📊 [调参Y] 预期{self.last_cmd_y} 实移{actual_y:.1f} | 参数: {old:.2f} -> {self.scale_y:.2f}")