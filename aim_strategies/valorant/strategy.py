import math
from typing import Tuple
import numpy as np
from config import config
from utils.logger import get_logger

logger = get_logger("AimStrategy")


class CalibrationState:
    def __init__(self, init_k_x: float, init_k_y: float):
        # 初始 k 由调用方根据物理公式动态计算后传入
        self.k_x = init_k_x
        self.k_y = init_k_y
        self.learning_rate = 0.05
        self.min_k = 0.1
        self.max_k = 20.0
        self.enabled = config.getbool("Strategy", "enable_auto_calibration", False)
        self.change_threshold = 0.05

    def update(self, ai_counts: float, real_pixels: float, axis: str):
        if not self.enabled: return
        if abs(real_pixels) < 1.0 or abs(ai_counts) < 5.0: return
        observed_k = ai_counts / real_pixels
        if observed_k <= 0 or not (self.min_k < observed_k < self.max_k): return
        current_k = self.k_x if axis == 'x' else self.k_y
        if abs(observed_k - current_k) / current_k < self.change_threshold: return

        new_k = (1 - self.learning_rate) * current_k + self.learning_rate * observed_k
        if axis == 'x':
            self.k_x = new_k
        else:
            self.k_y = new_k


# Game yaw in degrees per mouse count at sensitivity 1.0. SendInput emits
# counts, so count-to-view-angle mapping is independent of physical mouse DPI.
_YAW_DEG_PER_COUNT: dict[str, float] = {
    "valorant": 0.070,
    "cs2": 0.022,
    "csgo": 0.022,
    "overwatch2": 0.0066,
    "apex": 0.022,
    "default": 0.070,
}


def _degrees_per_count(game: str, sens: float) -> float:
    safe_sens = sens if 0.0 < sens <= 1000.0 else 1.0
    yaw = _YAW_DEG_PER_COUNT.get(game.lower(), _YAW_DEG_PER_COUNT["default"])
    return yaw * safe_sens


def _cm_per_360(game: str, sens: float, dpi: float) -> float:
    """Diagnostic physical distance; not used for synthetic count mapping."""
    safe_dpi = dpi if dpi > 0.0 else 800.0
    return 2.54 * 360.0 / (_degrees_per_count(game, sens) * safe_dpi)


class ValorantStrategy:
    """
    Tier S+++ │ 纯物理映射引擎 (UE5 Compatible)
    已剔除所有动态缩放私货，确保正逆变换绝对对称，保证 Kalman Ego-motion 精准。
    """

    def __init__(self):
        # ── 物理输入：硬件层 + 游戏内 sens ────────────────────────────────────
        self.mouse_dpi   = config.getfloat("Input",   "mouse_dpi",      800.0)
        self.in_game_sens = config.getfloat("Input",  "in_game_sens",    0.4)
        self.k_yx_ratio  = config.getfloat("Input",   "k_yx_ratio",      1.0)
        self.game        = config.getstr("Game",      "current_game",    "valorant")
        self.cm_360      = _cm_per_360(self.game, self.in_game_sens, self.mouse_dpi)

        # ── 屏幕/FOV ──────────────────────────────────────────────────────────
        # config.ini [Hardware].screen_width / screen_height 已由 Config.get 自动处理：
        #   0 / "auto" → 替换为 Windows 主显示器自动检测值
        #   显式值    → 直接用（用于多显示器/拉伸场景下覆盖桌面分辨率）
        self.game_fov     = config.getfloat("Hardware", "game_fov",       103.0)
        self.screen_width  = config.getfloat("Hardware", "screen_width",  0.0)
        self.screen_height = config.getfloat("Hardware", "screen_height", 0.0)
        # 启动日志：分辨率 + 来源（防静默失败 / 防 config 与 OS 不一致）
        try:
            from utils.screen import detect_dpi_scale
            dpi_scale = detect_dpi_scale()
        except Exception:
            dpi_scale = 1.0
        # 判断是否走自动检测（Config 记录了哪些 key 被替换）
        if config.was_replaced("Hardware", "screen_width") or config.was_replaced("Hardware", "screen_height"):
            det_w, det_h = config._detected_screen[0], config._detected_screen[1]
            screen_source = f"Windows 自动检测 → {det_w}×{det_h}（config.ini 写 0/auto）"
        else:
            screen_source = "config.ini 显式"
        self.focal_length = (self.screen_width / 2) / math.tan(math.radians(self.game_fov / 2))

        # ── 动态 k_factor：count/pixel，由游戏 yaw × sens × 投影焦距推导 ───
        k = self._compute_k_factor()
        kx = k
        ky = k * self.k_yx_ratio
        self.calib = CalibrationState(init_k_x=kx, init_k_y=ky)
        logger.info(
            f"AimStrategy 启动摘要:\n"
            f"   屏幕分辨率 = {int(self.screen_width)}×{int(self.screen_height)} "
            f"(来源: {screen_source}, DPI scale: {dpi_scale:.2f})\n"
            f"   游戏 FOV    = {self.game_fov}°\n"
            f"   鼠标 DPI    = {self.mouse_dpi}  游戏内 sens = {self.in_game_sens}  "
            f"cm_360 = {self.cm_360:.1f} cm\n"
            f"   动态 k_factor: k_x={kx:.3f}, k_y={ky:.3f} (count/pixel)"
        )

        # 开则正逆均为 1:1，便于桌面/静态图测检测框，不经 FOV/k
        self.bypass_mapping = config.getbool("Strategy", "bypass_strategy_mapping", False)
        if self.bypass_mapping:
            logger.warning("AimStrategy: bypass_strategy_mapping 已开启（1px≈1count，仅调试）")

    def _compute_k_factor(self) -> float:
        """
        Exact center Jacobian for perspective projection:

            pixels/count = focal_length * radians/count
            counts/pixel = 1 / (focal_length * radians/count)

        DPI affects physical hand travel only. Including it here makes the
        synthetic SendInput response disagree with the real game plant.
        """
        radians_per_count = math.radians(
            _degrees_per_count(self.game, self.in_game_sens)
        )
        return 1.0 / max(self.focal_length * radians_per_count, 1e-9)

    @property
    def k_factor_x(self) -> float:
        return float(self.calib.k_x)

    @property
    def k_factor_y(self) -> float:
        return float(self.calib.k_y)

    def apply_fov_distortion(self, dx: float, dy: float) -> Tuple[float, float]:
        angle_x = math.atan(dx / self.focal_length)
        angle_y = math.atan(dy / self.focal_length)
        return angle_x * self.focal_length, angle_y * self.focal_length

    def _velocity_jacobian(self, px_x: float, px_y: float) -> Tuple[float, float]:
        """
        速度变换的局部 Jacobian：d(count)/d(px) 在 (px_x, px_y) 处。
        位置正变换 count(px) = atan(px/f) * f * k → d(count)/d(px) = k / (1 + (px/f)²)
        所以 Jacobian (相对 k 的归一化值) = 1 / (1 + (px/f)²)
        px=0 时 jac=1.0；px=±f 时 jac=0.5（边缘速度被压缩一半）。
        """
        # 防止 px 远超焦距导致奇异（实际不会发生，但保护）
        rx = px_x / self.focal_length
        ry = px_y / self.focal_length
        return 1.0 / (1.0 + rx * rx), 1.0 / (1.0 + ry * ry)

    def calculate_mouse_move(self, dx: float, dy: float, bbox_w: float | None = None) -> Tuple[float, float]:
        """位置正变换：像素偏移 → 物理 Count (含 FOV 非线性)"""
        if self.bypass_mapping:
            return float(dx), float(dy)
        c_dx, c_dy = self.apply_fov_distortion(dx, dy)
        return c_dx * self.calib.k_x, c_dy * self.calib.k_y

    def calculate_velocity_move(
        self,
        vx: float,
        vy: float,
        px_x: float = 0.0,
        px_y: float = 0.0,
        bbox_w: float | None = None,
    ) -> Tuple[float, float]:
        """
        速度正变换：px/s → counts/s (FOV Jacobian 一阶近似)
        P0 修复：原实现 vx*k_x 与位置变换 calculate_mouse_move (atan 非线性) 不对称，
        导致屏幕边缘速度被系统性高估、ego_pos_px 长期漂移。
        正确公式：d(count)/dt = d(count)/d(px) * d(px)/dt = k / (1+(px/f)²) * v_px

        Parameters
        ----------
        vx, vy : 目标速度 (px/s)
        px_x, px_y : 速度锚点在屏幕上相对准星的位置 (px)，默认 0 即屏幕中心
                     （等价旧实现，向后兼容）
        """
        if self.bypass_mapping:
            return float(vx), float(vy)
        jac_x, jac_y = self._velocity_jacobian(px_x, px_y)
        return vx * self.calib.k_x * jac_x, vy * self.calib.k_y * jac_y

    def reverse_map(self, counts_x: float, counts_y: float, bbox_w: float | None = None) -> Tuple[float, float]:
        """位置逆变换：物理 Count → 屏幕像素"""
        if self.bypass_mapping:
            return float(counts_x), float(counts_y)
        kx = self.calib.k_x if abs(self.calib.k_x) > 0.01 else 1.0
        ky = self.calib.k_y if abs(self.calib.k_y) > 0.01 else 1.0
        corrected_dx = counts_x / kx
        corrected_dy = counts_y / ky
        dx = math.tan(corrected_dx / self.focal_length) * self.focal_length
        dy = math.tan(corrected_dy / self.focal_length) * self.focal_length
        return dx, dy

    def reverse_map_velocity(
        self,
        counts_x: float,
        counts_y: float,
        px_x: float = 0.0,
        px_y: float = 0.0,
        bbox_w: float | None = None,
    ) -> Tuple[float, float]:
        """
        速度逆变换：counts/s → px/s (FOV Jacobian 一阶近似)
        P0 修复：与 calculate_velocity_move 严格互逆，避免 WorldModel 累加 ego_pos_px
        时与控制器实际下发的 counts 不匹配造成漂移。

        Parameters
        ----------
        counts_x, counts_y : 鼠标位移速度 (counts/s)
        px_x, px_y : 速度锚点位置 (px)，默认 0 即屏幕中心
        """
        if self.bypass_mapping:
            return float(counts_x), float(counts_y)
        kx = self.calib.k_x if abs(self.calib.k_x) > 0.01 else 1.0
        ky = self.calib.k_y if abs(self.calib.k_y) > 0.01 else 1.0
        jac_x, jac_y = self._velocity_jacobian(px_x, px_y)
        # 严格互逆：v_px = v_count / (k * jac)
        # 保护：jac≈0 时退化（屏幕极远端 / 极端 FOV），避免除零
        eps = 1e-6
        jx = jac_x if abs(jac_x) > eps else eps
        jy = jac_y if abs(jac_y) > eps else eps
        return counts_x / (kx * jx), counts_y / (ky * jy)

    def feedback_update(self, h_ai_x, pix_dx, h_ai_y, pix_dy):
        self.calib.update(h_ai_x, pix_dx, 'x')
        self.calib.update(h_ai_y, pix_dy, 'y')
