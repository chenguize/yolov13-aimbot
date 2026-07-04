# utils/screen.py
"""
Windows 屏幕分辨率自动检测。

目的：让 config.ini 不再写死 screen_width/screen_height，启动时从 Windows 系统读取。
"""
import ctypes
import logging
import os
import platform
from typing import Tuple

logger = logging.getLogger("Screen")


def detect_screen_size() -> Tuple[int, int, str]:
    """
    自动检测主显示器物理分辨率。

    Returns
    -------
    (width, height, source) : Tuple[int, int, str]
        - width, height: 像素数（不含 Windows DPI scaling 缩放）
        - source: 来源标签，用于日志/调试

    Notes
    -----
    优先级：
      1. Windows API: ctypes.windll.user32.GetSystemMetrics
      2. 兜底: 1920x1080
    """
    # ── Windows: GetSystemMetrics ─────────────────────────────────────────
    if platform.system() == "Windows":
        try:
            user32 = ctypes.windll.user32
            # SM_CXSCREEN=0, SM_CYSCREEN=1
            w = int(user32.GetSystemMetrics(0))
            h = int(user32.GetSystemMetrics(1))
            if w > 0 and h > 0:
                return w, h, "Windows API (GetSystemMetrics)"
        except Exception as e:
            logger.warning(f"GetSystemMetrics 失败: {e}")

        # 备选：EnumDisplaySettings 读主显示器 DevMode
        try:
            user32 = ctypes.windll.user32
            ENUM_CURRENT_SETTINGS = -1
            DEVMODE = ctypes.c_uint32 * 158  # 简化结构占位
            devmode = ctypes.create_string_buffer(156)  # DEVMODE 真实大小因字段而异
            devmode_ptr = ctypes.c_void_p()
            # 跳过复杂结构，最常见的尺寸查询走 GetSystemMetrics 即可
        except Exception:
            pass

    # ── 兜底 ──────────────────────────────────────────────────────────────
    logger.warning(
        "无法自动检测屏幕分辨率，回退到默认值 1920x1080。"
        "请在 config.ini [Hardware] 显式设置 screen_width / screen_height。"
    )
    return 1920, 1080, "default (检测失败)"


def detect_dpi_scale() -> float:
    """
    检测 Windows DPI 缩放比例（1.0 = 100%, 1.5 = 150%, 2.0 = 200%）。
    用于诊断日志，与游戏内渲染分辨率无关。
    """
    if platform.system() == "Windows":
        try:
            user32 = ctypes.windll.user32
            # GetDpiForSystem (Win 10 1607+)
            try:
                shcore = ctypes.windll.shcore
                dpi = shcore.GetDpiForSystem()
                if dpi > 0:
                    return dpi / 96.0
            except Exception:
                pass
            # 兜底：GetSystemMetrics(SM_CYSCREEN) vs GetDeviceCaps
            try:
                hdc = user32.GetDC(0)
                gdi32 = ctypes.windll.gdi32
                LOGPIXELSX = 88
                dpi = gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
                user32.ReleaseDC(0, hdc)
                if dpi > 0:
                    return dpi / 96.0
            except Exception:
                pass
        except Exception:
            pass
    return 1.0
