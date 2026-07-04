# config.py - 全局配置管理模块
#
# 功能：
# 1. 从 config.ini 读取所有参数
# 2. 支持类型自动转换（bool/int/float/str）
# 3. 提供全局单例访问
# 4. 智能类型转换和默认值处理
#

import configparser
from pathlib import Path
from typing import Any, Optional


def _strip_inline_comment(raw: str) -> str:
    """去掉行尾「空白 + # …」注释（configparser 不会自动分割同值行内注释）。"""
    s = raw.strip()
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "#" and (i == 0 or s[i - 1].isspace()):
            return s[:i].rstrip()
        i += 1
    return s


class Config:
    """全局配置单例类 - 从 config.ini 读取所有参数，支持类型自动转换"""
    
    _instance: Optional['Config'] = None  # 配置单例实例

    def __new__(cls) -> 'Config':
        """实现单例模式，确保全局只有一个配置实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()  # 首次创建时加载配置
        return cls._instance

    def _load(self):
        """加载配置文件"""
        self.parser = configparser.ConfigParser(
            interpolation=None,
            allow_no_value=True
        )

        config_path = Path(__file__).parent / "config.ini"

        if not config_path.exists():
            raise FileNotFoundError(
                f"配置文件未找到: {config_path}\n"
                "请确保项目根目录下存在 config.ini"
            )

        # 保留原始大小写
        self.parser.optionxform = str
        self.parser.read(config_path, encoding='utf-8')

        # 记录哪些 (section, key) 在 get 时被自动替换（用于启动日志/调试）
        self._replaced_keys: set = set()

        # ── 自动检测屏幕分辨率（仅一次，启动期）──────────────────────────────
        # config.ini [Hardware].screen_width / screen_height 为 0 或 "auto" 时，
        # 实际取值由 detect_screen_size() 给出。检测结果缓存在 self._detected_screen。
        try:
            from utils.screen import detect_screen_size
            self._detected_screen = detect_screen_size()
        except Exception:
            self._detected_screen = (1920, 1080, "default (检测异常)")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """通用获取方法，会尝试智能转换类型"""
        if not self.parser.has_section(section) or not self.parser.has_option(section, key):
            return default

        val = _strip_inline_comment(self.parser.get(section, key))

        # 空值处理
        if not val:
            return default

        # 尝试整数转换（必须在 bool 之前：否则 "0"/"1" 被误判为 False/True）
        try:
            ival = int(val)
            # ── 自动屏幕分辨率：0 或 "auto" → 替换为检测值 ──────────────────
            if section == "Hardware" and key in ("screen_width", "screen_height") and ival <= 0:
                self._replaced_keys.add((section, key))
                return float(self._detected_screen[0 if key == "screen_width" else 1])
            return ival
        except ValueError:
            pass

        # 字符串 "auto" 也触发自动检测
        if val.strip().lower() == "auto":
            if section == "Hardware" and key in ("screen_width", "screen_height"):
                self._replaced_keys.add((section, key))
                return float(self._detected_screen[0 if key == "screen_width" else 1])
            return default

        # 尝试浮点数转换
        try:
            fval = float(val)
            # 浮点 0 也触发自动检测
            if section == "Hardware" and key in ("screen_width", "screen_height") and fval <= 0:
                self._replaced_keys.add((section, key))
                return float(self._detected_screen[0 if key == "screen_width" else 1])
            return fval
        except ValueError:
            pass

        # bool 转换 — 仅关键字，不含 "0"/"1"（已在上面被 int 捕获）
        if val.lower() in ('true', 'false', 'yes', 'no'):
            return val.lower() in ('true', 'yes')

        # 字符串原样返回
        return val

    def getstr(self, section: str, key: str, default: str = "") -> str:
        """获取字符串类型配置值"""
        return str(self.get(section, key, default))

    def getint(self, section: str, key: str, default: int = 0) -> int:
        """获取整数类型配置值"""
        return int(self.get(section, key, default))

    def getfloat(self, section: str, key: str, default: float = 0.0) -> float:
        """获取浮点数类型配置值"""
        return float(self.get(section, key, default))

    def getbool(self, section: str, key: str, default: bool = False) -> bool:
        """获取布尔类型配置值"""
        return bool(self.get(section, key, default))

    def getlist(self, section: str, key: str, default: list = None) -> list:
        """获取列表类型配置值（逗号分隔）"""
        val = self.getstr(section, key, "")
        if not val:
            return default or []
        return [item.strip() for item in val.split(',') if item.strip()]

    def was_replaced(self, section: str, key: str) -> bool:
        """判断指定 (section, key) 在 get 时是否被自动替换（如 screen_width 由 0 → 检测值）。"""
        return (section, key) in self._replaced_keys
    def reload(self):
        self._load()

# 全局单例访问 - 提供全局配置访问接口
config = Config()