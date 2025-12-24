# config.py
"""
全局配置单例 - 从 config.ini 读取所有参数
支持类型自动转换（bool/int/float/str）
"""

import configparser
from pathlib import Path
from typing import Any, Optional


class Config:
    _instance: Optional['Config'] = None

    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
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

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """通用获取方法，会尝试智能转换类型"""
        if not self.parser.has_section(section) or not self.parser.has_option(section, key):
            return default

        val = self.parser.get(section, key).strip()

        # 空值处理
        if not val:
            return default

        # bool 转换
        if val.lower() in ('true', 'false', 'yes', 'no', '1', '0'):
            return val.lower() in ('true', 'yes', '1')

        # 尝试数字
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    def getstr(self, section: str, key: str, default: str = "") -> str:
        return str(self.get(section, key, default))

    def getint(self, section: str, key: str, default: int = 0) -> int:
        return int(self.get(section, key, default))

    def getfloat(self, section: str, key: str, default: float = 0.0) -> float:
        return float(self.get(section, key, default))

    def getbool(self, section: str, key: str, default: bool = False) -> bool:
        return bool(self.get(section, key, default))

    def getlist(self, section: str, key: str, default: list = None) -> list:
        val = self.getstr(section, key, "")
        if not val:
            return default or []
        return [item.strip() for item in val.split(',') if item.strip()]


# 全局单例访问
config = Config()