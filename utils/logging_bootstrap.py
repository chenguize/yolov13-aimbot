# 必须在 import agent / output 等模块之前调用，保证子模块 logger 与格式一致
import logging
from config import config


def setup_root_logging() -> int:
    """
    从 config.ini [Debug] 读 log_level，配置 root（含 %(name)s 便于看环节）。
    返回生效的 level 数值，供需要时比对。
    """
    name = config.getstr("Debug", "log_level", "INFO").strip().upper()
    _n2l = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = _n2l.get(name, logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    kwargs = dict(level=level, format=fmt)
    try:
        logging.basicConfig(**kwargs, force=True)  # py3.8+
    except TypeError:
        # 极老环境无 force
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
        logging.basicConfig(**kwargs)
    return level
