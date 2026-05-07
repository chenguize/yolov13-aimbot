# utils/rust_engine/__init__.py
"""
CIPHER v1.0 Rust-Native Engine.

编译（勿同时激活 conda base + venv）:
  cd utils/rust_engine && pip install maturin && maturin develop --release

用法:  from utils.rust_engine import CipherEngine
降级:  若未编译，CipherEngine = None；此时 rust_enable 无效，自动走 Python。

说明: 扩展模块名为 utils.rust_engine._core（与 pyproject module-name 一致）。
      旧版 maturin 可能只安装 rust_engine._core，这里会尝试兼容导入。
"""

from utils.logger import get_logger

_log = get_logger("RustEngine")

CipherEngine = None
try:
    from ._core import CipherEngine
except ImportError:
    try:
        import importlib

        _mod = importlib.import_module("rust_engine._core")
        CipherEngine = getattr(_mod, "CipherEngine", None)
    except ImportError:
        CipherEngine = None

if CipherEngine is not None:
    _log.info("Rust engine loaded — CipherEngine is available")
else:
    _log.warning(
        "Rust engine NOT compiled or wrong module path. "
        "Set [Controller] rust_enable=False or rebuild from repo root: "
        "cd utils/rust_engine && maturin develop --release"
    )
