# perception/human_input.py
"""
物理鼠标输入监听器。

职责：
- 阻塞式监听 RawInput / pynput 鼠标事件
- 将人类物理位移（is_ai=False）写入 RingBuffer
- 与 RingBuffer 同属 perception 层，作为其唯一人类事件生产者

后端选择：
  inputs  —— inputs.get_mouse() 的 REL（部分全屏/独占下可能始终无事件）
  pynput  —— WH_MOUSE_LL 钩子上用屏幕坐标差分当相对量（需 pip install pynput）
"""

import threading
import time

from config import config
from utils.logger import get_logger

logger = get_logger("HumanInput")

_WM_MOUSEMOVE = 0x0200
_AI_SIGNATURE = 0xFFC0FFEE


class HumanMouseListener(threading.Thread):
    """
    物理手位移（is_ai=False）写入 RingBuffer，供人机离合器 / 到点停手唤醒。

    后端 [General] human_input_backend：
      inputs  —— inputs.get_mouse() 的 REL（部分全屏/独占下可能始终无事件 → RingBuffer 无人手事件）；
      pynput  —— WH_MOUSE_LL 钩子上用屏幕坐标差分当相对量，不少环境下比 inputs 稳（需 pip install pynput）。
    仍全 0 时：试管理员运行；FPS 强锁指针+仅原始轴时，只能上 Win32 RegisterRawInput（未内建）。
    """

    def __init__(self, ring_buffer, shutdown_evt: threading.Event):
        super().__init__(name="RawInputListener", daemon=True)
        self.ring_buffer = ring_buffer
        self.shutdown_evt = shutdown_evt
        b = (config.getstr("Input", "human_input_backend", "inputs") or "inputs").strip().lower()
        if b in ("pyn", "pynput", "hook"):
            b = "pynput"
        else:
            b = "inputs"
        self._backend = b
        if self._backend == "pynput" and hasattr(
            self.ring_buffer, "enable_observed_echo_reconciliation"
        ):
            self.ring_buffer.enable_observed_echo_reconciliation(True)

    def _run_inputs(self) -> None:
        try:
            from inputs import get_mouse
        except ImportError:
            logger.error("缺少依赖 'inputs'，请 pip install inputs")
            return
        logger.info("RawInput: inputs 已加载，get_mouse REL -> RingBuffer")

        try:
            from inputs import UnpluggedError
        except (ImportError, AttributeError):
            UnpluggedError = OSError

        while not self.shutdown_evt.is_set():
            try:
                events = get_mouse()
                dx = dy = 0
                for event in events:
                    if event.ev_type == "Relative":
                        if event.code == "REL_X":
                            dx += event.state
                        elif event.code == "REL_Y":
                            dy += event.state
                if dx or dy:
                    self.ring_buffer.add_event(dx, dy, is_ai=False)
                else:
                    time.sleep(0.001)
            except UnpluggedError:
                time.sleep(1.0)
            except Exception as e:
                logger.debug("HumanMouseListener(inputs) transient: %s", e)
                time.sleep(0.001)

    def _run_pynput(self) -> None:
        try:
            from pynput.mouse import Listener
        except ImportError:
            logger.error(
                "human_input_backend=pynput 但未安装 pynput，请 pip install pynput；回退 inputs"
            )
            self._run_inputs()
            return

        last: list = [None]
        tagged_injected_move: list[bool] = [False]

        def win32_event_filter(msg, data):
            if int(msg) == _WM_MOUSEMOVE:
                try:
                    tagged_injected_move[0] = (
                        int(data.dwExtraInfo) == _AI_SIGNATURE
                    )
                except (AttributeError, TypeError, ValueError):
                    tagged_injected_move[0] = False
            return True

        def on_move(x, y, injected=False) -> None:
            try:
                ix, iy = int(round(x)), int(round(y))
            except (TypeError, ValueError, OverflowError):
                return
            if last[0] is None:
                last[0] = (ix, iy)
                return
            lx, ly = last[0]
            dx, dy = ix - lx, iy - ly
            last[0] = (ix, iy)
            is_tagged_ai = tagged_injected_move[0]
            tagged_injected_move[0] = False
            # Games commonly recenter a locked cursor with injected events
            # that do not preserve our dwExtraInfo marker. Neither those
            # warps nor our SendInput commands are physical human intent.
            if bool(injected) or is_tagged_ai:
                return
            if dx or dy:
                self.ring_buffer.add_observed_cursor_event(
                    dx, dy, known_physical=True
                )

        listener = Listener(
            on_move=on_move,
            win32_event_filter=win32_event_filter,
        )
        listener.start()
        logger.info(
            "RawInput: pynput 鼠标钩子已启动（dwExtraInfo 标签分离 AI/物理输入）"
        )
        self.shutdown_evt.wait()
        try:
            listener.stop()
        except Exception:
            pass
        try:
            listener.join(timeout=2.0)
        except Exception:
            pass

    def run(self) -> None:
        logger.info(
            "RawInput Mouse Listener starting | human_input_backend=%s",
            self._backend,
        )
        if self._backend == "pynput":
            self._run_pynput()
        else:
            self._run_inputs()
        logger.info("RawInput Mouse Listener stopped")
