# -*- coding: utf-8 -*-
"""
瞄准环路边沿诊断：滑动窗口统计误差/速度符号翻转、flick↔track 跳变，
并给出与映射/延迟/检测/控制相关的可读提示。仅主线程调用。
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from config import config

_log = logging.getLogger("AimDiag")


def _flips_1d(arr: List[float], eps: float = 0.4) -> int:
    """明显符号穿越次数；靠近零的样本跳过。"""
    if len(arr) < 2:
        return 0
    c = 0
    for i in range(1, len(arr)):
        a, b = arr[i - 1], arr[i]
        if abs(a) < eps or abs(b) < eps:
            continue
        if a * b < 0:
            c += 1
    return c


def _mode_switches(modes: List[str]) -> int:
    if len(modes) < 2:
        return 0
    return sum(1 for i in range(1, len(modes)) if modes[i] != modes[i - 1])


def _std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    a = np.array(xs, dtype=np.float64)
    return float(np.std(a)) if a.size else 0.0


def _build_hints(
    fp: int,
    fpy: int,
    fvx: int,
    fvy: int,
    msw: int,
    h_ov_mean: float,
    rms: float,
    lead_ms: float,
    vh_ms: float,
    inf_ema: float,
    bypass: bool,
) -> str:
    parts: List[str] = []
    if fp >= 4 or fpy >= 4:
        parts.append(
            "【视觉/前馈】p_predict 穿零多：查检测稳(conf)、串流+WM 延迟(lead/vh/inf_ema)、丢框 coast"
        )
    if (fvx >= 5 or fvy >= 5) and fp < 3 and fpy < 3:
        parts.append("【控制】arm 速符号常翻而 p 较稳：查 CIPHER/OU/死区/阻抗与 flick 滞回")
    if msw >= 4:
        parts.append("【模式】flick↔track 抖：可微调 Controller mode 阈值或跟踪带宽")
    if 0.2 < h_ov_mean < 0.9:
        parts.append("【离合】human_override 中段：手微抖会抢权→试调 s 阈值或关 RawInput 对比")
    if rms < 6.0 and (fp + fpy + fvx) > 6:
        parts.append("【小误差大摆】近距仍晃：k 因子/控制增益/1kHz 叠 OU 常见")
    if lead_ms > 40 or vh_ms > 20:
        parts.append("【延迟】lead 或 vh 偏大：校 moonlight、base_hardware_lag、串流码率")
    if inf_ema > 25:
        parts.append("【推理慢】推理解码高→「视觉」滞后、易过冲回拉")
    if bypass:
        parts.append("【映射】bypass_strategy_mapping=1 仅桌测，与游戏 FOV/灵敏度 不一致必漂")
    if not parts:
        return "视 flip 哪条高：px/py→检测与延迟，avx/avy→控制，mode_sw→阈附近颤"
    return " | ".join(parts)


class AimDiagnostics:
    _instance: Optional["AimDiagnostics"] = None

    def __new__(cls) -> "AimDiagnostics":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._inited = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_inited", False):
            return
        self._inited = True
        self._win = 24
        self._px: Deque[float] = deque(maxlen=64)
        self._py: Deque[float] = deque(maxlen=64)
        self._ix: Deque[float] = deque(maxlen=64)
        self._iy: Deque[float] = deque(maxlen=64)
        self._avx: Deque[float] = deque(maxlen=64)
        self._avy: Deque[float] = deque(maxlen=64)
        self._mode: Deque[str] = deque(maxlen=64)
        self._hov: Deque[float] = deque(maxlen=64)
        self._pow: Deque[float] = deque(maxlen=64)
        self._last_emit = 0.0
        self._last_lead = 0.0
        self._last_vh = 0.0
        self._last_inf = 0.0
        self._last_dt = 0.0
        self._last_chase = ""
        self._last_bypass = False
        self._last_sp = 0.0
        self._last_re = 0.0

    def reconfigure(self) -> None:
        w = max(8, int(config.getint("Debug", "aim_diag_window", 24)))
        if w != self._win:
            self._win = w
            for d in (
                self._px,
                self._py,
                self._ix,
                self._iy,
                self._avx,
                self._avy,
                self._mode,
                self._hov,
                self._pow,
            ):
                d.clear()

    def record(
        self,
        *,
        px: float,
        py: float,
        intent_x: float,
        intent_y: float,
        power: float,
        human_override: float,
        spatial: float,
        reaction: float,
        arm_vx: float,
        arm_vy: float,
        mode: str,
        lead_ms: float,
        vh_s: float,
        inf_ema: float,
        dt: float,
        chase: str,
        bypass_map: bool,
    ) -> None:
        if not config.getbool("Debug", "aim_diagnostics", False) and not config.getbool(
            "Debug", "aim_diag_warn_only", False
        ):
            return
        self._px.append(float(px))
        self._py.append(float(py))
        self._ix.append(float(intent_x))
        self._iy.append(float(intent_y))
        self._avx.append(float(arm_vx))
        self._avy.append(float(arm_vy))
        self._mode.append(str(mode))
        self._hov.append(float(human_override))
        self._pow.append(float(power))
        self._last_lead = float(lead_ms)
        self._last_vh = float(vh_s)
        self._last_inf = float(inf_ema)
        self._last_dt = float(dt)
        self._last_chase = str(chase)
        self._last_bypass = bool(bypass_map)
        self._last_sp = float(spatial)
        self._last_re = float(reaction)

    def maybe_emit(self) -> None:
        en = config.getbool("Debug", "aim_diagnostics", False)
        w_only = config.getbool("Debug", "aim_diag_warn_only", False)
        if not en and not w_only:
            return
        itv = max(0.15, float(config.getfloat("Debug", "aim_diag_interval_sec", 0.4)))
        thr = max(2, int(config.getint("Debug", "aim_diag_warn_flips", 5)))
        now = time.perf_counter()
        if now - self._last_emit < itv:
            return
        if len(self._px) < 6:
            return

        w = min(self._win, len(self._px))
        px = list(self._px)[-w:]
        py = list(self._py)[-w:]
        avx = list(self._avx)[-w:]
        avy = list(self._avy)[-w:]
        modes = list(self._mode)[-w:]
        hov = list(self._hov)[-w:]
        ixw = list(self._ix)[-w:]

        fp = _flips_1d(px)
        fpy = _flips_1d(py)
        fvx = _flips_1d(avx, eps=1.0)
        fvy = _flips_1d(avy, eps=1.0)
        msw = _mode_switches(modes)
        rms = math.sqrt(sum(p * p + q * q for p, q in zip(px, py)) / w) if w else 0.0
        hov_m = float(np.mean(hov)) if hov else 0.0
        pw = list(self._pow)[-w:]
        pow_m = float(np.mean(pw)) if pw else 0.0

        nq = 0
        same = 0
        for i in range(w):
            if abs(px[i]) < 0.3 or abs(ixw[i]) < 0.5:
                continue
            nq += 1
            if px[i] * ixw[i] > 0:
                same += 1
        s_ratio = (same / nq) if nq else -1.0

        lead = self._last_lead
        vh_ms = self._last_vh * 1000.0
        inf = self._last_inf
        warn = (fp >= thr or fpy >= thr or fvx >= thr or fvy >= thr or msw >= max(2, thr // 2 + 1))

        tip = _build_hints(
            fp, fpy, fvx, fvy, msw, hov_m, rms, lead, vh_ms, inf, self._last_bypass
        )

        if en:
            _log.info(
                "AimDiag | w=%d | rms_p=%.1fpx | flips: px=%d py=%d avx=%d avy=%d | mode_sw=%d | "
                "px·ix同号比=%.2f(1.0 仅桌测/异常映射需对照游戏坐标)",
                w,
                rms,
                fp,
                fpy,
                fvx,
                fvy,
                msw,
                s_ratio,
            )
            _log.info(
                "AimDiag | p=(%.1f,%.1f) lead=%.0fms vh=%.0fms inf_ema=%.0fms | "
                "pwr=%.2f(±%.2f) h_ov=%.2f(±%.2f) | chase=%s sp=%.2f re=%.2f dt=%.4f",
                px[-1] if px else 0.0,
                py[-1] if py else 0.0,
                lead,
                vh_ms,
                inf,
                pow_m,
                _std(pw) if pw else 0.0,
                hov_m,
                _std(hov) if hov else 0.0,
                self._last_chase,
                self._last_sp,
                self._last_re,
                self._last_dt,
            )
            _log.info("AimDiag | hints: %s", tip)
        elif w_only and warn:
            _log.warning(
                "AimDiag[WARN] 振荡: px=%d py=%d avx=%d avy=%d mode_sw=%d | %s",
                fp,
                fpy,
                fvx,
                fvy,
                msw,
                tip,
            )

        self._last_emit = now


def aim_diag() -> AimDiagnostics:
    d = AimDiagnostics()
    d.reconfigure()
    return d
