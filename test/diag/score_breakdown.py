# _score_breakdown.py  ── 把 baseline 在 10 seeds 下的四个子分数按模式拆开看
# 目的：找出 pure_ai vs human_flick 哪一种拖后腿，针对性改算法
import contextlib
import io
import os
import random as pyrand
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)

from test import control as c  # noqa: E402
from test.simulation import BASELINE, VALIDATION_SEEDS  # noqa: E402


def _mean(arr, default=0.0):
    return float(np.mean(arr)) if len(arr) else default


def run_one(seed: int) -> dict:
    np.random.seed(seed)
    pyrand.seed(seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, _, a = c.run_simulation_with_diagnostics(
            duration=30.0, plot=False, verbose=False, use_fixed=True, params=BASELINE,
        )
    return a


def mode_subscores(m_mode: dict, mode_key: str) -> dict:
    """把 per-mode 的 metrics 换算成同量纲的 4 个子分（TTK/Prec/Nat/Bio）."""
    # v4.1: 两种模式都用 phase_times（see_idx → lock_idx），TTK 阈值 0.18→100, 0.40→0
    times = m_mode.get("phase_times", [])
    ttk_s = 100.0 - max(0.0, _mean(times) - 0.18) * (100.0 / 0.22) if times else 0.0
    ttk_s = max(0.0, min(100.0, ttk_s))
    err   = _mean(m_mode.get("headshot_errors", []))
    prec  = max(0.0, min(100.0, 100.0 - max(0.0, err - 10.0) * 10.0)) if m_mode.get("headshot_errors") else 0.0
    nat   = _mean(m_mode.get("natural_scores", []))
    bio   = _mean(m_mode.get("bio_bonuses", []))
    kills = len(m_mode.get("steady_maes", []))
    return {
        "kills":     kills,
        "mean_time": _mean(times),
        "ttk":       ttk_s,
        "precision": prec,
        "natural":   nat,
        "bio":       bio,
        "overall":   0.40 * ttk_s + 0.20 * prec + 0.25 * nat + 0.15 * bio,
    }


def print_mode_table(label: str, rows: list):
    print(f"\n{label}")
    print(f"{'seed':>5} {'kills':>5} {'avg_t':>6} {'Overall':>8} "
          f"{'TTK':>6} {'Prec':>6} {'Nat':>6} {'Bio':>6}")
    print("-" * 58)
    for r in rows:
        print(f"{r['seed']:>5} {r['kills']:>5} {r['mean_time']:>6.3f} "
              f"{r['overall']:>8.2f} "
              f"{r['ttk']:>6.1f} {r['precision']:>6.1f} "
              f"{r['natural']:>6.1f} {r['bio']:>6.1f}")
    print("-" * 58)
    if rows:
        import statistics as st
        def col(k): return [r[k] for r in rows if r["kills"] > 0]
        print(f"{'mean':>5} {sum(r['kills'] for r in rows):>5} "
              f"{st.mean(col('mean_time')):>6.3f} "
              f"{st.mean(col('overall')):>8.2f} "
              f"{st.mean(col('ttk')):>6.1f} {st.mean(col('precision')):>6.1f} "
              f"{st.mean(col('natural')):>6.1f} {st.mean(col('bio')):>6.1f}")


def main():
    pure_rows, flick_rows, combined = [], [], []
    for s in VALIDATION_SEEDS:
        a = run_one(s)
        m = a.get("metrics", {})
        pa = mode_subscores(m.get("pure_ai",     {}), "pure_ai")
        hf = mode_subscores(m.get("human_flick", {}), "human_flick")
        pa["seed"] = s
        hf["seed"] = s
        pure_rows.append(pa)
        flick_rows.append(hf)
        combined.append({
            "seed":     s,
            "kills":    int(a.get("total_kills", 0)),
            "overall":  a.get("overall_score", 0.0),
            "ttk":      a.get("acq_score", 0.0),
            "precision":a.get("precision_score", 0.0),
            "natural":  a.get("natural_score", 0.0),
            "bio":      a.get("bio_bonus", 0.0),
        })

    print_mode_table("🤖 纯 AI 模式 (pure_ai)", pure_rows)
    print_mode_table("🧑 人类先动 → AI 接管 (human_flick)", flick_rows)

    print("\n📊 综合（两种模式汇总）")
    print(f"{'seed':>5} {'kills':>5} {'Overall':>8} "
          f"{'TTK':>6} {'Prec':>6} {'Nat':>6} {'Bio':>6}")
    print("-" * 52)
    for r in combined:
        print(f"{r['seed']:>5} {r['kills']:>5} {r['overall']:>8.2f} "
              f"{r['ttk']:>6.1f} {r['precision']:>6.1f} "
              f"{r['natural']:>6.1f} {r['bio']:>6.1f}")
    import statistics as st
    def col(k): return [r[k] for r in combined]
    print("-" * 52)
    print(f"{'mean':>5} {sum(r['kills'] for r in combined):>5} "
          f"{st.mean(col('overall')):>8.2f} "
          f"{st.mean(col('ttk')):>6.1f} {st.mean(col('precision')):>6.1f} "
          f"{st.mean(col('natural')):>6.1f} {st.mean(col('bio')):>6.1f}")

    # 两种模式对比差距
    def avg(rows, k): return float(np.mean([r[k] for r in rows if r['kills'] > 0]))
    print("\n🔍 模式对比 (human_flick 比 pure_ai)")
    for k in ("overall", "ttk", "precision", "natural", "bio"):
        pa_v, hf_v = avg(pure_rows, k), avg(flick_rows, k)
        delta = hf_v - pa_v
        flag  = "↓差" if delta < -3 else ("↑好" if delta > 3 else "≈")
        print(f"  {k:<10} pure_ai={pa_v:6.2f}  human_flick={hf_v:6.2f}  Δ={delta:+6.2f}  {flag}")


if __name__ == "__main__":
    main()
