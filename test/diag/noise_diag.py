# _noise_diag.py  ── 诊断：同参数 + 同种子，重复调用 evaluate() 结果是否可复现？
# 如果可复现 → CMA-ES 噪声来自 params 扰动引发的 butterfly effect
# 如果不可复现 → 有额外随机源未被 seed 控制，需要修复
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)

from test.simulation import BASELINE, evaluate, VALIDATION_SEEDS  # noqa: E402


def main():
    runs = 3
    print(f"同 BASELINE + 同 {len(VALIDATION_SEEDS)} seeds，重复 {runs} 次：")
    all_per_seed = []
    for i in range(runs):
        mean_s, std_s, per_seed = evaluate(BASELINE, seeds=VALIDATION_SEEDS)
        print(f"  run#{i}: mean={mean_s:6.2f}  std={std_s:5.2f}  "
              f"per_seed={['%5.1f' % x for x in per_seed]}")
        all_per_seed.append(per_seed)
    arr = np.asarray(all_per_seed)
    print("\n各 seed 在 3 次重复下的标准差（若 0.0 → 完全可复现）：")
    for j, s in enumerate(VALIDATION_SEEDS):
        print(f"  seed={s:>4}  runs={arr[:, j].tolist()}  std={arr[:, j].std():.3f}")


if __name__ == "__main__":
    main()
