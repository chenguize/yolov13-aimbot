"""Deterministic multi-run benchmark for controller A/B comparisons.

Each run uses a fixed seed so that different controller tunings can be
compared fairly. The seeds are chosen to give a variety of target sequences
(close/far, slow/fast, stationary/moving), rather than just one lucky/unlucky
pattern.
"""
import io
import contextlib
import os
import random as pyrand
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)

import test.control as c


DEFAULT_SEEDS = (11, 23, 47, 71, 103, 137, 199, 251)


def run(seeds=DEFAULT_SEEDS, verbose: bool = True):
    overall, ttk, prec, nat, bio = [], [], [], [], []
    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        pyrand.seed(seed)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, _, a = c.run_simulation_with_diagnostics(plot=False, verbose=False)
        overall.append(a["overall_score"])
        ttk.append(a["acq_score"])
        prec.append(a["precision_score"])
        nat.append(a["natural_score"])
        bio.append(a["bio_bonus"])
        if verbose:
            print(
                f"seed={seed:<4d} overall={a['overall_score']:5.1f}  "
                f"TTK={a['acq_score']:5.1f}  Prec={a['precision_score']:5.1f}  "
                f"Nat={a['natural_score']:5.1f}  Bio={a['bio_bonus']:5.1f}"
            )

    print(
        f"\nMEAN overall={np.mean(overall):5.2f}  "
        f"std={np.std(overall):.2f}  "
        f"TTK={np.mean(ttk):.1f}  Prec={np.mean(prec):.1f}  "
        f"Nat={np.mean(nat):.1f}  Bio={np.mean(bio):.1f}"
    )
    return float(np.mean(overall))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(DEFAULT_SEEDS)
    run(DEFAULT_SEEDS[:n])
