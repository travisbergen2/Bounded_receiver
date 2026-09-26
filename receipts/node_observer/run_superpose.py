"""v0.5 census: hard (possibilistic) and soft (posterior) superposition filters on the phase world.

    python3 run_superpose.py --seeds 20 --out results/superpose.json
"""
from __future__ import annotations

import argparse
import json
import os

from node_observer.superpose import SuperposeParams, run_superpose, factored_demo


def q(xs, p=0.5):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


ARMS = {
    "hard/noiseless/n=4":   SuperposeParams(steps=2000, n=4, eps=0.0, p_switch=0.02, mode="hard"),
    "hard/noiseless/n=16":  SuperposeParams(steps=2000, n=16, eps=0.0, p_switch=0.02, mode="hard"),
    "hard/noiseless/n=64":  SuperposeParams(steps=2000, n=64, eps=0.0, p_switch=0.02, mode="hard"),
    "hard/noise0.15/n=4":   SuperposeParams(steps=2000, n=4, eps=0.15, p_switch=0.005, mode="hard"),
    "hard/noise0.05/n=4":   SuperposeParams(steps=2000, n=4, eps=0.05, p_switch=0.005, mode="hard"),
    "soft/noise0.15/n=4":   SuperposeParams(steps=2000, n=4, eps=0.15, p_switch=0.005, mode="soft"),
    "soft/noise0.05/n=4":   SuperposeParams(steps=2000, n=4, eps=0.05, p_switch=0.005, mode="soft"),
    "soft/noise0.15/n=4/p0.005": SuperposeParams(steps=2000, n=4, eps=0.15, p_switch=0.005, mode="soft", surprise_p=0.005),
    "soft/noise0.15/n=16":  SuperposeParams(steps=2000, n=16, eps=0.15, p_switch=0.005, mode="soft"),
}

KEYS = ["switches", "switches_while_collapsed", "absorbed", "alarms", "detected", "attributed", "false_alarms", "surprises", "min_p_obs",
        "median_delay", "median_attr_delay", "max_attr_delay", "collapsed_frac",
        "answer_accuracy", "mean_collapse_time", "max_collapse_time", "mean_size"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/superpose.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out = dict(seeds=list(range(a.seeds)), arms={}, factored=factored_demo(4, 16, steps=12, seed=0))
    print(f"{'arm':30s} {'switch':>6} {'whileC':>6} {'alarms':>6} {'attrib':>6} {'spurious':>8} {'surpr':>5} {'minP':>7} {'delay':>5} {'attrDel':>7} {'maxDel':>6} {'collapsed':>9} {'acc':>6} {'meanT':>6} {'maxT':>5} {'size':>6}")
    for name, p in ARMS.items():
        rows = [run_superpose(p, s) for s in range(a.seeds)]
        s = {k: q([r[k] for r in rows]) for k in KEYS}
        s["max_collapse_time_max"] = max((r["max_collapse_time"] for r in rows if r["max_collapse_time"] is not None), default=None)
        out["arms"][name] = dict(summary=s, runs=rows)
        print(f"{name:30s} {s['switches']!s:>6} {s['switches_while_collapsed']!s:>6} {s['alarms']!s:>6} {s['attributed']!s:>6} {s['false_alarms']!s:>8} {s['surprises']!s:>5} {s['min_p_obs']!s:>7} "
              f"{s['median_delay']!s:>5} {s['median_attr_delay']!s:>7} {s['max_attr_delay']!s:>6} "
              f"{s['collapsed_frac']!s:>9} {s['answer_accuracy']!s:>6} {s['mean_collapse_time']!s:>6} {s['max_collapse_time']!s:>5} {s['mean_size']!s:>6}")
    print("\nfactored demo (two noiseless worlds n=4 and n=16, quadrature nodes): step, size1, size2, joint = product, held = sum")
    for row in out["factored"]:
        print(f"  {row['step']:>2}: {row['size1']:>3} {row['size2']:>3}  joint {row['joint']:>4}  held {row['held']:>3}")
    r0 = out["arms"]["hard/noiseless/n=64"]["runs"][0]
    print("\nhard n=64 seed 0 size trace:", r0["size_trace"])
    r1 = out["arms"]["soft/noise0.15/n=4"]["runs"][0]
    print("soft n=4 eps 0.15 seed 0 effective-size trace:", r1["size_trace"][:24])
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
