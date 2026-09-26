"""v0.6 comparison: the heterogeneous observer (posterior / side-switch / Wald band) against the uniform
v0.2 observer (EMA / CUSUM / margin), same worlds, same seeds (0–19), same 20,000 steps.

    python3 run_hetero.py --out results/hetero.json
"""
from __future__ import annotations

import argparse
import json
import os

from node_observer.hetero import HeteroParams, run_hetero
from run_census2 import q, NOISE, TRICK

KEYS = ["score_per_1000", "hit_rate", "regime_accuracy", "ambiguous_frac", "alarms", "detected",
        "attributed", "spurious", "median_delay", "max_delay", "switches", "AR_final"]


def med(rows, k):
    return q([r.get(k) for r in rows], 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", default="results/hetero.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    c2 = json.load(open("results/census2.json"))
    base = {"noise": c2["arms"]["trusting/noise/channels"]["runs"],
            "trick": c2["arms"]["trusting/trick/channels"]["runs"]}
    out = dict(seeds=list(range(a.seeds)), steps=a.steps, arms={})
    print(f"{'arm':38s} {'score/1k':>8} {'hit':>7} {'regAcc':>7} {'held':>6} {'alarms':>6} {'detect':>6} {'spur':>5} {'delay':>5} {'maxD':>5} {'switch':>6} {'AR':>7}")
    for wname, world in (("noise", NOISE), ("trick", TRICK)):
        b = base[wname]
        print(f"{'uniform v0.2 (EMA/CUSUM/margin) / ' + wname:38s} {med(b,'score_per_1000')!s:>8} {round(med(b,'hit_rate'),4)!s:>7} "
              f"{med(b,'regime_accuracy')!s:>7} {med(b,'ambiguous_frac')!s:>6} {med(b,'alarms')!s:>6} {med(b,'detected')!s:>6} "
              f"{'—':>5} {med(b,'median_delay')!s:>5} {'—':>5} {med(b,'switches')!s:>6} {med(b,'AR_final')!s:>7}")
        for alpha in (0.05, 0.01):
            rows = [run_hetero(world, HeteroParams(alpha=alpha), a.steps, s) for s in range(a.seeds)]
            name = f"hetero alpha={alpha} / {wname}"
            out["arms"][name] = dict(summary={k: med(rows, k) for k in KEYS}, runs=rows)
            print(f"{name:38s} {med(rows,'score_per_1000')!s:>8} {round(med(rows,'hit_rate'),4)!s:>7} "
                  f"{med(rows,'regime_accuracy')!s:>7} {med(rows,'ambiguous_frac')!s:>6} {med(rows,'alarms')!s:>6} {med(rows,'detected')!s:>6} "
                  f"{med(rows,'spurious')!s:>5} {med(rows,'median_delay')!s:>5} {med(rows,'max_delay')!s:>5} {med(rows,'switches')!s:>6} {med(rows,'AR_final')!s:>7}")
    r0 = out["arms"]["hetero alpha=0.05 / noise"]["runs"][0]
    print(f"\nband A = {r0['band_A']} nats; per-channel LLR weight = {r0['llr_weight']} nats; eight agreeing channels = {8*r0['llr_weight']:.2f} nats per step")
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
