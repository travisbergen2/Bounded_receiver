"""Follow-up to census2 (written AFTER reading census2, so labelled post-hoc): (a) the skeptic's per-source
threshold recalibrated to the noise floor (src_FT 8 instead of 4); (b) a small jury (3 channels, one
bursty liar) where a single liar can swing the majority. Also extracts the false-burn timing of the
original skeptic from census2.json.

    python3 run_followup2.py --seeds 20 --steps 20000
"""
from __future__ import annotations

import argparse
import json
import os

from node_observer.layered import ChannelWorldParams, SkepticParams, LayeredRunParams, run_layered
from run_census2 import summarize, NOISE, TRICK

SMALL = ChannelWorldParams(n=3, eps=0.15, p_switch=0.005, trickster=1, lie_on=0.01, lie_len=60.0)
SMALL_NOISE = ChannelWorldParams(n=3, eps=0.15, p_switch=0.005)


def arms(steps):
    return {
        "skeptic8/noise/channels": LayeredRunParams(steps=steps, world=NOISE, observer=SkepticParams(trust_mode="skeptic", src_FT=8.0)),
        "skeptic8/trick/channels": LayeredRunParams(steps=steps, world=TRICK, observer=SkepticParams(trust_mode="skeptic", src_FT=8.0)),
        "trusting/small-trick": LayeredRunParams(steps=steps, world=SMALL, observer=SkepticParams(trust_mode="trusting")),
        "fool_once/small-trick": LayeredRunParams(steps=steps, world=SMALL, observer=SkepticParams(trust_mode="fool_once")),
        "skeptic4/small-trick": LayeredRunParams(steps=steps, world=SMALL, observer=SkepticParams(trust_mode="skeptic", src_FT=4.0)),
        "skeptic8/small-trick": LayeredRunParams(steps=steps, world=SMALL, observer=SkepticParams(trust_mode="skeptic", src_FT=8.0)),
        "forgiving/small-trick": LayeredRunParams(steps=steps, world=SMALL, observer=SkepticParams(trust_mode="forgiving")),
        "trusting/small-noise": LayeredRunParams(steps=steps, world=SMALL_NOISE, observer=SkepticParams(trust_mode="trusting")),
        "skeptic8/small-noise": LayeredRunParams(steps=steps, world=SMALL_NOISE, observer=SkepticParams(trust_mode="skeptic", src_FT=8.0)),
    }


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", default="results/followup2.json")
    a = ap.parse_args()

    # timing of the original skeptic's false burns, from census2
    c2 = json.load(open("results/census2.json"))
    for name in ("skeptic/noise/channels", "skeptic/trick/channels", "fool_once/noise/channels", "skeptic/trick/plus-band"):
        runs = c2["arms"][name]["runs"]
        print(f"{name:30s} first false burn median {med([r['first_false_burn'] for r in runs])!s:>6}  "
              f"first true burn median {med([r['first_true_burn'] for r in runs])!s:>6}  "
              f"lying frac median {med([r['lying_frac'] for r in runs])}")
    print()

    result = dict(seeds=list(range(a.seeds)), steps=a.steps, arms={})
    for name, rp in arms(a.steps).items():
        rows = [run_layered(rp, s) for s in range(a.seeds)]
        s = summarize(rows)
        result["arms"][name] = dict(summary=s, runs=rows)
        m = lambda k: s[k]["median"]
        print(f"{name:26s} score/1k {m('score_per_1000')!s:>7} hit {round(m('hit_rate'),4) if m('hit_rate') is not None else None!s:>7} "
              f"regimeAcc {m('regime_accuracy')!s:>7} burned {m('n_burned')!s:>4} trueBurnSeeds {s['seeds_with_true_burn']:>2} "
              f"falseBurns {s['total_false_burns']:>3} firstTrue {m('first_true_burn')!s:>6} firstFalse {m('first_false_burn')!s:>6} "
              f"alarms {m('alarms')!s:>4} AR {m('AR_final')!s:>7}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
