"""v0.2 census: the layered network's lenses (exact enumeration) and the optimistic-skeptic observer's
four trust modes on the noise-only and trickster worlds. Descriptive; unregistered.

    python3 run_census2.py --seeds 20 --steps 20000 --out results/census2.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

from node_observer.layered import (
    PLUS, TIMES, Circuit, configuration_table, lens_report,
    ChannelWorldParams, SkepticParams, LayeredRunParams, run_layered,
)

NOISE = ChannelWorldParams(n=8, eps=0.15, p_switch=0.005)
TRICK = ChannelWorldParams(n=8, eps=0.15, p_switch=0.005, trickster=3, lie_on=0.01, lie_len=60.0)
HIGH = ChannelWorldParams(n=8, eps=0.30, p_switch=0.005)


def arms(steps: int):
    out = {}
    for world_name, world in (("noise", NOISE), ("trick", TRICK)):
        for mode in ("trusting", "fool_once", "skeptic", "forgiving"):
            out[f"{mode}/{world_name}/channels"] = LayeredRunParams(
                steps=steps, world=world, observer=SkepticParams(trust_mode=mode))
    for mode in ("trusting", "skeptic"):
        out[f"{mode}/trick/plus-band"] = LayeredRunParams(
            steps=steps, world=TRICK, circuit_ops=(PLUS,), observer=SkepticParams(trust_mode=mode, witness_layer=1))
    out["trusting/noise/times-band"] = LayeredRunParams(
        steps=steps, world=NOISE, circuit_ops=(TIMES,), observer=SkepticParams(trust_mode="trusting", witness_layer=1))
    out["skeptic/noise/channels/AR0=0"] = LayeredRunParams(
        steps=steps, world=NOISE, observer=SkepticParams(trust_mode="skeptic", AR0=0.0))
    out["skeptic/high-noise/channels"] = LayeredRunParams(
        steps=steps, world=HIGH, observer=SkepticParams(trust_mode="skeptic"))
    out["fool_once/high-noise/channels"] = LayeredRunParams(
        steps=steps, world=HIGH, observer=SkepticParams(trust_mode="fool_once"))
    return out


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


KEYS = ["score_per_1000", "hit_rate", "regime_accuracy", "ambiguous_frac", "ambiguous_bet_rate",
        "alarms", "median_delay", "n_burned", "first_true_burn", "first_false_burn", "lying_frac",
        "AR_final", "narrative_rate", "true_rate", "ego_gap", "void"]


def summarize(rows):
    s = {k: dict(median=q([r[k] for r in rows], 0.5), q25=q([r[k] for r in rows], 0.25),
                 q75=q([r[k] for r in rows], 0.75), n=sum(r[k] is not None for r in rows)) for k in KEYS}
    s["seeds_with_true_burn"] = sum(bool(r["true_burns"]) for r in rows)
    s["seeds_with_false_burn"] = sum(bool(r["false_burns"]) for r in rows)
    s["total_false_burns"] = sum(len(r["false_burns"]) for r in rows)
    ar = [r["AR_final"] for r in rows]
    s["AR_sign_split"] = dict(positive=sum(a > 0.05 for a in ar), near_zero=sum(abs(a) <= 0.05 for a in ar),
                              negative=sum(a < -0.05 for a in ar))
    return s


def lenses():
    out = {"configuration_table": configuration_table(), "circuits": {}}
    specs = {
        "tree4 ++": Circuit.tree(4, [PLUS, PLUS]),
        "tree4 **": Circuit.tree(4, [TIMES, TIMES]),
        "tree4 *+": Circuit.tree(4, [TIMES, PLUS]),
        "tree4 +*": Circuit.tree(4, [PLUS, TIMES]),
        "tree8 +++": Circuit.tree(8, [PLUS] * 3),
        "tree8 ***": Circuit.tree(8, [TIMES] * 3),
        "tree8 **+": Circuit.tree(8, [TIMES, TIMES, PLUS]),
        "tree8 ++*": Circuit.tree(8, [PLUS, PLUS, TIMES]),
        "tree8 +*+": Circuit.tree(8, [PLUS, TIMES, PLUS]),
        "tree8 *+*": Circuit.tree(8, [TIMES, PLUS, TIMES]),
        "band8 + (node 0)": Circuit.band(8, [PLUS]),
        "band8 * (node 0)": Circuit.band(8, [TIMES]),
        "band8 ++ (node 0)": Circuit.band(8, [PLUS, PLUS]),
        "band8 +* (node 0)": Circuit.band(8, [PLUS, TIMES]),
    }
    for name, circ in specs.items():
        rep = lens_report(circ)[0]
        out["circuits"][name] = dict(shape=circ.describe(), **rep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", default="results/census2.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    L = lenses()
    print("configuration table:", L["configuration_table"])
    print(f"{'lens':20s} {'shape':>18s} {'zero':>5s} {'vals':>5s} {'H(value)':>9s} {'MI(dir)':>8s} {'H(dir)':>7s} {'MI(par)':>8s}")
    for name, rep in L["circuits"].items():
        print(f"{name:20s} {rep['shape']:>18s} {rep['zero_patterns']:>5d} {rep['distinct_values']:>5d} "
              f"{rep['H_value_bits']:>9.3f} {rep['MI_direction_bits']:>8.3f} {rep['H_direction_bits']:>7.3f} {rep['MI_parity_bits']:>8.3f}")
    print()

    result = dict(seeds=list(range(a.seeds)), steps=a.steps, lenses=L, arms={})
    for name, rp in arms(a.steps).items():
        rows = [run_layered(rp, s) for s in range(a.seeds)]
        s = summarize(rows)
        result["arms"][name] = dict(summary=s, runs=rows)
        m = lambda k: s[k]["median"]
        print(f"{name:34s} score/1k {m('score_per_1000')!s:>7} hit {m('hit_rate')!s:>7} regimeAcc {m('regime_accuracy')!s:>7} "
              f"burned {m('n_burned')!s:>4} trueBurnSeeds {s['seeds_with_true_burn']:>2} falseBurns {s['total_false_burns']:>3} "
              f"firstTrue {m('first_true_burn')!s:>6} alarms {m('alarms')!s:>4} AR {m('AR_final')!s:>7} split {s['AR_sign_split']}")
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
