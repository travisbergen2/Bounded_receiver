"""Seeded census over the registered arms. Descriptive only; nothing is a registered experiment.

    python3 run_census.py --seeds 30 --steps 20000 --out results/census.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import replace

from node_observer.model import (
    PAIR_I, PAIR_W, QuadraticPair, RunParams, WorldParams, ObserverParams, run,
    gaussian_table, eisenstein_table,
)


def arms(steps: int):
    base = RunParams(steps=steps)
    return {
        "honest":        replace(base, observer=ObserverParams(kappa=0.0)),
        "ego_0.5":       replace(base, observer=ObserverParams(kappa=0.5)),
        "ego_0.9":       replace(base, observer=ObserverParams(kappa=0.9)),
        "ego_1.0":       replace(base, observer=ObserverParams(kappa=1.0)),
        "ego_0.9_blinds_detect": replace(base, observer=ObserverParams(kappa=0.9, ego_blinds_detect=True)),
        "honest_no_detect": replace(base, observer=ObserverParams(kappa=0.0, detect_on=False)),
        "honest_1node":  replace(base, n_nodes=1),
        "honest_4nodes": replace(base, n_nodes=4),
        "honest_blind_pair": replace(base, pair=QuadraticPair(1, 1, "gauss")),
        "honest_eisenstein": replace(base, n_nodes=3, pair=PAIR_W, world=WorldParams(n=6)),
        "ego_0.9_eisenstein": replace(base, n_nodes=3, pair=PAIR_W, world=WorldParams(n=6),
                                      observer=ObserverParams(kappa=0.9)),
        "honest_high_noise": replace(base, world=WorldParams(eps=0.40)),
        "ego_0.9_high_noise": replace(base, world=WorldParams(eps=0.40), observer=ObserverParams(kappa=0.9)),
    }


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(rows):
    keys = ["score_per_1000", "hit_rate", "ambiguous_bet_rate", "alarms", "median_delay",
            "false_alarms", "missed", "switches", "AR_final", "narrative_rate", "true_rate",
            "ego_gap", "evidence_zero_frac", "bets", "ambiguous_frac"]
    for r in rows:
        r["ambiguous_frac"] = round(r["ambiguous_total"] / r["steps"], 4)
    out = {}
    for k in keys:
        vals = [r[k] for r in rows]
        out[k] = dict(median=q(vals, 0.5), q25=q(vals, 0.25), q75=q(vals, 0.75),
                      n=sum(v is not None for v in vals))
    ar = [r["AR_final"] for r in rows]
    out["AR_sign_split"] = dict(positive=sum(a > 0.05 for a in ar), near_zero=sum(abs(a) <= 0.05 for a in ar),
                                negative=sum(a < -0.05 for a in ar))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", default="results/census.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    result = dict(seeds=list(range(a.seeds)), steps=a.steps, arms={},
                  gaussian_table=gaussian_table(2), eisenstein_table=eisenstein_table(2))
    for name, rp in arms(a.steps).items():
        rows = [run(rp, s) for s in range(a.seeds)]
        result["arms"][name] = dict(summary=summarize(rows), runs=rows)
        s = result["arms"][name]["summary"]
        print(f"{name:26s} score/1k {s['score_per_1000']['median']!s:>8} hit {s['hit_rate']['median']!s:>7} "
              f"ambBet {s['ambiguous_bet_rate']['median']!s:>7} alarms {s['alarms']['median']!s:>5} "
              f"delay {s['median_delay']['median']!s:>5} FA {s['false_alarms']['median']!s:>4} "
              f"AR {s['AR_final']['median']!s:>7} split {s['AR_sign_split']} "
              f"narr {s['narrative_rate']['median']!s:>7} true {s['true_rate']['median']!s:>7} gap {s['ego_gap']['median']!s:>7}")
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
