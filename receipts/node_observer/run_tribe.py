"""The tribe experiment: eight witnesses read the same honest channels; seven estimate the regime with the
uniform moving average (UE 0.1), one with the exact posterior (v0.6). A judge applies the consensus-relative
trust rules of v0.2 to the witnesses' STATEMENTS (sign of their estimate): who gets burned, and when?

    python3 run_tribe.py --seeds 20 --steps 20000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List

from node_observer.layered import ChannelWorld, ChannelWorldParams, _sign
from node_observer.hetero import HeteroObserver, HeteroParams

WORLD = ChannelWorldParams(n=8, eps=0.15, p_switch=0.005)


def run_tribe(mode: str, src_FT: float, steps: int, seed: int, ue: float = 0.10, n_slow: int = 7) -> Dict:
    rng = random.Random(seed)
    world = ChannelWorld(WORLD, rng)
    E = [0.0] * n_slow                     # the slow witnesses' moving averages
    sharp = HeteroObserver(HeteroParams(), WORLD, random.Random(seed + 1000))
    m = n_slow + 1                         # witness index n_slow = the sharp one
    burned = [False] * m
    burn_time = [None] * m
    S = [0.0] * m
    right = [0] * m
    checked = [0] * m
    switches: List[int] = []
    group_right_all = group_right_slow = group_checked = 0

    def statements() -> List[int]:
        st = [_sign(e) for e in E]
        st.append(_sign(sharp.L))
        return st

    sharp.estimate(world.signs)
    for t in range(steps):
        world.step()
        if world.switched_last:
            switches.append(t + 1)
        e_mean = sum(world.signs) / WORLD.n
        for j in range(n_slow):
            E[j] = (1.0 - ue) * E[j] + ue * e_mean
        sharp.estimate(world.signs)
        st = statements()
        # accuracy vs the hidden regime
        for j in range(m):
            if st[j] != 0:
                checked[j] += 1
                right[j] += (st[j] == world.regime)
        # the group's consensus with and without the sharp member
        cons_all = _sign(sum(st))
        cons_slow = _sign(sum(st[:n_slow]))
        if cons_all != 0 and cons_slow != 0:
            group_checked += 1
            group_right_all += (cons_all == world.regime)
            group_right_slow += (cons_slow == world.regime)
        # the judge: consensus of the TRUSTED witnesses
        trusted = [st[j] for j in range(m) if not burned[j]]
        consensus = _sign(sum(trusted))
        if consensus == 0:
            continue
        for j in range(m):
            if burned[j] or st[j] == 0:
                continue
            contradicts = st[j] == -consensus
            if mode == "fool_once":
                if contradicts:
                    burned[j] = True
                    burn_time[j] = t + 1
            else:
                S[j] = max(0.0, S[j] + (1.0 if contradicts else 0.0) - 0.25)
                if S[j] >= src_FT:
                    burned[j] = True
                    burn_time[j] = t + 1
    acc = [round(right[j] / checked[j], 4) if checked[j] else None for j in range(m)]
    order = sorted([(burn_time[j], j) for j in range(m) if burned[j]])
    first_burned = order[0][1] if order else None
    return dict(
        seed=seed, mode=mode, src_FT=src_FT, switches=len(switches), first_switch=(switches[0] if switches else None),
        sharp_accuracy=acc[-1], slow_accuracy_median=sorted(acc[:n_slow])[n_slow // 2],
        sharp_burned=burned[-1], sharp_burn_time=burn_time[-1],
        sharp_burned_within_first_switch_plus_10=(burned[-1] and switches and burn_time[-1] <= switches[0] + 10),
        first_burned_is_sharp=(first_burned == n_slow),
        n_slow_burned=sum(burned[:n_slow]), slow_burn_times=[bt for bt in burn_time[:n_slow] if bt is not None][:3],
        group_accuracy_with_sharp=round(group_right_all / group_checked, 4) if group_checked else None,
        group_accuracy_without_sharp=round(group_right_slow / group_checked, 4) if group_checked else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", default="results/tribe.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out = {}
    print(f"{'rule':16s} {'sharpAcc':>8} {'slowAcc':>8} {'sharpBurned':>11} {'atFirstSwitch':>13} {'firstBurned=sharp':>17} {'medBurnT':>8} {'medFirstSw':>10} {'slowBurned':>10} {'grpWith':>8} {'grpWithout':>10}")
    for mode, ft in (("fool_once", 0.0), ("skeptic", 4.0), ("skeptic", 8.0)):
        rows = [run_tribe(mode, ft, a.steps, s) for s in range(a.seeds)]
        name = mode if mode == "fool_once" else f"skeptic FT={ft:g}"
        out[name] = rows
        med = lambda k: (sorted(r[k] for r in rows if r[k] is not None) or [None])[len([r for r in rows if r[k] is not None]) // 2]
        print(f"{name:16s} {med('sharp_accuracy')!s:>8} {med('slow_accuracy_median')!s:>8} "
              f"{sum(r['sharp_burned'] for r in rows):>8}/20 {sum(bool(r['sharp_burned_within_first_switch_plus_10']) for r in rows):>10}/20 "
              f"{sum(r['first_burned_is_sharp'] for r in rows):>14}/20 {med('sharp_burn_time')!s:>8} {med('first_switch')!s:>10} "
              f"{med('n_slow_burned')!s:>10} {med('group_accuracy_with_sharp')!s:>8} {med('group_accuracy_without_sharp')!s:>10}")
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
