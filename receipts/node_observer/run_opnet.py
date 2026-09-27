"""v0.3 report: the derived response table, the single-wire monoid, the verified gate ledger with costs,
and the lens census (the op-network between the channel world and the v0.2 observer).

    python3 run_opnet.py --seeds 20 --steps 20000 --out results/opnet.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional

from node_observer.layered import (
    G, PLUS, TIMES, _sign, ChannelWorld, ChannelWorldParams, SkepticObserver, SkepticParams,
)
from node_observer.opnet import OpNet, SILENT, response_table, chain_monoid, describe_map, gate_ledger
from run_census2 import summarize, NOISE, TRICK

ENCODE = {+1: TIMES, -1: PLUS}                 # world sign → operation: up = combine, down = cancel
DECODE = {TIMES: +1, PLUS: -1, SILENT: 0}      # operation → direction evidence


def lens_net(name: str, n: int) -> OpNet:
    if name == "raw":
        return OpNet.identity(n)
    if name in ("U1", "C1", "D1"):
        return OpNet.band(n, [name[0]], fan_in=1)
    if name == "U2":
        return OpNet.band(n, ["U"], fan_in=2)
    if name == "C2":
        return OpNet.band(n, ["C"], fan_in=2)
    raise ValueError(name)


def run_lens(world_p: ChannelWorldParams, obs_p: SkepticParams, lens: str, steps: int, seed: int) -> Dict:
    rng = random.Random(seed)
    net = lens_net(lens, world_p.n)
    world = ChannelWorld(world_p, rng)
    # the observer only needs objects with im_sign(); wrap the decoded evidence as G(0, e)
    from node_observer.layered import Circuit
    obs = SkepticObserver(obs_p, Circuit.identity(net.widths()[-1]), rng)

    def witnesses(signs: List[int]) -> List[G]:
        ops = [ENCODE[s] for s in signs]
        out = net.output(ops)
        return [G(0, DECODE[o]) for o in out]

    switches = []
    lying = 0
    vals = witnesses(world.signs)
    ev, consensus, e = obs.read(vals)
    blind = all(v.im_sign() == 0 for v in vals)
    for t in range(steps):
        rec = obs.commit(blind)
        world.step()
        if world.switched_last:
            switches.append(t + 1)
        lying += world.lying
        vals = witnesses(world.signs)
        ev, consensus, e = obs.read(vals)
        blind = all(v.im_sign() == 0 for v in vals)
        realized = _sign(sum(v.im_sign() for v in vals))
        obs.settle(rec, realized, world.regime)
        obs.update_trust(ev, consensus)
        ev, consensus, e = obs.read(vals)
        obs.detect(e)
        obs.estimate(e)
    c = obs.counts
    bets = c["confident"] + c["ambiguous-bet"]
    amb = c["ambiguous-bet"] + c["abstain-ambiguous"]
    trick = world_p.trickster
    touched = set()
    if trick is not None:
        touched = {trick} if lens in ("raw", "U1", "C1", "D1") else {trick, (trick - 1) % world_p.n}
    burned = [j for j, b in enumerate(obs.burned) if b]
    nr = obs.rec_right / (obs.rec_right + obs.rec_wrong) if (obs.rec_right + obs.rec_wrong) else None
    tr = obs.dec_right / (obs.dec_right + obs.dec_wrong) if (obs.dec_right + obs.dec_wrong) else None
    return dict(
        seed=seed, steps=steps, lens=lens, score=obs.score, score_per_1000=round(1000.0 * obs.score / steps, 2),
        bets=bets, hits=c["hits"], misses=c["misses"],
        hit_rate=(c["hits"] / (c["hits"] + c["misses"])) if (c["hits"] + c["misses"]) else None,
        void=c["void"], void_frac=round(c["void"] / steps, 4), counts=dict(c),
        ambiguous_frac=round(amb / steps, 4), ambiguous_bet_rate=(c["ambiguous-bet"] / amb) if amb else None,
        regime_accuracy=round(obs.regime_agree / obs.regime_checked, 4) if obs.regime_checked else None,
        alarms=len(obs.alarms), switches=len(switches), median_delay=None,
        n_burned=len(burned), burned=burned, true_burns=[j for j in burned if j in touched],
        false_burns=[j for j in burned if j not in touched],
        first_true_burn=min((obs.burn_time[j] for j in burned if j in touched), default=None),
        first_false_burn=min((obs.burn_time[j] for j in burned if j not in touched), default=None),
        lying_frac=round(lying / steps, 4), AR_final=round(obs.AR, 4),
        narrative_rate=None if nr is None else round(nr, 4), true_rate=None if tr is None else round(tr, 4),
        ego_gap=None if (nr is None or tr is None) else round(nr - tr, 4),
        E_final=round(obs.E, 4),
        params=dict(world=asdict(world_p), observer=asdict(obs_p)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", default="results/opnet.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    print("response table (derived):")
    for r in response_table():
        print(f"  {r['type']} {r['pair']:>8}   + → {r['+ value']:>3} class {r['+ class']:>2} sends {r['+ sends']:<6}"
              f"   × → {r['* value']:>3} class {r['* class']:>2} sends {r['* sends']}")
    mon = chain_monoid()
    print(f"\nsingle-wire monoid: {len(mon)} maps (shortest word, rightmost applied first):")
    for word, m in sorted(mon.items(), key=lambda kv: (len(kv[0]), kv[0])):
        print(f"  {word or 'id':>4}: {describe_map(m)}")
    print("\ngate ledger (exhaustively verified):")
    ledger = gate_ledger()
    print(f"  {'gate':16s} {'inputs':>6} {'biases':>6} {'nodes':>5} {'depth':>5} {'patterns':>8} verified")
    for r in ledger:
        print(f"  {r['gate']:16s} {r['inputs']:>6} {r['bias_wires']:>6} {r['nodes']:>5} {r['depth']:>5} {r['patterns']:>8} {r['verified']}")

    arms = {}
    for lens in ("raw", "U1", "C1", "D1"):
        arms[f"trusting/noise/{lens}"] = (NOISE, SkepticParams(trust_mode="trusting"), lens)
    for lens in ("raw", "U2", "C2"):
        arms[f"skeptic4/trick/{lens}"] = (TRICK, SkepticParams(trust_mode="skeptic", src_FT=4.0), lens)
    result = dict(seeds=list(range(a.seeds)), steps=a.steps, response_table=response_table(),
                  monoid={w: describe_map(m) for w, m in mon.items()}, gate_ledger=ledger, arms={})
    print("\nlens census:")
    for name, (wp, op, lens) in arms.items():
        rows = [run_lens(wp, op, lens, a.steps, s) for s in range(a.seeds)]
        s = summarize(rows)
        s["void_frac_median"] = sorted(r["void_frac"] for r in rows)[len(rows) // 2]
        s["E_final_median"] = sorted(r["E_final"] for r in rows)[len(rows) // 2]
        result["arms"][name] = dict(summary=s, runs=rows)
        m = lambda k: s[k]["median"]
        print(f"  {name:26s} score/1k {m('score_per_1000')!s:>7} hit {m('hit_rate')!s:>7} regimeAcc {m('regime_accuracy')!s:>7} "
              f"void {s['void_frac_median']!s:>6} alarms {m('alarms')!s:>4} burned {m('n_burned')!s:>3} "
              f"liarBurnSeeds {s['seeds_with_true_burn']:>2} falseBurns {s['total_false_burns']:>3} firstTrue {m('first_true_burn')!s:>5} "
              f"E {s['E_final_median']!s:>7} AR {m('AR_final')!s:>7}")
    json.dump(result, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
