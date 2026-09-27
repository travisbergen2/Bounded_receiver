"""Fair table: the uniform v0.2 observer (trusting) re-run on the same seeds with the SAME attribution
metrics as the heterogeneous observer (3-step detection window; spurious = alarm with no switch since the
previous alarm). Reads results/hetero.json for the heterogeneous arms.

    python3 run_hetero_fair.py
"""
from __future__ import annotations
import json, random
from node_observer.layered import ChannelWorld, SkepticObserver, SkepticParams, Circuit, G, _sign
from run_census2 import q, NOISE, TRICK

def run_uniform(world_p, steps, seed):
    rng = random.Random(seed)
    world = ChannelWorld(world_p, rng)
    obs = SkepticObserver(SkepticParams(trust_mode="trusting"), Circuit.identity(world_p.n), rng)
    vals = [G(0, s) for s in world.signs]
    ev, cons, e = obs.read(vals); blind = False
    switches = []
    for t in range(steps):
        rec = obs.commit(blind)
        world.step()
        if world.switched_last: switches.append(t + 1)
        vals = [G(0, s) for s in world.signs]
        ev, cons, e = obs.read(vals)
        realized = _sign(sum(v.im_sign() for v in vals))
        obs.settle(rec, realized, world.regime)
        obs.update_trust(ev, cons)
        ev, cons, e = obs.read(vals)
        obs.detect(e); obs.estimate(e)
    c = obs.counts; bets = c["confident"] + c["ambiguous-bet"]; amb = c["ambiguous-bet"] + c["abstain-ambiguous"]
    spurious, attributed, delays, prev = 0, 0, [], 0
    for a in obs.alarms:
        recent = [s for s in switches if prev < s <= a]
        if recent: attributed += 1; delays.append(a - recent[-1])
        else: spurious += 1
        prev = a
    detected = sum(1 for s in switches if any(s <= a <= s + 3 for a in obs.alarms))
    return dict(seed=seed, score_per_1000=round(1000.0 * obs.score / steps, 2),
                hit_rate=(c["hits"] / (c["hits"] + c["misses"])) if (c["hits"] + c["misses"]) else None,
                regime_accuracy=round(obs.regime_agree / obs.regime_checked, 4), ambiguous_frac=round(amb / steps, 4),
                alarms=len(obs.alarms), detected=detected, attributed=attributed, spurious=spurious,
                median_delay=(sorted(delays)[len(delays)//2] if delays else None), max_delay=(max(delays) if delays else None),
                switches=len(switches), AR_final=round(obs.AR, 4))

def med(rows, k): return q([r.get(k) for r in rows], 0.5)

if __name__ == "__main__":
    H = json.load(open("results/hetero.json"))
    out = {}
    print(f"{'arm':40s} {'score/1k':>8} {'hit':>7} {'regAcc':>7} {'held':>6} {'alarms':>6} {'det≤3':>6} {'attrib':>6} {'spur':>5} {'delay':>5} {'maxD':>5} {'switch':>6} {'AR':>7}")
    for wname, world in (("noise", NOISE), ("trick", TRICK)):
        rows = [run_uniform(world, 20000, s) for s in range(20)]
        out[f"uniform/{wname}"] = rows
        for name, rr in ((f"uniform v0.2 EMA/CUSUM/margin / {wname}", rows),
                         (f"hetero alpha=0.05 / {wname}", H["arms"][f"hetero alpha=0.05 / {wname}"]["runs"])):
            print(f"{name:40s} {med(rr,'score_per_1000')!s:>8} {round(med(rr,'hit_rate'),4)!s:>7} {med(rr,'regime_accuracy')!s:>7} "
                  f"{med(rr,'ambiguous_frac')!s:>6} {med(rr,'alarms')!s:>6} {med(rr,'detected')!s:>6} {med(rr,'attributed')!s:>6} "
                  f"{med(rr,'spurious')!s:>5} {med(rr,'median_delay')!s:>5} {med(rr,'max_delay')!s:>5} {med(rr,'switches')!s:>6} {med(rr,'AR_final')!s:>7}")
    json.dump(out, open("results/hetero_uniform_fair.json", "w"), indent=1)
    print("wrote results/hetero_uniform_fair.json")
