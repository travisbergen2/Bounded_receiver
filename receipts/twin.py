"""Python twin of core.js — same PRNG (mulberry32), same operation order, same arithmetic, same parameter
validation. Emits twin.json, the reference the page checks itself against. Pure Python 3.

    python3 twin.py > twin.json
"""
from __future__ import annotations

import json
import math
import sys
from typing import Dict, List

MASK = 0xFFFFFFFF


def mulberry32(seed: int):
    a = seed & MASK

    def rnd() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & MASK
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & MASK
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & MASK)) & MASK
        return ((t ^ (t >> 14)) & MASK) / 4294967296
    return rnd


def sign(x) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def median(arr):
    if not arr:
        return None
    a = sorted(arr)
    m = len(a) // 2
    return a[m] if len(a) % 2 else (a[m - 1] + a[m]) / 2


# ----------------------------------------------------------------------------- validation (mirror of core.js)
def must(cond, msg):
    if not cond:
        raise ValueError("Room: " + msg)


def is_int(x): return isinstance(x, int) and not isinstance(x, bool)
def is_num(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_world(p):
    must(is_int(p["n"]) and p["n"] >= 1, "n must be an integer >= 1")
    must(is_num(p["eps"]) and 0 <= p["eps"] <= 1, "eps must lie in [0, 1]")
    must(is_num(p["pSwitch"]) and 0 <= p["pSwitch"] <= 1, "pSwitch must lie in [0, 1]")
    if p["liar"]:
        must(is_int(p["liarIdx"]) and 0 <= p["liarIdx"] < p["n"], "liarIdx must index a channel when the liar is on")
    must(is_num(p["lieOn"]) and 0 <= p["lieOn"] <= 1, "lieOn must lie in [0, 1]")
    must(is_num(p["lieLen"]) and p["lieLen"] >= 1, "lieLen must be >= 1")


def validate_narrative(o):
    must(is_num(o["tau"]) and o["tau"] > 0, "tau must be > 0")
    must(is_num(o["eta"]) and 0 <= o["eta"] <= 1, "eta must lie in [0, 1]")
    must(is_num(o["kappa"]) and 0 <= o["kappa"] <= 1, "kappa must lie in [0, 1]")
    must(is_num(o["AR0"]), "AR0 must be finite")


def validate_uniform(o):
    validate_narrative(o)
    must(is_num(o["TI"]) and o["TI"] > 0, "TI must be > 0")
    must(is_num(o["FT"]) and o["FT"] > 0, "FT must be > 0")
    must(is_num(o["UE"]) and 0 < o["UE"] <= 1, "UE must lie in (0, 1]")
    must(is_num(o["SG"]), "SG must be finite")
    must(is_num(o["margin"]) and o["margin"] >= 0, "margin must be >= 0")
    must(is_int(o["seamLen"]) and o["seamLen"] >= 0, "seamLen must be an integer >= 0")
    must(is_num(o["boost"]) and o["boost"] >= 1, "boost must be >= 1")


def validate_hetero(o, wp):
    validate_narrative(o)
    must(is_num(o["alpha"]) and 0 < o["alpha"] < 1, "alpha must lie strictly inside (0, 1)")
    must(0 < wp["eps"] < 1, "the heterogeneous receiver needs 0 < eps < 1: a deterministic channel has no finite likelihood ratio")
    must(0 < wp["pSwitch"] < 1, "the heterogeneous receiver needs 0 < pSwitch < 1: its prior mixing bounds the odds only then")


def validate_steps(steps):
    must(is_int(steps) and steps >= 0, "steps must be an integer >= 0")


# ----------------------------------------------------------------------------- Gaussian integers
def g_add(x, y): return (x[0] + y[0], x[1] + y[1])
def g_mul(x, y): return (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0])


def g_str(z) -> str:
    a, b = z
    if b == 0:
        return str(a)
    im = "i" if abs(b) == 1 else f"{abs(b)}i"
    if a == 0:
        return ("-" if b < 0 else "") + im
    return f"{a}{'+' if b > 0 else '-'}{im}"


def pair_info(a, b):
    must(is_int(a) and is_int(b), "a and b must be integers")
    trace, norm = 2 * a, a * a + b * b
    return dict(a=a, b=b, trace=trace, norm=norm, gap=abs(trace - norm), dist2=norm - trace + 1, blind=trace == norm)


def sign_class(z) -> int:
    if z == (0, 0):
        return 0
    return sign(z[0]) if z[0] != 0 else sign(z[1])


def config_table():
    I, NI = (0, 1), (0, -1)
    rows = []
    for name, x, y in (("(+i,-i)", I, NI), ("(+i,+i)", I, I), ("(-i,-i)", NI, NI)):
        p, m = g_add(x, y), g_mul(x, y)
        rows.append(dict(config=name, plus=g_str(p), times=g_str(m), plus_class=sign_class(p), times_class=sign_class(m)))
    return rows


def blind_lattice(radius: int):
    return [[a, b] for a in range(-radius, radius + 1) for b in range(-radius, radius + 1) if pair_info(a, b)["blind"]]


def entropy_bits(counts: Dict) -> float:
    tot = sum(counts.values())
    return -sum((c / tot) * math.log2(c / tot) for c in counts.values() if c > 0)


def mutual_info(pairs) -> float:
    jx, mx, my = {}, {}, {}
    for x, y in pairs:
        k = f"{x}|{y}"
        jx[k] = jx.get(k, 0) + 1; mx[x] = mx.get(x, 0) + 1; my[y] = my.get(y, 0) + 1
    return entropy_bits(mx) + entropy_bits(my) - entropy_bits(jx)


def lens_tree4(op1: str, op2: str):
    ap = lambda op, x, y: g_add(x, y) if op == "+" else g_mul(x, y)
    vals, dirs, pars = [], [], []
    for m in range(16):
        s = [1 if (m >> k) & 1 else -1 for k in range(4)]
        x = [(0, v) for v in s]
        l1 = [ap(op1, x[0], x[1]), ap(op1, x[2], x[3])]
        out = ap(op2, l1[0], l1[1])
        vals.append(g_str(out)); dirs.append(sign(sum(s))); pars.append(sum(1 for v in s if v < 0) % 2)
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    nontie = [(vals[k], dirs[k]) for k in range(16) if dirs[k] != 0]
    blind = all(vals[m] == vals[15 - m] for m in range(16))
    return dict(ops=op1 + op2, zero=vals.count("0"), distinct=len(counts), H_value=entropy_bits(counts),
                MI_sign=mutual_info(nontie), MI_parity=mutual_info(list(zip(vals, pars))), sign_blind=blind)


DEFAULT_WORLD = dict(n=8, eps=0.15, pSwitch=0.005, liar=False, liarIdx=3, lieOn=0.01, lieLen=60)
DEFAULT_UNIFORM = dict(TI=8, SG=1, FT=4, UE=0.1, margin=0.3, seamLen=4, boost=3, AR0=0, eta=0.02, tau=0.15, kappa=0)
DEFAULT_HETERO = dict(alpha=0.05, AR0=0, eta=0.02, tau=0.15, kappa=0)


class World:
    def __init__(self, p, rnd):
        validate_world(p)
        self.n, self.eps, self.pSwitch = p["n"], p["eps"], p["pSwitch"]
        self.liar, self.liarIdx, self.lieOn, self.lieLen = bool(p["liar"]), p["liarIdx"], p["lieOn"], p["lieLen"]
        self.rnd = rnd; self.regime = 1; self.lying = False; self.signs = []; self.switched = False; self.t = 0
        self.step(); self.t = 0; self.switched = False

    def step(self):
        self.switched = False
        if self.rnd() < self.pSwitch:
            self.regime = -self.regime; self.switched = True
        if self.liar:
            if self.lying:
                if self.rnd() < 1 / self.lieLen:
                    self.lying = False
            elif self.rnd() < self.lieOn:
                self.lying = True
        s = []
        for j in range(self.n):
            v = self.regime
            if self.liar and j == self.liarIdx and self.lying:
                v = -v
            if self.rnd() < self.eps:
                v = -v
            s.append(v)
        self.signs = s; self.t += 1


class Narrative:
    def __init__(self, o, rnd):
        self.AR, self.eta, self.tau, self.kappa, self.rnd = o["AR0"], o["eta"], o["tau"], o["kappa"], rnd
        self.recRight = self.recWrong = self.decRight = self.decWrong = 0

    def bet_prob(self): return sigmoid(self.AR / self.tau)

    def settle(self, bet: bool, would_hit: bool):
        right = would_hit if bet else (not would_hit)
        if right: self.decRight += 1
        else: self.decWrong += 1
        p_rec = (1 + self.kappa) / 2 if right else (1 - self.kappa) / 2
        if self.rnd() < p_rec:
            if right: self.recRight += 1
            else: self.recWrong += 1
            self.AR = (1 - self.eta) * self.AR + self.eta * (1 if would_hit else -1)

    def narrative_rate(self):
        t = self.recRight + self.recWrong
        return self.recRight / t if t else None

    def true_rate(self):
        t = self.decRight + self.decWrong
        return self.decRight / t if t else None


class Uniform:
    def __init__(self, o, rnd):
        validate_uniform(o)
        self.o, self.rnd = o, rnd
        self.E = 0.0; self.S = 0.0; self.seam = 0; self.t = 0; self.score = 0
        self.hits = self.misses = self.voids = self.confident = self.ambBet = self.ambAbstain = self.seamAbstain = 0
        self.alarms = []; self.regimeAgree = self.regimeChecked = 0; self.nar = Narrative(o, rnd)

    def commit(self):
        d = sign(self.E) if abs(self.E) > 1e-12 else (1 if self.rnd() < 0.5 else -1)
        if self.seam > 0: return ("seam", d)
        if abs(self.E) >= self.o["margin"]: return ("confident", d)
        return (("ambBet" if self.rnd() < self.nar.bet_prob() else "ambAbstain"), d)

    def settle(self, rec, realized, regime):
        kind, pred = rec
        if kind == "seam": self.seamAbstain += 1
        elif kind == "confident": self.confident += 1
        elif kind == "ambBet": self.ambBet += 1
        else: self.ambAbstain += 1
        if self.seam > 0: self.seam -= 1
        self.t += 1
        if abs(self.E) > 1e-12:
            self.regimeChecked += 1
            if sign(self.E) == regime: self.regimeAgree += 1
        if realized == 0:
            self.voids += 1; return
        would_hit = pred == realized
        if kind in ("confident", "ambBet"):
            if would_hit: self.hits += 1; self.score += 1
            else: self.misses += 1; self.score -= 1
        if kind in ("ambBet", "ambAbstain"):
            self.nar.settle(kind == "ambBet", would_hit)

    def observe(self, e):
        d = sign(self.E) if abs(self.E) > 1e-12 else 0
        self.S = max(0.0, self.S - e * d - 1 / self.o["TI"])
        if self.S >= self.o["FT"]:
            self.S = 0.0; self.E = 0.0; self.seam = self.o["seamLen"]; self.alarms.append(self.t)
        ue = min(1.0, self.o["UE"] * self.o["boost"]) if self.seam > 0 else self.o["UE"]
        self.E = (1 - ue) * self.E + ue * self.o["SG"] * e


def mix_odds(o, ps):
    return (o * (1 - ps) + ps) / (o * ps + (1 - ps))


class Hetero:
    def __init__(self, o, wp, rnd):
        validate_hetero(o, wp)
        self.o, self.rnd = o, rnd
        self.w = (1 - wp["eps"]) / wp["eps"]; self.band = (1 - o["alpha"]) / o["alpha"]; self.pSwitch = wp["pSwitch"]
        self.odds = 1.0; self.prevMap = 0; self.t = 0; self.score = 0
        self.hits = self.misses = self.voids = self.confident = self.ambBet = self.ambAbstain = 0
        self.alarms = []; self.regimeAgree = self.regimeChecked = 0; self.nar = Narrative(o, rnd)

    def L(self): return math.log(self.odds)

    def commit(self):
        d = sign(self.odds - 1) if self.odds != 1 else (1 if self.rnd() < 0.5 else -1)
        if self.odds >= self.band or self.odds <= 1 / self.band: return ("confident", d)
        return (("ambBet" if self.rnd() < self.nar.bet_prob() else "ambAbstain"), d)

    def settle(self, rec, realized, regime):
        kind, pred = rec
        if kind == "confident": self.confident += 1
        elif kind == "ambBet": self.ambBet += 1
        else: self.ambAbstain += 1
        self.t += 1
        if self.odds != 1:
            self.regimeChecked += 1
            if sign(self.odds - 1) == regime: self.regimeAgree += 1
        if realized == 0:
            self.voids += 1; return
        would_hit = pred == realized
        if kind in ("confident", "ambBet"):
            if would_hit: self.hits += 1; self.score += 1
            else: self.misses += 1; self.score -= 1
        if kind in ("ambBet", "ambAbstain"):
            self.nar.settle(kind == "ambBet", would_hit)

    def observe(self, signs):
        odds = mix_odds(self.odds, self.pSwitch)
        for s in signs:
            if s > 0: odds *= self.w
            else: odds /= self.w
        self.odds = odds
        cur = sign(self.odds - 1)
        flip = self.prevMap != 0 and cur != 0 and cur != self.prevMap
        if cur != 0: self.prevMap = cur
        if flip: self.alarms.append(self.t)


def summarize(obs, is_hetero):
    bets = obs.confident + obs.ambBet; amb = obs.ambBet + obs.ambAbstain; scored = obs.hits + obs.misses
    s = dict(score=obs.score, hits=obs.hits, misses=obs.misses, voids=obs.voids, bets=bets,
             scored_bets=scored, voided_bets=bets - scored,
             hit_rate=(obs.hits / scored) if scored else None, ambiguous=amb,
             ambiguous_bet_rate=(obs.ambBet / amb) if amb else None, alarms=len(obs.alarms),
             regime_accuracy=(obs.regimeAgree / obs.regimeChecked) if obs.regimeChecked else None,
             AR_final=obs.nar.AR, narrative_rate=obs.nar.narrative_rate(), true_rate=obs.nar.true_rate(),
             recorded=obs.nar.recRight + obs.nar.recWrong, resolved=obs.nar.decRight + obs.nar.decWrong,
             seam_abstain=0 if is_hetero else obs.seamAbstain)
    if is_hetero:
        s["odds_final"] = obs.odds
        s["odds_finite"] = math.isfinite(obs.odds) and obs.odds > 0
    return s


def attribution(alarms, switches):
    spurious = attributed = 0; delays = []; prev = 0
    for a in alarms:
        recent = [s for s in switches if prev < s <= a]
        if recent: attributed += 1; delays.append(a - recent[-1])
        else: spurious += 1
        prev = a
    return dict(attributed=attributed, spurious=spurious, median_delay=median(delays))


def run(seed, steps, wp=None, up=None, hp=None):
    must(is_int(seed) and seed >= 0, "seed must be an integer >= 0")
    validate_steps(steps)
    wp = {**DEFAULT_WORLD, **(wp or {})}; up = {**DEFAULT_UNIFORM, **(up or {})}; hp = {**DEFAULT_HETERO, **(hp or {})}
    rndW, rndU, rndH = mulberry32(seed), mulberry32(seed + 1000), mulberry32(seed + 2000)
    world = World(wp, rndW); U = Uniform(up, rndU); H = Hetero(hp, wp, rndH)
    switches = []
    U.observe(sum(world.signs) / world.n); H.observe(world.signs)
    for t in range(steps):
        ru, rh = U.commit(), H.commit()
        world.step()
        if world.switched: switches.append(t + 1)
        realized = sign(sum(world.signs))
        U.settle(ru, realized, world.regime); H.settle(rh, realized, world.regime)
        U.observe(sum(world.signs) / world.n); H.observe(world.signs)
    return dict(seed=seed, steps=steps, switches=len(switches),
                uniform={**summarize(U, False), **attribution(U.alarms, switches)},
                hetero={**summarize(H, True), **attribution(H.alarms, switches)})


def tribe(seed, steps, wp=None, rule="strike", FT=0.0, n_slow=None):
    must(is_int(seed) and seed >= 0, "seed must be an integer >= 0")
    validate_steps(steps)
    wp = {**DEFAULT_WORLD, **(wp or {})}; validate_world(wp)
    must(rule in ("strike", "cusum"), "rule must be 'strike' or 'cusum'")
    if rule == "cusum": must(is_num(FT) and FT > 0, "FT must be > 0 for the cusum rule")
    n_slow = 7 if n_slow is None else n_slow
    must(is_int(n_slow) and n_slow >= 1, "nSlow must be an integer >= 1 (no silent default for 0)")
    world = World(wp, mulberry32(seed))
    m = n_slow + 1; E = [0.0] * n_slow
    sharp = Hetero(DEFAULT_HETERO, wp, mulberry32(seed + 3000))
    burned = [False] * m; burn_time = [None] * m; S = [0.0] * m; right = [0] * m; checked = [0] * m
    switches = []; grp_all = grp_slow = grp_checked = 0
    sharp.observe(world.signs)
    for t in range(steps):
        world.step()
        if world.switched: switches.append(t + 1)
        e_mean = sum(world.signs) / wp["n"]
        for j in range(n_slow): E[j] = 0.9 * E[j] + 0.1 * e_mean
        sharp.observe(world.signs)
        st = [sign(e) for e in E] + [sign(sharp.odds - 1)]
        for j in range(m):
            if st[j] != 0:
                checked[j] += 1
                if st[j] == world.regime: right[j] += 1
        c_all, c_slow = sign(sum(st)), sign(sum(st[:n_slow]))
        if c_all != 0 and c_slow != 0:
            grp_checked += 1
            if c_all == world.regime: grp_all += 1
            if c_slow == world.regime: grp_slow += 1
        trusted = sum(st[j] for j in range(m) if not burned[j])
        consensus = sign(trusted)
        if consensus == 0: continue
        for j in range(m):
            if burned[j] or st[j] == 0: continue
            contradicts = st[j] == -consensus
            if rule == "strike":
                if contradicts: burned[j] = True; burn_time[j] = t + 1
            else:
                S[j] = max(0.0, S[j] + (1.0 if contradicts else 0.0) - 0.25)
                if S[j] >= FT: burned[j] = True; burn_time[j] = t + 1
    return dict(seed=seed, steps=steps, rule=rule, FT=FT, nSlow=n_slow, switches=len(switches),
                first_switch=(switches[0] if switches else None),
                accuracy=[(right[j] / checked[j]) if checked[j] else None for j in range(m)],
                burned=burned, burn_time=burn_time, sharp_burned=burned[n_slow], sharp_burn_time=burn_time[n_slow],
                slow_burned=sum(burned[:n_slow]),
                group_with=(grp_all / grp_checked) if grp_checked else None,
                group_without=(grp_slow / grp_checked) if grp_checked else None)


# ----------------------------------------------------------------------------- self-check cases
SELFCHECK_RUNS = [
    dict(seed=1, steps=4000, wp={}, up={}, hp={}),
    dict(seed=2, steps=4000, wp={"liar": True}, up={}, hp={}),
    dict(seed=3, steps=4000, wp={}, up={"kappa": 0.9, "AR0": 0.5}, hp={"kappa": 0.9, "AR0": 0.5}),
    dict(seed=7, steps=60000, wp={"eps": 0.02, "pSwitch": 0.001}, up={}, hp={}),   # long run, small noise: odds stay finite
    dict(seed=8, steps=2000, wp={"n": 2, "eps": 0.5}, up={}, hp={}),                # tie-heavy world: voids and denominators
    dict(seed=9, steps=3000, wp={"n": 5, "eps": 0.3, "pSwitch": 0.02}, up={}, hp={}),  # odd n, faster switching
]
SELFCHECK_TRIBES = [dict(seed=4, steps=4000, rule="strike", FT=0.0), dict(seed=4, steps=4000, rule="cusum", FT=4.0),
                    dict(seed=4, steps=4000, rule="cusum", FT=8.0), dict(seed=5, steps=3000, rule="strike", FT=0.0, nSlow=3)]

# parameter sets that MUST be rejected (the JavaScript core must throw for the same ones)
INVALID_CASES = [
    dict(kind="run", args=dict(wp={"n": 0})), dict(kind="run", args=dict(wp={"eps": 1.5})), dict(kind="run", args=dict(wp={"pSwitch": -0.1})),
    dict(kind="run", args=dict(wp={"eps": 0.0})), dict(kind="run", args=dict(wp={"eps": 1.0})), dict(kind="run", args=dict(wp={"pSwitch": 0.0})),
    dict(kind="run", args=dict(hp={"alpha": 0.0})), dict(kind="run", args=dict(hp={"alpha": 1.0})), dict(kind="run", args=dict(hp={"tau": 0.0})),
    dict(kind="run", args=dict(up={"TI": 0})), dict(kind="run", args=dict(up={"tau": 0})), dict(kind="run", args=dict(up={"UE": 0})),
    dict(kind="run", args=dict(steps=-1)), dict(kind="run", args=dict(steps=2.5)),
    dict(kind="tribe", args=dict(rule="strik")), dict(kind="tribe", args=dict(nSlow=0)), dict(kind="tribe", args=dict(rule="cusum", FT=0)),
]


def invalid_case_raises(case) -> bool:
    try:
        if case["kind"] == "run":
            a = case["args"]
            run(1, a.get("steps", 100), a.get("wp"), a.get("up"), a.get("hp"))
        else:
            a = case["args"]
            tribe(1, 100, None, a.get("rule", "strike"), a.get("FT", 0.0), a.get("nSlow"))
        return False
    except ValueError:
        return True
    except TypeError:
        return True


def mixing_bound_receipt():
    rows = []
    for ps in (0.005, 0.2):
        lo, hi = ps / (1 - ps), (1 - ps) / ps
        for o in (1e-12, 1e-6, 0.5, 1.0, 7.0, 1e6, 1e12):
            m = mix_odds(o, ps)
            rows.append(dict(ps=ps, odds=o, mixed=m, inside=(lo - 1e-12 <= m <= hi + 1e-12)))
    return rows


def main():
    out = dict(
        version="Room X twin v2 (2026-09-27; validation, conventional median, boundary receipts)",
        config_table=config_table(), blind_lattice=blind_lattice(3),
        lenses=[lens_tree4(a, b) for a, b in (("+", "+"), ("*", "*"), ("*", "+"), ("+", "*"))],
        runs=[dict(params=r, result=run(r["seed"], r["steps"], r["wp"], r["up"], r["hp"])) for r in SELFCHECK_RUNS],
        tribes=[dict(params=tp, result=tribe(tp["seed"], tp["steps"], None, tp["rule"], tp["FT"], tp.get("nSlow"))) for tp in SELFCHECK_TRIBES],
        prng_first5=(lambda g: [g() for _ in range(5)])(mulberry32(1)),
        boundary=dict(invalid_cases=[dict(case=c, raises=invalid_case_raises(c)) for c in INVALID_CASES],
                      mixing=mixing_bound_receipt()),
    )
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
