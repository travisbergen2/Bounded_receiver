"""v0.2 — the signal passes THROUGH a layered network of Gaussian-integer nodes, and the observer at
the end is built "like Travis": optimistic by default, skeptical in the checking, one-strike on trust.

NETWORK   an arithmetic circuit over Z[i]. The hidden world injects a sign on each of n input
          channels (x_j = +i or -i). Every node is typed '+' or '*' and combines two parents from the
          previous layer; values propagate layer by layer; the observer reads one layer's node values
          (its "witnesses"). The three configurations a 2-input node can see are Travis's
          (+i,-i), (+i,+i), (-i,-i):   '+' gives 0, 2i, -2i    '*' gives 1, -1, -1.
          So '+' keeps the direction and cancels the conjugate pair; '*' keeps agreement and is
          sign-blind (i*i = (-i)*(-i)). Trees and bands of these gates compute counts, parities,
          agreement counts and all-agree tests — enumerated exactly by `lens_report`.

WORLD     a hidden regime r in {+1,-1} sets every channel's sign; each channel flips independently
          with probability eps (noise). Optionally one channel is a TRICKSTER that lies in bursts
          (structured deception: sends -r*i for a geometric stretch), on top of the noise.

OBSERVER  the three blocks of v0.1 (ESTIMATE / DETECT / COMMIT, prior AR from AR0) fed by a
          trust-weighted consensus of the witnesses, with four trust modes:
            trusting   every witness weight 1 forever
            fool_once  the first time a witness contradicts the trusted consensus it is burned forever
            skeptic    a per-witness CUSUM on contradiction (drift 1/src_TI, threshold src_FT); when it
                       fires the witness is burned forever — "fool me once", but only for STRUCTURED
                       fooling; isolated noise does not accumulate
            forgiving  an EMA reliability weight that recovers when the witness agrees again
          A regime switch flips all honest witnesses together, so the consensus flips with them and
          no witness is read as lying: consensus-relative contradiction separates "the world changed"
          from "this source misled me".
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

PLUS, TIMES = "+", "*"


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ----------------------------------------------------------------------------- Gaussian integers


@dataclass(frozen=True)
class G:
    """a + b i with exact integer coordinates."""

    a: int
    b: int

    def __add__(self, o: "G") -> "G":
        return G(self.a + o.a, self.b + o.b)

    def __mul__(self, o: "G") -> "G":
        return G(self.a * o.a - self.b * o.b, self.a * o.b + self.b * o.a)

    def __neg__(self) -> "G":
        return G(-self.a, -self.b)

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0

    @property
    def norm(self) -> int:
        return self.a * self.a + self.b * self.b

    def kind(self) -> str:
        """Axis class: '0', '+1', '-1', '+i', '-i', or 'off' (off the axes)."""
        if self.is_zero():
            return "0"
        if self.b == 0:
            return "+1" if self.a > 0 else "-1"
        if self.a == 0:
            return "+i" if self.b > 0 else "-i"
        return "off"

    def im_sign(self) -> int:
        return _sign(self.b)

    def re_sign(self) -> int:
        return _sign(self.a)

    def __str__(self) -> str:
        if self.b == 0:
            return str(self.a)
        im = "i" if abs(self.b) == 1 else f"{abs(self.b)}i"
        if self.a == 0:
            return ("-" if self.b < 0 else "") + im
        return f"{self.a}{'+' if self.b > 0 else '-'}{im}"


I = G(0, 1)
NEG_I = G(0, -1)
ZERO = G(0, 0)
ONE = G(1, 0)


def apply(op: str, x: G, y: G) -> G:
    if op == PLUS:
        return x + y
    if op == TIMES:
        return x * y
    raise ValueError(op)


def configuration_table() -> List[Dict[str, str]]:
    """Travis's three configurations under the two operations."""
    rows = []
    for name, (x, y) in (("(+i,-i)", (I, NEG_I)), ("(+i,+i)", (I, I)), ("(-i,-i)", (NEG_I, NEG_I))):
        rows.append(dict(config=name, plus=str(x + y), times=str(x * y),
                         plus_kind=(x + y).kind(), times_kind=(x * y).kind()))
    return rows


# ----------------------------------------------------------------------------- circuits


@dataclass(frozen=True)
class Gate:
    op: str
    p: int  # parent index in the previous layer
    q: int


class Circuit:
    def __init__(self, n_inputs: int, layers: Sequence[Sequence[Gate]]):
        self.n = n_inputs
        self.layers = [list(l) for l in layers]
        width = n_inputs
        for L, layer in enumerate(self.layers):
            for g in layer:
                if g.op not in (PLUS, TIMES):
                    raise ValueError(g.op)
                if not (0 <= g.p < width and 0 <= g.q < width):
                    raise ValueError(f"layer {L}: parent out of range")
            width = len(layer)

    @property
    def depth(self) -> int:
        return len(self.layers)

    def widths(self) -> List[int]:
        return [self.n] + [len(l) for l in self.layers]

    def forward(self, xs: Sequence[G]) -> List[List[G]]:
        if len(xs) != self.n:
            raise ValueError("wrong number of inputs")
        vals = [list(xs)]
        for layer in self.layers:
            prev = vals[-1]
            vals.append([apply(g.op, prev[g.p], prev[g.q]) for g in layer])
        return vals

    def output(self, xs: Sequence[G]) -> List[G]:
        return self.forward(xs)[-1]

    def truth_table(self) -> List[Tuple[Tuple[int, ...], List[G]]]:
        """All 2^n sign patterns (+1 -> +i, -1 -> -i) with the output-layer values. n <= 14."""
        if self.n > 14:
            raise ValueError("truth table too large")
        rows = []
        for signs in product((+1, -1), repeat=self.n):
            xs = [G(0, s) for s in signs]
            rows.append((signs, self.output(xs)))
        return rows

    def describe(self) -> str:
        return " -> ".join(str(w) for w in self.widths()) + " ops " + "".join(
            "".join(sorted(set(g.op for g in layer))) for layer in self.layers)

    # builders
    @staticmethod
    def tree(n: int, ops: Sequence[str]) -> "Circuit":
        """Binary tree: layer k pairs neighbours (2j, 2j+1) with op ops[k]; n must be 2^len(ops)."""
        if n != 2 ** len(ops):
            raise ValueError("tree needs n = 2^depth")
        layers, width = [], n
        for op in ops:
            layers.append([Gate(op, 2 * j, 2 * j + 1) for j in range(width // 2)])
            width //= 2
        return Circuit(n, layers)

    @staticmethod
    def band(n: int, ops: Sequence[str]) -> "Circuit":
        """Constant-width band: every layer has n gates, gate j = op(j, j+1 mod n)."""
        return Circuit(n, [[Gate(op, j, (j + 1) % n) for j in range(n)] for op in ops])

    @staticmethod
    def identity(n: int) -> "Circuit":
        return Circuit(n, [])


def _entropy(counts: Dict[Any, int]) -> float:
    tot = sum(counts.values())
    return -sum(c / tot * math.log2(c / tot) for c in counts.values() if c)


def _mutual_information(pairs: List[Tuple[Any, Any]]) -> float:
    joint: Dict[Tuple[Any, Any], int] = {}
    mx: Dict[Any, int] = {}
    my: Dict[Any, int] = {}
    for x, y in pairs:
        joint[(x, y)] = joint.get((x, y), 0) + 1
        mx[x] = mx.get(x, 0) + 1
        my[y] = my.get(y, 0) + 1
    return _entropy(mx) + _entropy(my) - _entropy(joint)


def lens_report(circ: Circuit) -> List[Dict[str, Any]]:
    """For each output node, under uniform input signs: how many patterns cancel to 0, how many
    distinct values appear, the entropy of the value (bits per step the lens can carry), and how
    much it reveals about the world's direction (sign of the input sum) and about the parity of
    the number of -i inputs. Exact enumeration."""
    tt = circ.truth_table()
    n_out = len(tt[0][1])
    report = []
    for k in range(n_out):
        vals = [row[1][k] for row in tt]
        dirs = [_sign(sum(row[0])) for row in tt]          # -1, 0 (tie), +1
        pars = [sum(1 for s in row[0] if s < 0) % 2 for row in tt]
        counts: Dict[str, int] = {}
        for v in vals:
            counts[str(v)] = counts.get(str(v), 0) + 1
        # sign-only: restrict to patterns with a nonzero sum. A sign-blind lens (invariant under the
        # global flip x -> -x) has exactly zero MI here, while it may still see the TIE: a tie has an
        # even number of -i inputs, so a parity lens rules ties in or out.
        nontie = [(str(v), d) for v, d in zip(vals, dirs) if d != 0]
        report.append(dict(
            node=k,
            patterns=len(tt),
            zero_patterns=sum(v.is_zero() for v in vals),
            distinct_values=len(counts),
            H_value_bits=round(_entropy(counts), 4),
            H_direction_bits=round(_entropy({d: dirs.count(d) for d in set(dirs)}), 4),
            MI_direction_bits=round(_mutual_information([(str(v), d) for v, d in zip(vals, dirs)]), 4),
            MI_sign_bits=round(_mutual_information(nontie), 4) if nontie else 0.0,
            MI_parity_bits=round(_mutual_information([(str(v), p) for v, p in zip(vals, pars)]), 4),
            sign_blind=all(str(v) == str(circ.output([G(0, -s) for s in row[0]])[k]) for row, v in zip(tt, vals)),
            values_sample=sorted(counts.items(), key=lambda kv: -kv[1])[:6],
        ))
    return report


# ----------------------------------------------------------------------------- world


@dataclass
class ChannelWorldParams:
    n: int = 8
    eps: float = 0.15          # independent flip probability per channel per step (noise)
    p_switch: float = 0.005    # regime flip probability per step
    trickster: Optional[int] = None  # channel index that lies in bursts, or None
    lie_on: float = 0.01       # per-step probability a quiet trickster starts a lying burst
    lie_len: float = 60.0      # mean burst length (geometric)
    regime0: int = +1


class ChannelWorld:
    def __init__(self, params: ChannelWorldParams, rng: random.Random):
        self.p = params
        self.rng = rng
        self.regime = params.regime0
        self.lying = False
        self.t = 0
        self.switched_last = False
        self.signs: List[int] = []
        self.step()  # initial signs
        self.t = 0
        self.switched_last = False

    def step(self) -> List[G]:
        p = self.p
        self.switched_last = False
        if self.rng.random() < p.p_switch:
            self.regime = -self.regime
            self.switched_last = True
        if p.trickster is not None:
            if self.lying:
                if self.rng.random() < 1.0 / p.lie_len:
                    self.lying = False
            elif self.rng.random() < p.lie_on:
                self.lying = True
        signs = []
        for j in range(p.n):
            s = self.regime
            if p.trickster is not None and j == p.trickster and self.lying:
                s = -s
            if self.rng.random() < p.eps:
                s = -s
            signs.append(s)
        self.signs = signs
        self.t += 1
        return [G(0, s) for s in signs]

    def majority(self) -> int:
        return _sign(sum(self.signs))


# ----------------------------------------------------------------------------- observer


@dataclass
class SkepticParams:
    TI: float = 8.0
    SG: float = 1.0
    FT: float = 4.0
    UE: float = 0.10
    margin: float = 0.30
    eta: float = 0.02
    tau: float = 0.15
    kappa: float = 0.0
    seam_len: int = 4
    reset_boost: float = 3.0
    AR0: float = 0.5           # optimistic start ("like me"); 0.0 = blank
    trust_mode: str = "skeptic"  # trusting | fool_once | skeptic | forgiving
    src_TI: float = 4.0        # per-witness CUSUM drift = 1/src_TI
    src_FT: float = 4.0        # per-witness CUSUM threshold
    forgive_rate: float = 0.05  # EMA rate for the forgiving reliability weight
    witness_layer: int = 0     # which circuit layer the observer reads (0 = the channels)


class SkepticObserver:
    """v0.1's three blocks, fed by a trust-weighted consensus of witnesses."""

    def __init__(self, params: SkepticParams, circ: Circuit, rng: random.Random):
        self.p = params
        self.circ = circ
        self.rng = rng
        self.m = circ.widths()[params.witness_layer]
        if params.trust_mode not in ("trusting", "fool_once", "skeptic", "forgiving"):
            raise ValueError(params.trust_mode)
        # trust state
        self.burned = [False] * self.m
        self.burn_time: List[Optional[int]] = [None] * self.m
        self.S_src = [0.0] * self.m
        self.rel = [1.0] * self.m       # forgiving: EMA of agreement, starts fully trusted
        # blocks
        self.E = 0.0
        self.S = 0.0
        self.seam = 0
        self.AR = params.AR0
        self.t = 0
        self.alarms: List[int] = []
        self.score = 0
        self.counts: Dict[str, int] = {k: 0 for k in (
            "confident", "ambiguous-bet", "abstain-ambiguous", "abstain-seam", "abstain-blind",
            "hits", "misses", "void")}
        self.rec_right = self.rec_wrong = self.dec_right = self.dec_wrong = 0
        self.regime_agree = 0
        self.regime_checked = 0
        self.AR_trace: List[float] = []

    # -- witnesses
    def weights(self) -> List[float]:
        if self.p.trust_mode == "forgiving":
            return [max(0.0, 2.0 * r - 1.0) for r in self.rel]
        return [0.0 if b else 1.0 for b in self.burned]

    def read(self, layer_vals: Sequence[G]) -> Tuple[List[int], int, float]:
        """Witness direction evidence (sign of the imaginary part), the trusted consensus sign,
        and the weighted mean evidence e in [-1, 1]."""
        ev = [v.im_sign() for v in layer_vals]
        w = self.weights()
        tot_w = sum(w)
        weighted = sum(wi * ei for wi, ei in zip(w, ev))
        consensus = _sign(weighted)
        e = weighted / tot_w if tot_w > 0 else 0.0
        return ev, consensus, e

    def update_trust(self, ev: Sequence[int], consensus: int) -> None:
        p = self.p
        if consensus == 0 or p.trust_mode == "trusting":
            return
        for j, e_j in enumerate(ev):
            if e_j == 0:
                continue
            contradicts = e_j == -consensus
            if p.trust_mode == "forgiving":
                self.rel[j] = (1.0 - p.forgive_rate) * self.rel[j] + p.forgive_rate * (0.0 if contradicts else 1.0)
                continue
            if self.burned[j]:
                continue
            if p.trust_mode == "fool_once":
                if contradicts:
                    self.burned[j] = True
                    self.burn_time[j] = self.t
            else:  # skeptic: structured contradiction only
                self.S_src[j] = max(0.0, self.S_src[j] + (1.0 if contradicts else 0.0) - 1.0 / p.src_TI)
                if self.S_src[j] >= p.src_FT:
                    self.burned[j] = True
                    self.burn_time[j] = self.t

    # -- blocks (as v0.1, with continuous evidence)
    def detect(self, e: float) -> bool:
        p = self.p
        d = _sign(self.E) if abs(self.E) > 1e-12 else 0
        self.S = max(0.0, self.S - e * d - 1.0 / p.TI)
        if self.S >= p.FT:
            self.S = 0.0
            self.E = 0.0
            self.seam = p.seam_len
            self.alarms.append(self.t)
            return True
        return False

    def estimate(self, e: float) -> None:
        p = self.p
        ue = min(1.0, p.UE * p.reset_boost) if self.seam > 0 else p.UE
        self.E = (1.0 - ue) * self.E + ue * p.SG * e

    def commit(self, blind: bool) -> Dict[str, Any]:
        p = self.p
        rec: Dict[str, Any] = dict(kind=None, pred=0)
        if blind:
            rec["kind"] = "abstain-blind"
            return rec
        if self.seam > 0:
            rec["kind"] = "abstain-seam"
            return rec
        d = _sign(self.E) if abs(self.E) > 1e-12 else self.rng.choice((+1, -1))
        rec["pred"] = d
        if abs(self.E) >= p.margin:
            rec["kind"] = "confident"
        elif self.rng.random() < _sigmoid(self.AR / p.tau):
            rec["kind"] = "ambiguous-bet"
        else:
            rec["kind"] = "abstain-ambiguous"
        return rec

    def settle(self, rec: Dict[str, Any], realized: int, regime: int) -> None:
        p = self.p
        kind = rec["kind"]
        self.counts[kind] += 1
        if self.seam > 0:
            self.seam -= 1
        self.t += 1
        if abs(self.E) > 1e-12:
            self.regime_checked += 1
            self.regime_agree += (_sign(self.E) == regime)
        if realized == 0:
            self.counts["void"] += 1
            return
        if kind in ("confident", "ambiguous-bet"):
            hit = rec["pred"] == realized
            self.counts["hits" if hit else "misses"] += 1
            self.score += 1 if hit else -1
        if kind not in ("ambiguous-bet", "abstain-ambiguous"):
            return
        bet = kind == "ambiguous-bet"
        would_hit = rec["pred"] == realized
        decision_right = would_hit if bet else (not would_hit)
        if decision_right:
            self.dec_right += 1
        else:
            self.dec_wrong += 1
        p_record = (1.0 + p.kappa) / 2.0 if decision_right else (1.0 - p.kappa) / 2.0
        if self.rng.random() < p_record:
            if decision_right:
                self.rec_right += 1
            else:
                self.rec_wrong += 1
            self.AR = (1.0 - p.eta) * self.AR + p.eta * (1 if would_hit else -1)


# ----------------------------------------------------------------------------- run


@dataclass
class LayeredRunParams:
    steps: int = 20000
    world: ChannelWorldParams = field(default_factory=ChannelWorldParams)
    observer: SkepticParams = field(default_factory=SkepticParams)
    circuit_ops: Tuple[str, ...] = ()   # band layers applied to the channels, e.g. ('+',) ; () = read the channels
    ar_trace_every: int = 200


def run_layered(params: LayeredRunParams, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    circ = Circuit.band(params.world.n, params.circuit_ops) if params.circuit_ops else Circuit.identity(params.world.n)
    world = ChannelWorld(params.world, rng)
    obs = SkepticObserver(params.observer, circ, rng)
    L = params.observer.witness_layer
    switches: List[int] = []
    lying_steps = 0

    xs = [G(0, s) for s in world.signs]
    layer_vals = circ.forward(xs)[L]
    ev, consensus, e = obs.read(layer_vals)
    blind = all(v.im_sign() == 0 for v in layer_vals)
    for t in range(params.steps):
        rec = obs.commit(blind)
        xs = world.step()
        if world.switched_last:
            switches.append(t + 1)
        lying_steps += world.lying
        layer_vals = circ.forward(xs)[L]
        ev, consensus, e = obs.read(layer_vals)
        blind = all(v.im_sign() == 0 for v in layer_vals)
        realized = _sign(sum(v.im_sign() for v in layer_vals))  # the observable outcome
        obs.settle(rec, realized, world.regime)
        obs.update_trust(ev, consensus)
        ev, consensus, e = obs.read(layer_vals)  # re-read with updated weights
        obs.detect(e)
        obs.estimate(e)
        if (t + 1) % params.ar_trace_every == 0:
            obs.AR_trace.append(round(obs.AR, 4))

    c = obs.counts
    bets = c["confident"] + c["ambiguous-bet"]
    amb = c["ambiguous-bet"] + c["abstain-ambiguous"]
    trick = params.world.trickster
    # which witnesses the trickster touches at the witness layer
    touched = set()
    if trick is not None:
        if L == 0:
            touched = {trick}
        else:
            # a band gate j reads j and j+1: trace parents back L layers
            frontier = {trick}
            for _ in range(L):
                frontier = {j for j in range(params.world.n) if (j in frontier) or ((j + 1) % params.world.n in frontier)}
            touched = frontier
    burned = [j for j, b in enumerate(obs.burned) if b]
    true_burns = [j for j in burned if j in touched]
    false_burns = [j for j in burned if j not in touched]
    delays = []
    for s in switches:
        later = [a for a in obs.alarms if a >= s]
        if later and later[0] - s <= 200:
            delays.append(later[0] - s)
    ds = sorted(delays)
    nr = obs.rec_right / (obs.rec_right + obs.rec_wrong) if (obs.rec_right + obs.rec_wrong) else None
    tr = obs.dec_right / (obs.dec_right + obs.dec_wrong) if (obs.dec_right + obs.dec_wrong) else None
    return dict(
        seed=seed, steps=params.steps, circuit=circ.describe(), witness_layer=L, witnesses=obs.m,
        score=obs.score, score_per_1000=round(1000.0 * obs.score / params.steps, 2),
        bets=bets, hits=c["hits"], misses=c["misses"], hit_rate=(c["hits"] / bets) if bets else None,
        void=c["void"], counts=dict(c),
        ambiguous_frac=round(amb / params.steps, 4),
        ambiguous_bet_rate=(c["ambiguous-bet"] / amb) if amb else None,
        regime_accuracy=round(obs.regime_agree / obs.regime_checked, 4) if obs.regime_checked else None,
        alarms=len(obs.alarms), switches=len(switches), detected=len(delays),
        median_delay=ds[len(ds) // 2] if ds else None,
        burned=burned, n_burned=len(burned), true_burns=true_burns, false_burns=false_burns,
        first_true_burn=min((obs.burn_time[j] for j in true_burns), default=None),
        first_false_burn=min((obs.burn_time[j] for j in false_burns), default=None),
        forgiving_weights=[round(w, 3) for w in obs.weights()] if params.observer.trust_mode == "forgiving" else None,
        lying_frac=round(lying_steps / params.steps, 4),
        AR_final=round(obs.AR, 4), AR_trace=obs.AR_trace,
        narrative_rate=None if nr is None else round(nr, 4), true_rate=None if tr is None else round(tr, 4),
        ego_gap=None if (nr is None or tr is None) else round(nr - tr, 4),
        params=dict(steps=params.steps, circuit_ops=list(params.circuit_ops), world=asdict(params.world),
                    observer=asdict(params.observer)),
    )
