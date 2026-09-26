"""The model, in four layers.

  WORLD      a hidden phase on U_n (the n-th roots of unity; n = 4 gives the Gaussian units
             {1, i, -1, -i}, n = 6 the Eisenstein units) turning one notch per step in the
             direction set by a hidden regime r in {+1, -1}, with noise and regime switches.

  NETWORK    nodes. Each node holds a quadratic conjugate pair (alpha, conj alpha) and an axis
             on U_n. The world's phase, seen from the node's axis, selects the operation the
             node performs on its pair:  '*' (combine: alpha * conj alpha = norm) when the
             phase is in the node's front half-turn, '+' (cancel: alpha + conj alpha = trace)
             otherwise. For the pair (+i, -i) this is literally "combined into 1 or cancelled
             to 0", so the network's output is a binary sequence.

  SENSOR     the observer's decoder: from the emitted symbols and the public spec of each node,
             recover the set of phases consistent with the symbols, then the direction evidence
             of the last step (+1, -1, or 0 when undetermined).

  OBSERVER   three coupled blocks (IMM Paper 18, T-OR-1), parameterised by the RPCS-1 primitives:
             ESTIMATE (SG signal gain, UE update elasticity)  E <- (1-UE) E + UE * SG * e
             DETECT   (FT filter threshold, TI temporal integration)  a CUSUM on contradiction of
                      the current model; an alarm resets E, opens a seam, boosts UE for the seam
             COMMIT   (AR ambiguity resolution)  predicts the next readout bit; confident when
                      |E| >= margin, otherwise resolves the ambiguity with the prior AR, which
                      starts at 0 (blank) and accumulates a narrative from an attribution-filtered
                      ledger (kappa = 0 honest, kappa = 1 pure self-serving ego).

All node arithmetic is exact integer arithmetic. All randomness comes from one seeded PRNG.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

PLUS = "+"
TIMES = "*"

# ----------------------------------------------------------------------------- nodes


@dataclass(frozen=True)
class QuadraticPair:
    """The conjugate pair (alpha, conj alpha) of an algebraic integer alpha.

    ring = 'gauss': alpha = a + b*i,      trace 2a,      norm a^2 + b^2
    ring = 'eisen': alpha = a + b*w,      trace 2a - b,  norm a^2 - ab + b^2,  w = exp(2 pi i / 3)

    The two operations a node can perform on its pair:
        '+'  cancel :  alpha + conj alpha = trace
        '*'  combine:  alpha * conj alpha = norm
    A pair is BLIND when the two operations give the same integer (trace == norm), i.e. when
    |alpha - 1|^2 = norm - trace + 1 = 1: alpha lies on the circle |z - 1| = 1, the reciprocal
    image of the critical line (Room VII's circle). Its lattice points are 1 + (roots of unity).
    """

    a: int
    b: int
    ring: str = "gauss"

    def __post_init__(self) -> None:
        if self.ring not in ("gauss", "eisen"):
            raise ValueError(f"unknown ring {self.ring!r}")
        if not (isinstance(self.a, int) and isinstance(self.b, int)):
            raise TypeError("coordinates must be integers")

    @property
    def trace(self) -> int:
        return 2 * self.a if self.ring == "gauss" else 2 * self.a - self.b

    @property
    def norm(self) -> int:
        if self.ring == "gauss":
            return self.a * self.a + self.b * self.b
        return self.a * self.a - self.a * self.b + self.b * self.b

    @property
    def gap(self) -> int:
        """|trace - norm| = | |alpha-1|^2 - 1 |: how far apart the two operations' outputs are."""
        return abs(self.trace - self.norm)

    @property
    def blind(self) -> bool:
        return self.gap == 0

    @property
    def dist2_from_one(self) -> int:
        """|alpha - 1|^2, exactly, as an integer."""
        return self.norm - self.trace + 1

    def emit(self, op: str) -> int:
        if op == PLUS:
            return self.trace
        if op == TIMES:
            return self.norm
        raise ValueError(f"unknown op {op!r}")

    def decode(self, symbol: int) -> Optional[str]:
        """Which operation produced `symbol`? None when the pair is blind."""
        if self.blind:
            return None
        if symbol == self.norm:
            return TIMES
        if symbol == self.trace:
            return PLUS
        raise ValueError(f"symbol {symbol} is neither trace nor norm of {self.label()}")

    def label(self) -> str:
        unit = "i" if self.ring == "gauss" else "w"
        if self.b == 0:
            return f"{self.a}"
        sign = "+" if self.b > 0 else "-"
        mag = "" if abs(self.b) == 1 else str(abs(self.b))
        return f"{self.a}{sign}{mag}{unit}" if self.a != 0 else f"{'-' if self.b < 0 else ''}{mag}{unit}"


PAIR_I = QuadraticPair(0, 1, "gauss")  # (+i, -i): trace 0, norm 1 -> the binary node
PAIR_W = QuadraticPair(0, 1, "eisen")  # (w, conj w): trace -1, norm 1


def gaussian_table(radius: int = 2) -> List[Dict[str, Any]]:
    rows = []
    for a in range(-radius, radius + 1):
        for b in range(0, radius + 1):  # conjugates give the same pair; keep b >= 0
            p = QuadraticPair(a, b, "gauss")
            rows.append(dict(alpha=p.label(), a=a, b=b, trace=p.trace, norm=p.norm,
                             gap=p.gap, dist2=p.dist2_from_one, blind=p.blind))
    return rows


def eisenstein_table(radius: int = 2) -> List[Dict[str, Any]]:
    rows = []
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            p = QuadraticPair(a, b, "eisen")
            rows.append(dict(alpha=p.label(), a=a, b=b, trace=p.trace, norm=p.norm,
                             gap=p.gap, dist2=p.dist2_from_one, blind=p.blind))
    return rows


@dataclass(frozen=True)
class Node:
    pair: QuadraticPair
    axis: int  # a phase index on U_n: the node's receptive axis


class Network:
    """Nodes reading one hidden phase. Front half-turn -> '*' (combine), back half-turn -> '+' (cancel)."""

    def __init__(self, nodes: Sequence[Node], n: int):
        if n % 2:
            raise ValueError("n must be even so that a half-turn is a whole number of notches")
        self.nodes = list(nodes)
        self.n = n

    def op(self, phase: int, j: int) -> str:
        return TIMES if (phase - self.nodes[j].axis) % self.n < self.n // 2 else PLUS

    def ops(self, phase: int) -> List[str]:
        return [self.op(phase, j) for j in range(len(self.nodes))]

    def emit(self, phase: int) -> List[int]:
        return [node.pair.emit(self.op(phase, j)) for j, node in enumerate(self.nodes)]

    def readout_bit(self, phase: int, j: int = 0) -> int:
        """The binary readout of node j: 1 = combined (norm), 0 = cancelled (trace)."""
        return 1 if self.op(phase, j) == TIMES else 0

    def bits(self, phase: int) -> List[int]:
        return [self.readout_bit(phase, j) for j in range(len(self.nodes))]

    def candidates(self, symbols: Sequence[int]) -> List[int]:
        """Phases consistent with the emitted symbols, using only non-blind nodes."""
        out = []
        for h in range(self.n):
            ok = True
            for j, node in enumerate(self.nodes):
                op = node.pair.decode(symbols[j])
                if op is None:
                    continue
                if op != self.op(h, j):
                    ok = False
                    break
            if ok:
                out.append(h)
        return out


def default_network(n: int = 4, n_nodes: int = 2, pair: QuadraticPair = PAIR_I) -> Network:
    """n_nodes copies of `pair` on evenly spread axes 0, 1, 2, ... (quadrature for n = 4, 2 nodes)."""
    return Network([Node(pair, j % n) for j in range(n_nodes)], n)


# ----------------------------------------------------------------------------- world


@dataclass
class WorldParams:
    n: int = 4          # phase alphabet U_n
    eps: float = 0.15   # probability that a step is uniform noise instead of the regime's notch
    p_switch: float = 0.005  # per-step probability that the regime flips
    regime0: int = +1


class World:
    def __init__(self, params: WorldParams, rng: random.Random):
        self.p = params
        self.rng = rng
        self.phase = rng.randrange(params.n)
        self.regime = params.regime0
        self.t = 0
        self.last_step: Optional[int] = None
        self.switched_last = False

    def step(self) -> int:
        p = self.p
        self.switched_last = False
        if self.rng.random() < p.p_switch:
            self.regime = -self.regime
            self.switched_last = True
        if self.rng.random() < p.eps:
            k = self.rng.randrange(p.n)
        else:
            k = 1 if self.regime > 0 else p.n - 1
        self.phase = (self.phase + k) % p.n
        self.last_step = k
        self.t += 1
        return k

    def direction_of(self, k: int) -> int:
        if k == 1:
            return +1
        if k == self.p.n - 1:
            return -1
        return 0


# ----------------------------------------------------------------------------- observer


@dataclass
class ObserverParams:
    TI: float = 8.0        # temporal integration: CUSUM drift = 1/TI (contradiction must persist ~TI steps)
    SG: float = 1.0        # signal gain on the direction evidence entering the estimate
    FT: float = 4.0        # filter threshold: CUSUM alarm level
    UE: float = 0.10       # update elasticity: learning rate of the estimate
    margin: float = 0.30   # |E| below this is 'ambiguous' and is resolved by the prior AR
    eta: float = 0.02      # narrative accumulation rate (AR is a leaky integrator of recorded outcomes)
    tau: float = 0.15      # softness of the prior -> bet map: P(bet | ambiguous) = sigmoid(AR / tau)
    kappa: float = 0.0     # self-serving attribution bias: 0 honest ... 1 records only decisions that went right
    seam_len: int = 4      # steps of abstention after an alarm (a seam event)
    reset_boost: float = 3.0  # UE multiplier while the seam is open
    detect_on: bool = True
    ego_blinds_detect: bool = False  # optional coupling COMMIT -> DETECT: FT_eff = FT * (1 + |AR|)
    AR0: float = 0.0       # blank prior


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


class Observer:
    """Three coupled blocks. Call order per world step:  sense -> detect -> estimate -> commit -> (world steps) -> settle."""

    def __init__(self, params: ObserverParams, net: Network, rng: random.Random):
        self.p = params
        self.net = net
        self.rng = rng
        self.n = net.n
        # block states
        self.E = 0.0                # ESTIMATE
        self.S = 0.0                # DETECT (CUSUM statistic)
        self.seam = 0               # DETECT -> COMMIT (steps of forced abstention remaining)
        self.AR = params.AR0        # COMMIT prior (the narrative)
        self.prev_candidates: Optional[List[int]] = None
        self.t = 0
        # ledgers
        self.alarms: List[int] = []
        self.score = 0
        self.counts: Dict[str, int] = {k: 0 for k in (
            "confident", "structural", "ambiguous-bet", "abstain-ambiguous",
            "abstain-seam", "abstain-undetermined", "hits", "misses")}
        self.rec_right = 0   # decisions the block RECORDED as right (attribution-filtered)
        self.rec_wrong = 0
        self.dec_right = 0   # decisions that WERE right (bet and hit, or abstained and would have missed)
        self.dec_wrong = 0
        self.evidence_zero = 0

    # -- sensor
    def sense(self, symbols: Sequence[int]) -> Tuple[List[int], int]:
        cands = self.net.candidates(symbols)
        e = 0
        if self.prev_candidates is not None and len(self.prev_candidates) == 1 and len(cands) == 1:
            k = (cands[0] - self.prev_candidates[0]) % self.n
            e = +1 if k == 1 else (-1 if k == self.n - 1 else 0)
        self.prev_candidates = cands
        if e == 0:
            self.evidence_zero += 1
        return cands, e

    # -- DETECT
    def detect(self, e: int) -> bool:
        p = self.p
        if not p.detect_on:
            return False
        d = _sign(self.E) if abs(self.E) > 1e-12 else 0
        contradiction = -e * d  # +1 when the evidence contradicts the current model
        self.S = max(0.0, self.S + contradiction - 1.0 / p.TI)
        ft = p.FT * (1.0 + abs(self.AR)) if p.ego_blinds_detect else p.FT
        if self.S >= ft:
            self.S = 0.0
            self.E = 0.0            # "evidence answers an expired question" (R-5)
            self.seam = p.seam_len  # seam opens; UE boosted while it is open (R-2)
            self.alarms.append(self.t)
            return True
        return False

    # -- ESTIMATE
    def estimate(self, e: int) -> None:
        p = self.p
        ue = min(1.0, p.UE * p.reset_boost) if self.seam > 0 else p.UE
        self.E = (1.0 - ue) * self.E + ue * p.SG * e

    # -- COMMIT
    def commit(self, cands: List[int]) -> Dict[str, Any]:
        """Predict the readout bit of node 0 at the next step. Returns the decision record."""
        p = self.p
        rec: Dict[str, Any] = dict(kind=None, pred=None, direction=0)
        if not cands:
            rec["kind"] = "abstain-undetermined"
            return rec
        bits_by_dir = {}
        for d in (+1, -1):
            bits_by_dir[d] = {self.net.readout_bit((h + d) % self.n, 0) for h in cands}
        if len(bits_by_dir[+1]) == 1 and bits_by_dir[+1] == bits_by_dir[-1]:
            rec.update(kind="structural", pred=next(iter(bits_by_dir[+1])))
            return rec
        if self.seam > 0:
            rec["kind"] = "abstain-seam"
            return rec
        d = _sign(self.E) if abs(self.E) > 1e-12 else self.rng.choice((+1, -1))
        bits = bits_by_dir[d]
        if len(bits) != 1:
            rec["kind"] = "abstain-undetermined"
            return rec
        pred = next(iter(bits))
        rec.update(pred=pred, direction=d)
        if abs(self.E) >= p.margin:
            rec["kind"] = "confident"
        else:
            if self.rng.random() < _sigmoid(self.AR / p.tau):
                rec["kind"] = "ambiguous-bet"
            else:
                rec["kind"] = "abstain-ambiguous"
        return rec

    # -- settlement and narrative
    def settle(self, rec: Dict[str, Any], actual_bit: int) -> None:
        p = self.p
        kind = rec["kind"]
        self.counts[kind] += 1
        if self.seam > 0:
            self.seam -= 1
        self.t += 1
        if kind in ("confident", "structural", "ambiguous-bet"):
            hit = rec["pred"] == actual_bit
            self.counts["hits" if hit else "misses"] += 1
            self.score += 1 if hit else -1
        if kind not in ("ambiguous-bet", "abstain-ambiguous"):
            return
        # --- the narrative: only decisions the prior resolved are entered
        bet = kind == "ambiguous-bet"
        would_hit = rec["pred"] == actual_bit
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
            y = 1 if would_hit else -1   # what the recorded event says about the edge of betting
            # leaky mean of recorded outcomes: with kappa = 0 the fixed point is the TRUE edge of
            # betting under ambiguity, P(would hit) - P(would miss); with kappa = 1 it is +1 or -1
            self.AR = (1.0 - p.eta) * self.AR + p.eta * y

    # -- summaries
    def narrative_rate(self) -> Optional[float]:
        tot = self.rec_right + self.rec_wrong
        return self.rec_right / tot if tot else None

    def true_rate(self) -> Optional[float]:
        tot = self.dec_right + self.dec_wrong
        return self.dec_right / tot if tot else None


# ----------------------------------------------------------------------------- run


@dataclass
class RunParams:
    steps: int = 20000
    n_nodes: int = 2
    pair: QuadraticPair = PAIR_I
    world: WorldParams = field(default_factory=WorldParams)
    observer: ObserverParams = field(default_factory=ObserverParams)
    ar_trace_every: int = 200
    bits_sample: int = 64
    delay_cap: int = 200
    false_alarm_window: int = 50


def run(params: RunParams, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    net = default_network(params.world.n, params.n_nodes, params.pair)
    world = World(params.world, rng)
    obs = Observer(params.observer, net, rng)

    switches: List[int] = []
    ar_trace: List[float] = []
    bits0: List[int] = []
    # the observer first sees the initial phase (no evidence yet)
    symbols = net.emit(world.phase)
    cands, e = obs.sense(symbols)
    for t in range(params.steps):
        rec = obs.commit(cands)
        world.step()
        if world.switched_last:
            switches.append(t + 1)
        symbols = net.emit(world.phase)
        actual = net.readout_bit(world.phase, 0)
        if len(bits0) < params.bits_sample:
            bits0.append(actual)
        obs.settle(rec, actual)
        cands, e = obs.sense(symbols)
        obs.detect(e)
        obs.estimate(e)
        if (t + 1) % params.ar_trace_every == 0:
            ar_trace.append(round(obs.AR, 4))

    # detection quality
    delays: List[int] = []
    missed = 0
    for s in switches:
        later = [a for a in obs.alarms if a >= s]
        if later and later[0] - s <= params.delay_cap:
            delays.append(later[0] - s)
        else:
            missed += 1
    false_alarms = 0
    for a in obs.alarms:
        if not any(a - params.false_alarm_window <= s <= a for s in switches):
            false_alarms += 1
    delays_sorted = sorted(delays)
    median_delay = delays_sorted[len(delays_sorted) // 2] if delays_sorted else None

    c = obs.counts
    bets = c["confident"] + c["structural"] + c["ambiguous-bet"]
    amb_total = c["ambiguous-bet"] + c["abstain-ambiguous"]
    nr, tr = obs.narrative_rate(), obs.true_rate()
    return dict(
        seed=seed,
        steps=params.steps,
        score=obs.score,
        score_per_1000=round(1000.0 * obs.score / params.steps, 2),
        bets=bets,
        hits=c["hits"],
        misses=c["misses"],
        hit_rate=(c["hits"] / bets) if bets else None,
        counts=dict(c),
        ambiguous_total=amb_total,
        ambiguous_bet_rate=(c["ambiguous-bet"] / amb_total) if amb_total else None,
        alarms=len(obs.alarms),
        switches=len(switches),
        detected=len(delays),
        missed=missed,
        median_delay=median_delay,
        false_alarms=false_alarms,
        AR_final=round(obs.AR, 4),
        AR_trace=ar_trace,
        narrative_rate=None if nr is None else round(nr, 4),
        true_rate=None if tr is None else round(tr, 4),
        ego_gap=None if (nr is None or tr is None) else round(nr - tr, 4),
        recorded=obs.rec_right + obs.rec_wrong,
        resolved=obs.dec_right + obs.dec_wrong,
        evidence_zero_frac=round(obs.evidence_zero / (params.steps + 1), 4),
        bits0=bits0,
        node_table=[dict(alpha=nd.pair.label(), axis=nd.axis, trace=nd.pair.trace,
                         norm=nd.pair.norm, blind=nd.pair.blind) for nd in net.nodes],
        params=dict(steps=params.steps, n_nodes=params.n_nodes, pair=params.pair.label(),
                    ring=params.pair.ring, world=asdict(params.world), observer=asdict(params.observer)),
    )
