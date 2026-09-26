"""v0.5 — an emergent superposition that scales until it collapses (the two classical forms).

Travis (2026-09-26): "an emergent superposition property that can scale until it collapses to the answer."

HARD (possibilistic).  The observer holds the SET of joint hypotheses (phase h, regime r) still consistent
with everything it has seen. It starts full (2n hypotheses — the superposition), each observation
intersects it, the dynamics advance it (h -> h + r). Collapse = the set becomes a singleton. The EMPTY
set = every hypothesis has been refuted = the world changed: a change detector with no threshold. Then
the observer resets to the full set restricted to the current observation. Exact, integer, no parameters.

SOFT (probabilistic).  The same hypotheses carry weights: predict with the noise and switch model,
multiply by the observation's consistency, normalise — the exact hidden-Markov posterior (the E-1B
filter, here in floating point). Collapse = one hypothesis holds >= `collapse_mass`; detection = the
observation's predicted probability falls below `surprise_p` (the price of surviving noise is a threshold).

FACTORED.  Two independent worlds tracked by two filters hold the PRODUCT of their candidate sets at the
SUM of their costs — the sense in which a classical superposition "scales".

Neither form interferes: sets union, weights add. The quantum form (Grover's amplitude amplification)
collapses to the answer because wrong amplitudes cancel — the (+i)+(−i)=0 that v0.4 deliberately kept.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .model import Network, World, WorldParams, default_network, PAIR_I

Hyp = Tuple[int, int]   # (phase, regime)


class HardFilter:
    def __init__(self, n: int, net: Network):
        self.n = n
        self.net = net
        self.full: Set[Hyp] = {(h, r) for h in range(n) for r in (+1, -1)}
        self.cands: Set[Hyp] = set(self.full)
        self.resets = 0
        self.t = 0

    def consistent_phases(self, symbols: Sequence[int]) -> Set[int]:
        return set(self.net.candidates(symbols))

    def observe(self, symbols: Sequence[int]) -> bool:
        """Intersect with the observation. Returns True if the set emptied (contradiction -> reset)."""
        phases = self.consistent_phases(symbols)
        new = {(h, r) for (h, r) in self.cands if h in phases}
        if not new:
            self.resets += 1
            self.cands = {(h, r) for (h, r) in self.full if h in phases}
            return True
        self.cands = new
        return False

    def predict(self) -> None:
        self.cands = {((h + r) % self.n, r) for (h, r) in self.cands}
        self.t += 1

    @property
    def size(self) -> int:
        return len(self.cands)

    @property
    def collapsed(self) -> bool:
        return len(self.cands) == 1

    def answer(self) -> Optional[Hyp]:
        return next(iter(self.cands)) if self.collapsed else None


class SoftFilter:
    def __init__(self, n: int, net: Network, eps: float, p_switch: float,
                 collapse_mass: float = 0.95, surprise_p: float = 0.02):
        self.n = n
        self.net = net
        self.eps = eps
        self.p_switch = p_switch
        self.collapse_mass = collapse_mass
        self.surprise_p = surprise_p
        self.w: Dict[Hyp, float] = {(h, r): 1.0 / (2 * n) for h in range(n) for r in (+1, -1)}
        self.last_p_obs = 1.0
        self.min_p_obs = 1.0
        self.surprises = 0          # steps with P(obs | past) < surprise_p  (single-step surprise; see README)
        self.alarms = 0             # MAP-regime flips: the posterior changing sides (the emergent detector)
        self.prev_map: Optional[int] = None
        self.t = 0

    def regime_mass(self) -> Dict[int, float]:
        m = {+1: 0.0, -1: 0.0}
        for (h, r), w in self.w.items():
            m[r] += w
        return m

    def map_regime(self) -> int:
        m = self.regime_mass()
        return +1 if m[+1] >= m[-1] else -1

    def predict(self) -> None:
        n, eps, ps = self.n, self.eps, self.p_switch
        new: Dict[Hyp, float] = {(h, r): 0.0 for h in range(n) for r in (+1, -1)}
        for (h, r), w in self.w.items():
            if w == 0.0:
                continue
            for r2, pr in ((r, 1.0 - ps), (-r, ps)):
                # step by the regime's notch with prob 1-eps, else uniform over all n notches
                new[((h + r2) % n, r2)] += w * pr * (1.0 - eps)
                share = w * pr * eps / n
                for k in range(n):
                    new[((h + k) % n, r2)] += share
        self.w = new
        self.t += 1

    def observe(self, symbols: Sequence[int]) -> bool:
        phases = set(self.net.candidates(symbols))
        total = 0.0
        for hyp in self.w:
            if hyp[0] not in phases:
                self.w[hyp] = 0.0
            total += self.w[hyp]
        self.last_p_obs = total
        if total <= 0.0:
            # impossible under the model (cannot happen with eps > 0); reset to uniform on the observation
            for hyp in self.w:
                self.w[hyp] = 1.0 if hyp[0] in phases else 0.0
            total = sum(self.w.values())
        for hyp in self.w:
            self.w[hyp] /= total
        self.min_p_obs = min(self.min_p_obs, self.last_p_obs)
        if self.last_p_obs < self.surprise_p:
            self.surprises += 1
        cur = self.map_regime()
        flip = self.prev_map is not None and cur != self.prev_map
        self.prev_map = cur
        if flip:
            self.alarms += 1
        return flip

    def top(self) -> Tuple[Hyp, float]:
        hyp = max(self.w, key=self.w.get)
        return hyp, self.w[hyp]

    @property
    def collapsed(self) -> bool:
        return self.top()[1] >= self.collapse_mass

    def entropy_bits(self) -> float:
        return -sum(p * math.log2(p) for p in self.w.values() if p > 0)

    def effective_size(self) -> float:
        return 2.0 ** self.entropy_bits()


@dataclass
class SuperposeParams:
    steps: int = 2000
    n: int = 4
    n_nodes: int = 2
    eps: float = 0.0
    p_switch: float = 0.02
    mode: str = "hard"          # hard | soft
    collapse_mass: float = 0.95
    surprise_p: float = 0.02
    trace_len: int = 40


def run_superpose(p: SuperposeParams, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    net = default_network(p.n, p.n_nodes, PAIR_I)
    world = World(WorldParams(n=p.n, eps=p.eps, p_switch=p.p_switch), rng)
    if p.mode == "hard":
        filt: Any = HardFilter(p.n, net)
    else:
        filt = SoftFilter(p.n, net, p.eps, p.p_switch, p.collapse_mass, p.surprise_p)
    filt.observe(net.emit(world.phase))

    sizes: List[float] = []
    switches: List[int] = []
    alarms: List[int] = []
    collapse_steps: List[int] = []     # steps from each (re)start until first collapse
    since_start = 0
    collapsed_now = filt.collapsed if p.mode == "hard" else filt.collapsed
    if collapsed_now:
        collapse_steps.append(0)
    answers_right = 0
    answers_total = 0
    switches_while_collapsed: List[int] = []
    trace: List[Any] = []
    uncollapsed_len = 0
    for t in range(p.steps):
        was_collapsed = filt.collapsed
        filt.predict()
        world.step()
        if world.switched_last:
            switches.append(t + 1)
            if was_collapsed:
                switches_while_collapsed.append(t + 1)
        alarm = filt.observe(net.emit(world.phase))
        if alarm:
            alarms.append(t + 1)
        size = filt.size if p.mode == "hard" else filt.effective_size()
        sizes.append(size)
        if len(trace) < p.trace_len:
            trace.append(round(size, 2) if p.mode == "soft" else size)
        if filt.collapsed:
            if not collapsed_now:
                collapse_steps.append(uncollapsed_len)   # length of the superposition stretch just ended
                collapsed_now = True
            uncollapsed_len = 0
            hyp = filt.answer() if p.mode == "hard" else filt.top()[0]
            answers_total += 1
            answers_right += (hyp == (world.phase, world.regime))
        else:
            collapsed_now = False
            uncollapsed_len += 1

    # detection quality, two readings:
    #  (i) window: an alarm within 3 steps after a switch counts as detecting it
    #  (ii) attribution: each alarm is attributed to the most recent switch since the previous alarm;
    #       an alarm with no switch since the previous alarm is SPURIOUS (the true false-alarm count)
    detected, delays = 0, []
    for s in switches:
        later = [a for a in alarms if s <= a <= s + 3]
        if later:
            detected += 1
            delays.append(later[0] - s)
    detected_collapsed = sum(1 for s in switches_while_collapsed if any(s <= a <= s + 3 for a in alarms))
    spurious, attributed, attr_delays = 0, 0, []
    prev_alarm = 0
    for a in alarms:
        recent = [s for s in switches if prev_alarm < s <= a]
        if recent:
            attributed += 1
            attr_delays.append(a - recent[-1])
        else:
            spurious += 1
        prev_alarm = a
    false_alarms = spurious
    cs = sorted(collapse_steps)
    return dict(
        seed=seed, mode=p.mode, n=p.n, n_nodes=p.n_nodes, eps=p.eps, steps=p.steps,
        switches=len(switches), switches_while_collapsed=len(switches_while_collapsed),
        detected_while_collapsed=detected_collapsed,
        absorbed=len(switches) - detected,
        alarms=len(alarms), detected=detected, attributed=attributed, false_alarms=false_alarms,
        surprises=(filt.surprises if p.mode == "soft" else None),
        min_p_obs=(round(filt.min_p_obs, 5) if p.mode == "soft" else None),
        median_delay=(sorted(delays)[len(delays) // 2] if delays else None),
        median_attr_delay=(sorted(attr_delays)[len(attr_delays) // 2] if attr_delays else None),
        max_attr_delay=(max(attr_delays) if attr_delays else None),
        collapsed_frac=round(answers_total / p.steps, 4),
        answer_accuracy=(round(answers_right / answers_total, 4) if answers_total else None),
        mean_collapse_time=(round(sum(cs) / len(cs), 3) if cs else None),
        max_collapse_time=(cs[-1] if cs else None),
        mean_size=round(sum(sizes) / len(sizes), 3),
        size_trace=trace,
        params=asdict(p),
    )


def factored_demo(n1: int = 4, n2: int = 16, steps: int = 12, seed: int = 0) -> List[Dict[str, Any]]:
    """Two independent noiseless worlds, two hard filters: the joint superposition is the product of the
    two candidate sets, held at the sum of their sizes."""
    rng = random.Random(seed)
    nets = (default_network(n1, 2, PAIR_I), default_network(n2, 2, PAIR_I))
    worlds = (World(WorldParams(n=n1, eps=0.0, p_switch=0.0), rng), World(WorldParams(n=n2, eps=0.0, p_switch=0.0), rng))
    filts = (HardFilter(n1, nets[0]), HardFilter(n2, nets[1]))
    for f, w, net in zip(filts, worlds, nets):
        f.observe(net.emit(w.phase))
    rows = [dict(step=0, size1=filts[0].size, size2=filts[1].size,
                 joint=filts[0].size * filts[1].size, held=filts[0].size + filts[1].size)]
    for t in range(1, steps + 1):
        for f, w, net in zip(filts, worlds, nets):
            f.predict()
            w.step()
            f.observe(net.emit(w.phase))
        rows.append(dict(step=t, size1=filts[0].size, size2=filts[1].size,
                         joint=filts[0].size * filts[1].size, held=filts[0].size + filts[1].size))
    return rows
