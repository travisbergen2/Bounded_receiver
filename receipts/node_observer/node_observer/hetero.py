"""v0.6 — three coupled systems, each with its own architecture, one shared currency.

Travis (2026-09-26): "would it work more efficiently if each system had a unique yet compatible
architecture with the other systems, working together with different purposes?"

The three jobs have three different optimal algorithms, and all three are functionals of ONE stream:
the log-likelihood-ratio increment  ℓ_t = log p(x_t | regime +) − log p(x_t | regime −).

    ESTIMATE   the posterior: accumulate ℓ_t with prior mixing for regime switches (exact two-state HMM)
    DETECT     the posterior changing sides (MAP flip) — no threshold; the accumulated form of "contradiction"
    COMMIT     Wald's band: act when |log-odds| ≥ A = log((1−α)/α); inside the band the prior decides
               (the v0.2 narrative machinery, honest κ = 0). No seam block: the band IS the seam.

For the channel world (n channels, flip noise ε) the increment is ℓ_t = (n₊ − n₋) · log((1−ε)/ε):
v0.4's direction channel, scaled by the noise's log-odds weight. Presence n₊ + n₋ is the count.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from .layered import ChannelWorld, ChannelWorldParams, _sign, _sigmoid


@dataclass
class HeteroParams:
    alpha: float = 0.05      # SPRT error target → band A = log((1-α)/α)
    AR0: float = 0.0
    eta: float = 0.02
    tau: float = 0.15
    kappa: float = 0.0


class HeteroObserver:
    def __init__(self, params: HeteroParams, world: ChannelWorldParams, rng: random.Random):
        self.p = params
        self.eps = world.eps
        self.ps = world.p_switch
        self.wgt = math.log((1.0 - world.eps) / world.eps)
        self.A = math.log((1.0 - params.alpha) / params.alpha)
        self.rng = rng
        self.L = 0.0                      # log-odds regime + vs −
        self.prev_map = 0
        self.AR = params.AR0
        self.t = 0
        self.alarms: List[int] = []
        self.score = 0
        self.counts: Dict[str, int] = {k: 0 for k in ("confident", "ambiguous-bet", "abstain-ambiguous", "hits", "misses", "void")}
        self.rec_right = self.rec_wrong = self.dec_right = self.dec_wrong = 0
        self.regime_agree = self.regime_checked = 0

    # ESTIMATE: exact two-state HMM in log-odds
    def estimate(self, signs: List[int]) -> None:
        o = math.exp(max(-60.0, min(60.0, self.L)))
        o2 = (o * (1.0 - self.ps) + self.ps) / (o * self.ps + (1.0 - self.ps))   # prior mixing
        self.L = math.log(o2) + self.wgt * sum(signs)                              # + evidence

    # DETECT: the posterior changing sides
    def detect(self) -> bool:
        cur = _sign(self.L)
        flip = self.prev_map != 0 and cur != 0 and cur != self.prev_map
        if cur != 0:
            self.prev_map = cur
        if flip:
            self.alarms.append(self.t)
        return flip

    # COMMIT: Wald's band, prior inside it
    def commit(self) -> Dict[str, Any]:
        p = self.p
        d = _sign(self.L) if abs(self.L) > 1e-12 else self.rng.choice((+1, -1))
        rec: Dict[str, Any] = dict(pred=d, kind=None)
        if abs(self.L) >= self.A:
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
        self.t += 1
        if abs(self.L) > 1e-12:
            self.regime_checked += 1
            self.regime_agree += (_sign(self.L) == regime)
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
        right = would_hit if bet else (not would_hit)
        if right:
            self.dec_right += 1
        else:
            self.dec_wrong += 1
        if self.rng.random() < ((1.0 + p.kappa) / 2.0 if right else (1.0 - p.kappa) / 2.0):
            if right:
                self.rec_right += 1
            else:
                self.rec_wrong += 1
            self.AR = (1.0 - p.eta) * self.AR + p.eta * (1 if would_hit else -1)


def run_hetero(world_p: ChannelWorldParams, obs_p: HeteroParams, steps: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    world = ChannelWorld(world_p, rng)
    obs = HeteroObserver(obs_p, world_p, rng)
    obs.estimate(world.signs)
    obs.detect()
    switches: List[int] = []
    for t in range(steps):
        rec = obs.commit()
        world.step()
        if world.switched_last:
            switches.append(t + 1)
        realized = _sign(sum(world.signs))
        obs.settle(rec, realized, world.regime)
        obs.estimate(world.signs)
        obs.detect()
    c = obs.counts
    bets = c["confident"] + c["ambiguous-bet"]
    amb = c["ambiguous-bet"] + c["abstain-ambiguous"]
    # attribution of alarms (as in superpose.py)
    spurious, attributed, delays = 0, 0, []
    prev = 0
    for a in obs.alarms:
        recent = [s for s in switches if prev < s <= a]
        if recent:
            attributed += 1
            delays.append(a - recent[-1])
        else:
            spurious += 1
        prev = a
    detected = sum(1 for s in switches if any(s <= a <= s + 3 for a in obs.alarms))
    nr = obs.rec_right / (obs.rec_right + obs.rec_wrong) if (obs.rec_right + obs.rec_wrong) else None
    tr = obs.dec_right / (obs.dec_right + obs.dec_wrong) if (obs.dec_right + obs.dec_wrong) else None
    return dict(
        seed=seed, steps=steps, score=obs.score, score_per_1000=round(1000.0 * obs.score / steps, 2),
        bets=bets, hit_rate=(c["hits"] / bets) if bets else None, void=c["void"], counts=dict(c),
        ambiguous_frac=round(amb / steps, 4), ambiguous_bet_rate=(c["ambiguous-bet"] / amb) if amb else None,
        regime_accuracy=round(obs.regime_agree / obs.regime_checked, 4) if obs.regime_checked else None,
        switches=len(switches), alarms=len(obs.alarms), detected=detected, attributed=attributed, spurious=spurious,
        median_delay=(sorted(delays)[len(delays) // 2] if delays else None),
        max_delay=(max(delays) if delays else None),
        AR_final=round(obs.AR, 4), narrative_rate=None if nr is None else round(nr, 4),
        true_rate=None if tr is None else round(tr, 4), band_A=round(obs.A, 4), llr_weight=round(obs.wgt, 4),
        params=dict(world=asdict(world_p), observer=asdict(obs_p)),
    )
