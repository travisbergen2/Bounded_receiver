"""v0.4 sketch — holding a contradiction instead of cancelling it.

Travis (2026-09-26): "a way to get (1, −1, 0, superposition) as outputs so it can hold contradictions
like I do."

The one change: keep TWO channels per value instead of one. A value is a signed multiset of votes,
(n_plus, n_minus). From it:

        direction  d = n_plus − n_minus        (what the old node kept: the sign readout)
        presence   p = n_plus + n_minus        (what the old node threw away)

and the four outputs are the sign patterns of (d, p):

        0            (0, 0)   nothing arrived
        +1           (+, +)   plus, uncontested        (with counts: majority plus, and how contested)
        −1           (−, +)   minus, uncontested
        S            (0, +)   SUPERPOSITION: something is there and it points nowhere — a held contradiction

Operations, on Travis's sign table (real units; 2026-09-25 21:55):

        HOLD     x ⊔ y = (x_plus + y_plus, x_minus + y_minus)          his '+', but (+) ⊔ (−) = S, not 0
        COMPARE  x ⊗ y = (x_plus·y_plus + x_minus·y_minus,  x_plus·y_minus + x_minus·y_plus)
                 his '×' row exactly: (+)⊗(+) = +, (−)⊗(−) = +, (+)⊗(−) = −

This is the group ring Z[Z/2] — the split-complex integers, j² = +1 — and in (presence, direction)
coordinates HOLD adds and COMPARE multiplies COMPONENTWISE: p(x⊗y) = p(x)·p(y), d(x⊗y) = d(x)·d(y).
The two channels never mix. The old v0.3 node is the projection onto d (COLLAPSE); the superposition
values are exactly its kernel — the part of the state the collapse cannot see.

Names: the four values are Belnap–Dunn's FOUR (None / True / False / Both, 1976–77); HOLD is Belnap's
knowledge-order join (graded by counts); COMPARE is the set product, not one of Belnap's truth
connectives. Known logic; the identification with Travis's sign table and with the (trace, norm)
readout of a conjugate pair is a dictionary, elementary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .layered import _sign


@dataclass(frozen=True)
class Held:
    n_plus: int
    n_minus: int

    def __post_init__(self) -> None:
        if self.n_plus < 0 or self.n_minus < 0:
            raise ValueError("vote counts are non-negative")

    # channels
    @property
    def direction(self) -> int:
        return self.n_plus - self.n_minus

    @property
    def presence(self) -> int:
        return self.n_plus + self.n_minus

    # readouts
    def readout(self) -> str:
        """Four-valued: '0' nothing, '+' / '-' uncontested, 'S' contested (both votes present)."""
        if self.presence == 0:
            return "0"
        if self.n_minus == 0:
            return "+"
        if self.n_plus == 0:
            return "-"
        return "S"

    def collapse(self) -> int:
        """The old node: sign of the direction, 0 on a tie. Erases presence."""
        return _sign(self.direction)

    def collapse_readout(self) -> str:
        return {1: "+", -1: "-", 0: "0"}[self.collapse()]

    # operations
    def hold(self, other: "Held") -> "Held":
        return Held(self.n_plus + other.n_plus, self.n_minus + other.n_minus)

    def compare(self, other: "Held") -> "Held":
        return Held(self.n_plus * other.n_plus + self.n_minus * other.n_minus,
                    self.n_plus * other.n_minus + self.n_minus * other.n_plus)

    def __str__(self) -> str:
        return f"({self.n_plus}+,{self.n_minus}-)"


NOTHING = Held(0, 0)
PLUS = Held(1, 0)
MINUS = Held(0, 1)
BOTH = Held(1, 1)
BASIC: Dict[str, Held] = {"0": NOTHING, "+": PLUS, "-": MINUS, "S": BOTH}


def from_sign(s: int) -> Held:
    return PLUS if s > 0 else MINUS if s < 0 else NOTHING


def tables() -> Dict[str, Dict[Tuple[str, str], str]]:
    """HOLD and COMPARE on the four basic values, by readout."""
    out: Dict[str, Dict[Tuple[str, str], str]] = {"HOLD": {}, "COMPARE": {}}
    for a, x in BASIC.items():
        for b, y in BASIC.items():
            out["HOLD"][(a, b)] = x.hold(y).readout()
            out["COMPARE"][(a, b)] = x.compare(y).readout()
    return out


def travis_configurations() -> List[Dict[str, str]]:
    """His three pairs under HOLD (his '+') and COMPARE (his '×'), old collapse beside the new readout."""
    rows = []
    for name, (a, b) in (("(+,-)", (PLUS, MINUS)), ("(+,+)", (PLUS, PLUS)), ("(-,-)", (MINUS, MINUS))):
        h, c = a.hold(b), a.compare(b)
        rows.append(dict(pair=name, hold=h.readout(), hold_channels=f"d={h.direction:+d} p={h.presence}",
                         hold_old_collapse=h.collapse_readout(),
                         compare=c.readout(), compare_channels=f"d={c.direction:+d} p={c.presence}"))
    return rows
