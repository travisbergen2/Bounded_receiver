"""v0.3 — the OPERATION propagates.

Travis (2026-09-25, 21:31): "propagate the operations +, × through layers of (+i,−i), (+i,+i), (−i,−i):
if the first layer is + and it hits (+i,−i) = 0, (+i,+i) = up, (−i,−i) = down, which would × or + to
the next layer, so that complex problems can be figured more efficiently."

Nodes hold FIXED pairs:   C = (+i, −i)   U = (+i, +i)   D = (−i, −i)
An incoming operation is applied to the pair (exact Gaussian-integer arithmetic) and the result is
read by its SIGN CLASS: 0 (cancelled), +1 (positive real or positive imaginary), −1 (negative). With
several live parents a node applies each parent's operation to its own pair, sums the sign classes
(its activation), and the sign of the sum chooses the operation it sends on:

        activation > 0  →  sends ×        activation < 0  →  sends +        activation = 0  →  silent

(the 'sign' routing; the 'pass' variant forwards the parents' common operation on a zero activation).
Everything below is derived from the arithmetic, never hard-coded: `response(t, op)` computes the pair.

Derived response table (sign class of the pair under the op, and the op sent on):
        C: + → 0 (silent)    × → +1 (sends ×)        a filter: kills +, passes ×
        U: + → +1 (sends ×)  × → −1 (sends +)        a swap: NOT
        D: + → −1 (sends +)  × → −1 (sends +)        a reset: everything becomes +
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .layered import G, I, NEG_I, PLUS, TIMES, apply, _sign

SILENT: Optional[str] = None
PAIRS: Dict[str, Tuple[G, G]] = {"C": (I, NEG_I), "U": (I, I), "D": (NEG_I, NEG_I)}      # imaginary units
ONE, NEG_ONE = G(1, 0), G(-1, 0)
PAIRS_REAL: Dict[str, Tuple[G, G]] = {"C": (ONE, NEG_ONE), "U": (ONE, ONE), "D": (NEG_ONE, NEG_ONE)}  # real units: Travis's sign table
UNITS = {"imaginary": PAIRS, "real": PAIRS_REAL}
OPS = (PLUS, TIMES)


def sign_class(v: G) -> int:
    """0 for zero; otherwise the sign of the nonzero coordinate. All values reachable from ±i pairs
    under one + or × lie on an axis, so the class is well defined."""
    if v.is_zero():
        return 0
    if v.a != 0 and v.b != 0:
        raise ValueError(f"{v} is off the axes; no sign class")
    return _sign(v.a) if v.a != 0 else _sign(v.b)


def response(node_type: str, op: str, pairs: Dict[str, Tuple[G, G]] = PAIRS) -> int:
    x, y = pairs[node_type]
    return sign_class(apply(op, x, y))


def route_sign(activation: int) -> Optional[str]:
    if activation > 0:
        return TIMES
    if activation < 0:
        return PLUS
    return SILENT


def response_table(pairs: Dict[str, Tuple[G, G]] = PAIRS) -> List[Dict[str, Any]]:
    rows = []
    for t, (x, y) in pairs.items():
        row: Dict[str, Any] = dict(type=t, pair=f"({x},{y})")
        for op in OPS:
            v = apply(op, x, y)
            row[f"{op} value"] = str(v)
            row[f"{op} class"] = response(t, op, pairs)
            row[f"{op} sends"] = route_sign(response(t, op, pairs)) or "silent"
        rows.append(row)
    return rows


@dataclass(frozen=True)
class OpNode:
    type: str
    parents: Tuple[int, ...]


class OpNet:
    """Layered network. Layer 0 is the input wires (operations or silence). A node may list the same
    parent more than once (an integer weight)."""

    def __init__(self, n_inputs: int, layers: Sequence[Sequence[OpNode]], route: str = "sign",
                 units: str = "imaginary"):
        if route not in ("sign", "pass"):
            raise ValueError(route)
        if units not in UNITS:
            raise ValueError(units)
        self.n = n_inputs
        self.layers = [list(l) for l in layers]
        self.route = route
        self.units = units
        self.pairs = UNITS[units]
        width = n_inputs
        for L, layer in enumerate(self.layers):
            for node in layer:
                if node.type not in PAIRS:
                    raise ValueError(node.type)
                if not node.parents or any(not (0 <= p < width) for p in node.parents):
                    raise ValueError(f"layer {L}: bad parents {node.parents}")
            width = len(layer)

    def widths(self) -> List[int]:
        return [self.n] + [len(l) for l in self.layers]

    @property
    def depth(self) -> int:
        return len(self.layers)

    @property
    def size(self) -> int:
        return sum(len(l) for l in self.layers)

    def node_output(self, node: OpNode, prev: Sequence[Optional[str]]) -> Tuple[Optional[str], int]:
        live = [prev[p] for p in node.parents if prev[p] is not SILENT]
        if not live:
            return SILENT, 0
        a = sum(response(node.type, op, self.pairs) for op in live)
        out = route_sign(a)
        if out is SILENT and self.route == "pass" and len(set(live)) == 1:
            out = live[0]
        return out, a

    def forward(self, inputs: Sequence[Optional[str]]) -> List[List[Optional[str]]]:
        if len(inputs) != self.n:
            raise ValueError("wrong number of inputs")
        vals: List[List[Optional[str]]] = [list(inputs)]
        for layer in self.layers:
            prev = vals[-1]
            vals.append([self.node_output(node, prev)[0] for node in layer])
        return vals

    def output(self, inputs: Sequence[Optional[str]]) -> List[Optional[str]]:
        return self.forward(inputs)[-1]

    # builders
    @staticmethod
    def chain(types: Sequence[str], units: str = "imaginary") -> "OpNet":
        return OpNet(1, [[OpNode(t, (0,))] for t in types], units=units)

    @staticmethod
    def band(n: int, types: Sequence[str], fan_in: int = 2, units: str = "imaginary") -> "OpNet":
        return OpNet(n, [[OpNode(t, tuple((j + k) % n for k in range(fan_in))) for j in range(n)] for t in types],
                     units=units)

    @staticmethod
    def identity(n: int, units: str = "imaginary") -> "OpNet":
        return OpNet(n, [], units=units)


# ----------------------------------------------------------------------------- single-wire chains


def wire_map(node_type: str, pairs: Dict[str, Tuple[G, G]] = PAIRS) -> Dict[Optional[str], Optional[str]]:
    """What one node does to a single wire: op in → op out (silence stays silence)."""
    m: Dict[Optional[str], Optional[str]] = {SILENT: SILENT}
    for op in OPS:
        m[op] = route_sign(response(node_type, op, pairs))
    return m


def compose(f: Dict, g: Dict) -> Dict:
    """f after g."""
    return {x: f[g[x]] for x in g}


def chain_monoid(pairs: Dict[str, Tuple[G, G]] = PAIRS) -> Dict[str, Dict[Optional[str], Optional[str]]]:
    """Closure of {C, U, D} under composition as maps on {+, ×, silent}; keys are shortest words
    realising each map, rightmost letter applied first (so 'CU' means C after U)."""
    gens = {t: wire_map(t, pairs) for t in pairs}
    found: Dict[str, Dict] = {"": {PLUS: PLUS, TIMES: TIMES, SILENT: SILENT}}
    frontier = [""]
    while frontier:
        nxt = []
        for word in frontier:
            for t, g in gens.items():
                new = compose(g, found[word])
                if new not in found.values():
                    found[t + word] = new
                    nxt.append(t + word)
        frontier = nxt
    return found


def describe_map(m: Dict[Optional[str], Optional[str]]) -> str:
    def s(x):
        return "·" if x is SILENT else x
    return f"+→{s(m[PLUS])}, ×→{s(m[TIMES])}"


# ----------------------------------------------------------------------------- Boolean gates
#
# Encoding on live wires: × = TRUE, + = FALSE; silence = no value (must not occur on a gate's output).
# A U node over inputs plus bias wires has activation  #(+ inputs) − #(× inputs)  and sends × when
# positive: it is an INVERTING MAJORITY with a free NOT (U on one wire). D on any live wire is the
# constant +; U(D) is the constant ×. That is the whole gate library.

TRUE, FALSE = TIMES, PLUS


def bool_of(op: Optional[str]) -> Optional[bool]:
    return None if op is SILENT else (op == TRUE)


def verify(net: OpNet, n_vars: int, biases: Sequence[str], fn: Callable[..., bool]) -> Tuple[bool, int, List]:
    """Exhaustively check net(vars + biases)[0] == fn(vars). Returns (ok, patterns, failures)."""
    fails = []
    count = 0
    for bits in product((False, True), repeat=n_vars):
        count += 1
        inputs = [TRUE if b else FALSE for b in bits] + list(biases)
        got = bool_of(net.output(inputs)[0])
        want = fn(*bits)
        if got != want:
            fails.append((bits, got, want))
    return (not fails), count, fails


Gate = Tuple[OpNet, Sequence[str], Callable]


def gate_not() -> Gate:
    return OpNet(1, [[OpNode("U", (0,))]]), (), (lambda x: not x)


def gate_nor() -> Gate:
    # U(x, y, ×): both FALSE gives #+ − #× = 2 − 1 > 0 → × = TRUE; otherwise ≤ −1 → + = FALSE
    return OpNet(3, [[OpNode("U", (0, 1, 2))]]), (TIMES,), (lambda x, y: not (x or y))


def gate_or() -> Gate:
    return OpNet(3, [[OpNode("U", (0, 1, 2))], [OpNode("U", (0,))]]), (TIMES,), (lambda x, y: x or y)


def gate_nand() -> Gate:
    # U(x, y, +): both TRUE gives 1 − 2 < 0 → + = FALSE; otherwise ≥ 1 → × = TRUE
    return OpNet(3, [[OpNode("U", (0, 1, 2))]]), (PLUS,), (lambda x, y: not (x and y))


def gate_and() -> Gate:
    return OpNet(3, [[OpNode("U", (0, 1, 2))], [OpNode("U", (0,))]]), (PLUS,), (lambda x, y: x and y)


def gate_maj3() -> Gate:
    return OpNet(3, [[OpNode("U", (0, 1, 2))], [OpNode("U", (0,))]]), (), (lambda x, y, z: (x + y + z) >= 2)


def gate_threshold(n: int, k: int) -> Gate:
    """At least k of n inputs TRUE. First U over the n inputs and |β| bias wires, β = 2k − 1 − n:
    activation = n − 2s + β with s = #TRUE, negative iff s ≥ k (never zero: n + β is odd); the second
    U re-inverts. Any threshold gate is 2 nodes + bias wires — the unit-weight threshold-logic power."""
    beta = 2 * k - 1 - n
    biases = tuple([PLUS] * beta if beta > 0 else [TIMES] * (-beta))
    m = n + len(biases)
    net = OpNet(m, [[OpNode("U", tuple(range(m)))], [OpNode("U", (0,))]])
    return net, biases, (lambda *bits: sum(bits) >= k)


def gate_xor() -> Gate:
    """XOR(x, y). Why it is not one node: XOR = OR ∧ ¬AND needs OPPOSITE couplings on OR and AND, and a
    U node couples every live input with the same sign (+ → +1, × → −1). C is the parity-free node
    (× → ×, + → silent) that supplies a second coupling, and D supplies constants. Inputs: x, y, bias ×,
    bias +. Depth 5, 13 nodes; every layer keeps a live + wire via D."""
    layers = [
        [OpNode("U", (0, 1, 2)),          # L1: NOR(x,y)     with bias ×
         OpNode("U", (0, 1, 3)),          # L1: NAND(x,y)    with bias +
         OpNode("D", (3,))],              # L1: +
        [OpNode("U", (0,)),               # L2: OR
         OpNode("U", (1,)),               # L2: AND
         OpNode("D", (2,))],              # L2: +
        [OpNode("U", (0,)),               # L3: NOR
         OpNode("C", (1,)),               # L3: × iff AND, else silent
         OpNode("D", (2,))],              # L3: +
        [OpNode("C", (0,)),               # L4: × iff NOR, else silent
         OpNode("C", (1,)),               # L4: × iff AND, else silent   (C∘C = C on {×, silent})
         OpNode("D", (2,))],              # L4: +
        [OpNode("U", (0, 0, 1, 1, 2))],   # L5: activation = −2[NOR] − 2[AND] + 1 → × iff neither
    ]
    return OpNet(4, layers), (TIMES, PLUS), (lambda x, y: x != y)


def gate_parity(n: int) -> Gate:
    """XOR tree over n = 2^d inputs. Wire layout at every layer: [signals..., +, ×]; the + wire is
    carried by D and the × wire recomputed each layer as U(+), so both biases are always available.
    Each level = the 5 XOR layers with the two bias nodes appended. Inputs: n signals, then +, ×."""
    if n & (n - 1) or n < 2:
        raise ValueError("n must be a power of two >= 2")
    xor_layers = gate_xor()[0].layers
    layers: List[List[OpNode]] = []
    width = n
    while width > 1:
        pairs = width // 2
        plus_idx, times_idx = width, width + 1
        prev_block_offsets: List[int] = []
        for Li, xl in enumerate(xor_layers):
            new_layer: List[OpNode] = []
            block_offsets: List[int] = []
            for b in range(pairs):
                block_offsets.append(len(new_layer))
                for node in xl:
                    if Li == 0:
                        m = {0: 2 * b, 1: 2 * b + 1, 2: times_idx, 3: plus_idx}
                        parents = tuple(m[p] for p in node.parents)
                    else:
                        parents = tuple(prev_block_offsets[b] + p for p in node.parents)
                    new_layer.append(OpNode(node.type, parents))
            new_layer.append(OpNode("D", (plus_idx,)))     # +
            new_layer.append(OpNode("U", (plus_idx,)))     # × = NOT +
            plus_idx, times_idx = len(new_layer) - 2, len(new_layer) - 1
            prev_block_offsets = block_offsets
            layers.append(new_layer)
        width = pairs   # the last XOR layer has one node per block, at indices 0..pairs-1
    return OpNet(n + 2, layers), (PLUS, TIMES), (lambda *bits: (sum(bits) % 2) == 1)


GATES: Dict[str, Callable[[], Gate]] = {
    "NOT": gate_not, "NOR": gate_nor, "OR": gate_or, "NAND": gate_nand, "AND": gate_and,
    "MAJ3": gate_maj3, "XOR": gate_xor,
}


def gate_ledger(parity_sizes: Sequence[int] = (2, 4, 8, 16)) -> List[Dict[str, Any]]:
    rows = []

    def add(name: str, g: Gate, n_vars: int):
        net, biases, fn = g
        ok, patterns, fails = verify(net, n_vars, biases, fn)
        rows.append(dict(gate=name, inputs=n_vars, bias_wires=len(biases), nodes=net.size, depth=net.depth,
                         verified=ok, patterns=patterns, failures=len(fails)))

    for name, build in GATES.items():
        g = build()
        add(name, g, g[0].n - len(g[1]))
    for n, k in ((5, 3), (7, 4), (8, 8), (8, 1), (9, 6)):
        tag = " (AND)" if k == n else " (OR)" if k == 1 else ""
        add(f"THR_{n}>={k}{tag}", gate_threshold(n, k), n)
    for n in parity_sizes:
        add(f"PARITY_{n}", gate_parity(n), n)
    return rows
