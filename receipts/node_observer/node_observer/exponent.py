"""Receipts for Travis's exponent proposal (2026-09-26, 10:06): "(i¹, −i¹) and (1ⁱ, (−1)ⁱ), and send + and ×
together since (×^+)".

1. THE EXPONENT LAW, exactly.  The Gaussian units are the powers of i:  i^n, n in Z/4  (1, i, −1, −i).
   Multiplying units is adding exponents mod 4:  i^a · i^b = i^(a+b).  So "send + and × together" is the
   exponential map — one operation seen in two coordinates. Checked exactly in Gaussian-integer arithmetic.

2. COMPLEX POWERS, principal branch.  z^i = exp(i · log z) with log z = ln|z| + i·arg z, arg in (−π, π]:
        z^i = e^(−arg z) · e^(i · ln|z|)
   The imaginary power SWAPS the two channels: the angle becomes a (decaying) magnitude and the log-size
   becomes the angle. For units (|z| = 1) it is pure magnitude:  1^i = 1,  i^i = e^(−π/2),  (−1)^i = e^(−π),
   (−i)^i = e^(π/2) — four distinct real numbers; direction turned into size. Fence: z^i is multi-valued
   (other branches differ by factors e^(−2πk)); everything here uses the principal branch.

3. THE AFFINE FENCE.  A network whose nodes only add exponents mod 4 computes only affine maps over Z/4
   (composition of affine maps is affine). AND on {0,1} ⊂ Z/4 is not affine. The nonlinearity has to come
   from the VALUE side (adding units: 1 + i is off the axes, i + (−i) = 0) or from a readout (sign/norm) —
   which is where v0.2 and v0.3 got theirs.
"""
from __future__ import annotations

import cmath
import math
from itertools import product
from typing import Dict, List, Tuple

from .layered import G, I, ONE as ONE_G, apply, PLUS, TIMES

UNITS_BY_EXP: Dict[int, G] = {0: G(1, 0), 1: G(0, 1), 2: G(-1, 0), 3: G(0, -1)}
EXP_BY_UNIT: Dict[Tuple[int, int], int] = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}


def unit(n: int) -> G:
    return UNITS_BY_EXP[n % 4]


def exponent(u: G) -> int:
    return EXP_BY_UNIT[(u.a, u.b)]


def exponent_law_receipt() -> Dict[str, object]:
    """i^a · i^b == i^(a+b) for all a, b in Z/4, exactly; and the multiplication table is the addition table."""
    ok = True
    table = {}
    for a in range(4):
        for b in range(4):
            prod = unit(a) * unit(b)
            ok &= (prod == unit(a + b))
            table[(a, b)] = exponent(prod)
    return dict(exponent_law_holds=ok, times_table_as_exponent_sums=table,
                is_cyclic_group_of_order_4=all(table[(a, b)] == (a + b) % 4 for a in range(4) for b in range(4)))


def value_plus_leaves_the_units() -> Dict[str, str]:
    """Adding two units: the results, with their norms — the value-side + is where the nonlinearity lives."""
    out = {}
    for a in range(4):
        for b in range(a, 4):
            s = unit(a) + unit(b)
            out[f"i^{a} + i^{b}"] = f"{s} (norm {s.norm})"
    return out


def complex_power_i(z: complex) -> complex:
    """Principal branch of z^i = exp(i log z)."""
    if z == 0:
        raise ValueError("0^i is undefined")
    return cmath.exp(1j * cmath.log(z))


def z_to_the_i_table() -> List[Dict[str, object]]:
    rows = []
    for n in range(4):
        u = complex(unit(n).a, unit(n).b)
        w = complex_power_i(u)
        rows.append(dict(unit=str(unit(n)), exponent=n, arg_deg=round(math.degrees(cmath.phase(u)), 1),
                         z_to_the_i=complex(round(w.real, 6), round(w.imag, 6)),
                         predicted_exp_minus_arg=round(math.exp(-cmath.phase(u)), 6),
                         is_real=abs(w.imag) < 1e-12))
    return rows


def channel_swap_receipt(samples: List[complex]) -> List[Dict[str, float]]:
    """z^i = e^(−arg z) · e^(i ln|z|): |z^i| = e^(−arg z) and arg(z^i) = ln|z| (mod 2π), principal branch."""
    rows = []
    for z in samples:
        w = complex_power_i(z)
        rows.append(dict(z=z, abs_w=abs(w), predicted_abs=math.exp(-cmath.phase(z)),
                         arg_w=cmath.phase(w), predicted_arg=math.atan2(math.sin(math.log(abs(z))), math.cos(math.log(abs(z))))))
    return rows


def branch_receipt(z: complex = -1, k_values=(0, 1, 2)) -> Dict[int, float]:
    """The other branches of (−1)^i: e^(−π(2k+1)) — the same base, different sheets."""
    return {k: math.exp(-(cmath.phase(z) + 2 * math.pi * k)) for k in k_values}


def is_affine_mod4(f: Dict[Tuple[int, ...], int], n_inputs: int) -> bool:
    """Is f: (Z/4)^n -> Z/4 of the form c + Σ w_j x_j (mod 4)? Fit from the basis points and verify everywhere."""
    zero = tuple([0] * n_inputs)
    c = f[zero]
    w = []
    for j in range(n_inputs):
        e = tuple(1 if k == j else 0 for k in range(n_inputs))
        w.append((f[e] - c) % 4)
    for x, y in f.items():
        if (c + sum(wj * xj for wj, xj in zip(w, x))) % 4 != y % 4:
            return False
    return True


def affine_fence_receipt() -> Dict[str, object]:
    """Every network of mod-4 adders is affine; AND is not; the value-side + gives a non-affine function."""
    # a two-layer adder network with integer weights: x ↦ 3x + y + 2, then ↦ 2·(that) + x
    adder_net = {(x, y): (2 * ((3 * x + y + 2) % 4) + x) % 4 for x, y in product(range(4), repeat=2)}
    and_fn = {(x, y): (1 if (x == 1 and y == 1) else 0) for x, y in product(range(4), repeat=2)}
    # value-side +: exponent of the sum's sign-class-like readout: 0 if the sum is 0, else 1  (agreement test)
    agree = {(x, y): (0 if (unit(x) + unit(y)).is_zero() else 1) for x, y in product(range(4), repeat=2)}
    return dict(adder_network_is_affine=is_affine_mod4(adder_net, 2), AND_is_affine=is_affine_mod4(and_fn, 2),
                value_plus_agreement_is_affine=is_affine_mod4(agree, 2))
