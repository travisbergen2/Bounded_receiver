"""Receipts for "would it help to build this as a Grassmannian?" (Travis, 2026-09-26).

Three small, exact computations:

1. grover(N, target, k): the actual Grover iteration on a real amplitude vector of length N (oracle sign
   flip, then reflection about the mean), checked against the closed form
        a_k = sin((2k+1)θ) on the target,  b_k = cos((2k+1)θ)/sqrt(N−1) on every other index,  sin θ = 1/√N,
   and checked to stay in the 2-plane span{target, uniform-rest} — i.e. the whole algorithm is a rotation
   on Gr(1, 2), the projective line inside that plane. Also counts the classical work: every iteration
   touches all N amplitudes (the mean), so simulating the rotation costs more than brute-force search.

2. over_holding(): the span of two sign vectors contains vectors that are not sign vectors — a linear
   hold admits blends that were never among the alternatives. And coordinate_plane_is_candidate_set():
   a coordinate k-plane of R^N carries exactly the information of a k-subset — the hard filter.

3. DensityMatrix: the weighted version of "hold k alternatives". rank = dimension of the superposition,
   eigenvalues = weights. The diagonal sector is the classical posterior (same entropy as the soft
   filter); a pure superposition of k alternatives is rank 1 with zero entropy (ONE point of Gr(1, N)),
   and the uniform classical mixture is rank k with entropy log2 k.

Pure Python. Dense linear algebra only where N is small.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# ----------------------------------------------------------------------------- 1. Grover on Gr(1, 2)


def grover_theta(N: int) -> float:
    return math.asin(1.0 / math.sqrt(N))


def grover_optimal_iterations(N: int) -> int:
    th = grover_theta(N)
    return int(round(math.pi / (4.0 * th) - 0.5))


def grover_success_probability(N: int, k: int) -> float:
    return math.sin((2 * k + 1) * grover_theta(N)) ** 2


def grover_iterate(N: int, target: int, k: int) -> Tuple[List[float], int]:
    """Run k Grover iterations on the uniform state. Returns (amplitudes, classical_touches)."""
    psi = [1.0 / math.sqrt(N)] * N
    touches = N  # preparing the uniform state
    for _ in range(k):
        psi[target] = -psi[target]                 # oracle: flip the sign of the marked amplitude
        mean = sum(psi) / N                        # reflection about the mean needs every amplitude
        psi = [2.0 * mean - a for a in psi]
        touches += 2 * N
    return psi, touches


def grover_receipt(N: int = 1024, target: int = 777) -> Dict[str, float]:
    th = grover_theta(N)
    k_star = grover_optimal_iterations(N)
    psi, touches = grover_iterate(N, target, k_star)
    a = math.sin((2 * k_star + 1) * th)
    b = math.cos((2 * k_star + 1) * th) / math.sqrt(N - 1)
    closed_form_err = max(abs(psi[target] - a), max(abs(psi[j] - b) for j in range(N) if j != target))
    # in-plane check: every non-target amplitude equal (the state is a combination of |target> and |rest>)
    others = [psi[j] for j in range(N) if j != target]
    plane_dev = max(others) - min(others)
    # amplitude trajectory of the wrong answers: the interference that makes them shrink
    wrong_amp = [math.cos((2 * k + 1) * th) / math.sqrt(N - 1) for k in range(k_star + 1)]
    return dict(N=N, theta=th, angle_per_iteration=2 * th, k_star=k_star,
                success_probability=psi[target] ** 2,
                success_probability_closed_form=grover_success_probability(N, k_star),
                closed_form_error=closed_form_err, plane_deviation=plane_dev,
                norm=sum(x * x for x in psi),
                wrong_amplitude_start=wrong_amp[0], wrong_amplitude_end=wrong_amp[-1],
                classical_touches=touches, brute_force_checks=N,
                classical_slower_by=touches / N)


# ----------------------------------------------------------------------------- 2. holding as a span


def over_holding() -> Dict[str, object]:
    """span{(+1,+1), (+1,-1)} is the whole plane; it contains (1, 0), which is not a sign vector."""
    u, v = (1.0, 1.0), (1.0, -1.0)
    blend = tuple(0.5 * (ui + vi) for ui, vi in zip(u, v))   # (1, 0)
    return dict(alternatives=[u, v], blend=blend, blend_is_sign_vector=all(abs(c) == 1.0 for c in blend))


def coordinate_plane_is_candidate_set(N: int, subset: Sequence[int]) -> Tuple[List[List[int]], List[int]]:
    """A coordinate k-plane of R^N (span of k basis vectors) and the k-subset it encodes are the same
    datum: the projector onto the plane is the diagonal 0/1 matrix of the subset's indicator."""
    proj = [[1 if (i == j and i in subset) else 0 for j in range(N)] for i in range(N)]
    recovered = [i for i in range(N) if proj[i][i] == 1]
    return proj, recovered


# ----------------------------------------------------------------------------- 3. density matrices


@dataclass
class DensityMatrix:
    rho: List[List[float]]

    @property
    def n(self) -> int:
        return len(self.rho)

    def trace(self) -> float:
        return sum(self.rho[i][i] for i in range(self.n))

    def eigenvalues(self) -> List[float]:
        """Eigenvalues by Jacobi rotations (real symmetric, small n)."""
        A = [row[:] for row in self.rho]
        n = self.n
        for _ in range(100):
            off = sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
            if off < 1e-24:
                break
            for p in range(n):
                for q in range(p + 1, n):
                    if abs(A[p][q]) < 1e-15:
                        continue
                    phi = 0.5 * math.atan2(2 * A[p][q], A[q][q] - A[p][p])
                    c, s = math.cos(phi), math.sin(phi)
                    for k in range(n):
                        akp, akq = A[k][p], A[k][q]
                        A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
                    for k in range(n):
                        apk, aqk = A[p][k], A[q][k]
                        A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
        return sorted((A[i][i] for i in range(n)), reverse=True)

    def rank(self, tol: float = 1e-9) -> int:
        return sum(1 for e in self.eigenvalues() if e > tol)

    def entropy_bits(self) -> float:
        return -sum(e * math.log2(e) for e in self.eigenvalues() if e > 1e-15)

    def purity(self) -> float:
        n = self.n
        return sum(self.rho[i][j] * self.rho[j][i] for i in range(n) for j in range(n))

    @staticmethod
    def classical(p: Sequence[float]) -> "DensityMatrix":
        """Diagonal: the soft filter's posterior as a density matrix (no coherence)."""
        n = len(p)
        return DensityMatrix([[p[i] if i == j else 0.0 for j in range(n)] for i in range(n)])

    @staticmethod
    def pure(amplitudes: Sequence[float]) -> "DensityMatrix":
        """|psi><psi| for a real unit vector: a pure superposition — one point of Gr(1, n)."""
        norm = math.sqrt(sum(a * a for a in amplitudes))
        v = [a / norm for a in amplitudes]
        return DensityMatrix([[v[i] * v[j] for j in range(len(v))] for i in range(len(v))])

    @staticmethod
    def uniform_mixture(n: int, k: int) -> "DensityMatrix":
        """P/k for the coordinate k-plane: the hard filter's candidate set with equal weights."""
        return DensityMatrix([[(1.0 / k if (i == j and i < k) else 0.0) for j in range(n)] for i in range(n)])


def shannon_bits(p: Sequence[float]) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)
