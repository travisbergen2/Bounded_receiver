import math
import unittest

from node_observer.grassmann import (
    grover_receipt, grover_optimal_iterations, grover_success_probability, over_holding,
    coordinate_plane_is_candidate_set, DensityMatrix, shannon_bits,
)


class TestGrover(unittest.TestCase):
    def test_receipt_N_1024(self):
        r = grover_receipt(1024, target=777)
        self.assertEqual(r["k_star"], 25)
        self.assertGreater(r["success_probability"], 0.999)
        self.assertLess(r["closed_form_error"], 1e-10)      # the iteration IS the rotation
        self.assertLess(r["plane_deviation"], 1e-10)        # and it never leaves the 2-plane
        self.assertAlmostEqual(r["norm"], 1.0, places=10)
        self.assertLess(abs(r["wrong_amplitude_end"]), abs(r["wrong_amplitude_start"]) / 30)
        self.assertGreater(r["classical_touches"], r["brute_force_checks"])   # simulating it is slower than searching

    def test_scaling_is_square_root(self):
        ks = [grover_optimal_iterations(4 ** m) for m in range(3, 8)]
        ratios = [ks[i + 1] / ks[i] for i in range(len(ks) - 1)]
        for r in ratios:
            self.assertAlmostEqual(r, 2.0, delta=0.15)      # N × 4 → iterations × 2
        self.assertGreater(grover_success_probability(4 ** 7, ks[-1]), 0.999)


class TestHolding(unittest.TestCase):
    def test_span_over_holds(self):
        r = over_holding()
        self.assertEqual(r["blend"], (1.0, 0.0))
        self.assertFalse(r["blend_is_sign_vector"])

    def test_coordinate_plane_is_a_subset(self):
        proj, recovered = coordinate_plane_is_candidate_set(6, [1, 3, 4])
        self.assertEqual(recovered, [1, 3, 4])
        self.assertEqual(sum(proj[i][i] for i in range(6)), 3)   # trace = k = the set's size


class TestDensity(unittest.TestCase):
    def test_classical_diagonal_matches_the_soft_filter_entropy(self):
        p = [0.5, 0.3, 0.15, 0.05]
        rho = DensityMatrix.classical(p)
        self.assertAlmostEqual(rho.trace(), 1.0)
        self.assertEqual(rho.rank(), 4)
        self.assertAlmostEqual(rho.entropy_bits(), shannon_bits(p), places=9)

    def test_pure_superposition_is_one_point(self):
        rho = DensityMatrix.pure([1.0, 1.0, 1.0, 1.0])     # equal superposition of four alternatives
        self.assertEqual(rho.rank(), 1)
        self.assertAlmostEqual(rho.entropy_bits(), 0.0, places=9)
        self.assertAlmostEqual(rho.purity(), 1.0, places=9)

    def test_uniform_mixture_is_the_candidate_set(self):
        rho = DensityMatrix.uniform_mixture(6, 4)
        self.assertEqual(rho.rank(), 4)
        self.assertAlmostEqual(rho.entropy_bits(), 2.0, places=9)   # log2 4
        self.assertAlmostEqual(rho.purity(), 0.25, places=9)         # 1/k


if __name__ == "__main__":
    unittest.main()
