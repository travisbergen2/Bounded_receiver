import cmath
import math
import unittest

from node_observer.exponent import (
    unit, exponent, exponent_law_receipt, value_plus_leaves_the_units, complex_power_i, z_to_the_i_table,
    channel_swap_receipt, branch_receipt, affine_fence_receipt,
)


class TestExponentLaw(unittest.TestCase):
    def test_units_are_powers_of_i_and_times_is_plus_on_exponents(self):
        r = exponent_law_receipt()
        self.assertTrue(r["exponent_law_holds"])
        self.assertTrue(r["is_cyclic_group_of_order_4"])
        self.assertEqual(exponent(unit(3) * unit(3)), 2)      # (−i)(−i) = −1 = i^2

    def test_value_plus_leaves_the_unit_group(self):
        v = value_plus_leaves_the_units()
        self.assertEqual(v["i^0 + i^1"], "1+i (norm 2)")
        self.assertEqual(v["i^1 + i^3"], "0 (norm 0)")
        self.assertEqual(v["i^0 + i^0"], "2 (norm 4)")


class TestComplexPower(unittest.TestCase):
    def test_i_to_the_i_is_real(self):
        rows = {r["unit"]: r for r in z_to_the_i_table()}
        self.assertTrue(all(r["is_real"] for r in rows.values()))
        self.assertAlmostEqual(rows["i"]["z_to_the_i"].real, math.exp(-math.pi / 2), places=6)
        self.assertAlmostEqual(rows["-1"]["z_to_the_i"].real, math.exp(-math.pi), places=6)
        self.assertAlmostEqual(rows["-i"]["z_to_the_i"].real, math.exp(math.pi / 2), places=6)
        self.assertAlmostEqual(rows["1"]["z_to_the_i"].real, 1.0, places=12)
        self.assertEqual(len({round(r["z_to_the_i"].real, 6) for r in rows.values()}), 4)   # four distinct sizes

    def test_channel_swap(self):
        for row in channel_swap_receipt([2 + 1j, -0.5 + 3j, 0.3 - 0.4j, 5.0]):
            self.assertAlmostEqual(row["abs_w"], row["predicted_abs"], places=9)
            self.assertAlmostEqual(math.cos(row["arg_w"]), math.cos(row["predicted_arg"]), places=9)
            self.assertAlmostEqual(math.sin(row["arg_w"]), math.sin(row["predicted_arg"]), places=9)

    def test_branches_differ_by_e_to_minus_2pi(self):
        b = branch_receipt(-1, (0, 1, 2))
        self.assertAlmostEqual(b[0], math.exp(-math.pi), places=12)
        self.assertAlmostEqual(b[1] / b[0], math.exp(-2 * math.pi), places=12)


class TestAffineFence(unittest.TestCase):
    def test_adders_are_affine_and_or_not(self):
        r = affine_fence_receipt()
        self.assertTrue(r["adder_network_is_affine"])
        self.assertFalse(r["AND_is_affine"])
        self.assertFalse(r["value_plus_agreement_is_affine"])


if __name__ == "__main__":
    unittest.main()
