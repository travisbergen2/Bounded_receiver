import unittest

from run_tribe import run_tribe


class TestTribe(unittest.TestCase):
    def test_one_strike_burns_the_sharp_witness_first_and_threshold_eight_keeps_it(self):
        r1 = run_tribe("fool_once", 0.0, steps=3000, seed=0)
        self.assertGreater(r1["sharp_accuracy"], r1["slow_accuracy_median"])
        self.assertTrue(r1["sharp_burned"])
        self.assertTrue(r1["first_burned_is_sharp"])
        self.assertEqual(r1["n_slow_burned"], 0)
        r8 = run_tribe("skeptic", 8.0, steps=3000, seed=0)
        self.assertFalse(r8["sharp_burned"])
        self.assertEqual(r8["n_slow_burned"], 0)

    def test_group_loses_nothing_measurable(self):
        r = run_tribe("fool_once", 0.0, steps=5000, seed=1)
        self.assertAlmostEqual(r["group_accuracy_with_sharp"], r["group_accuracy_without_sharp"], places=2)


if __name__ == "__main__":
    unittest.main()
