import math
import random
import unittest

from node_observer.layered import ChannelWorldParams
from node_observer.hetero import HeteroObserver, HeteroParams, run_hetero


class TestHetero(unittest.TestCase):
    def test_currency_and_band(self):
        w = ChannelWorldParams(n=8, eps=0.15, p_switch=0.005)
        obs = HeteroObserver(HeteroParams(alpha=0.05), w, random.Random(0))
        self.assertAlmostEqual(obs.wgt, math.log(0.85 / 0.15), places=12)   # ℓ per channel = log((1−ε)/ε)
        self.assertAlmostEqual(obs.A, math.log(0.95 / 0.05), places=12)     # Wald's band
        obs.estimate([1] * 8)
        self.assertGreater(obs.L, obs.A)                                     # eight agreeing channels clear the band at once
        obs.estimate([-1] * 8)
        self.assertLess(obs.L, 0.0)                                          # and eight opposing ones flip it in one step

    def test_prior_mixing_saturates_the_odds(self):
        w = ChannelWorldParams(n=8, eps=0.15, p_switch=0.005)
        obs = HeteroObserver(HeteroParams(), w, random.Random(0))
        for _ in range(200):
            obs.estimate([1] * 8)
        cap = math.log((1 - w.p_switch) / w.p_switch) + obs.wgt * 8
        self.assertLessEqual(obs.L, cap + 1e-9)                              # the switch prior bounds the confidence

    def test_side_switch_detects_a_switch_without_a_threshold(self):
        w = ChannelWorldParams(n=8, eps=0.0, p_switch=0.0)
        obs = HeteroObserver(HeteroParams(), w if w.eps > 0 else ChannelWorldParams(n=8, eps=0.01, p_switch=0.01),
                             random.Random(0))
        obs.estimate([1] * 8); obs.detect()
        obs.estimate([1] * 8); self.assertFalse(obs.detect())
        obs.estimate([-1] * 8); self.assertTrue(obs.detect())
        self.assertEqual(len(obs.alarms), 1)

    def test_run_noise_world(self):
        r = run_hetero(ChannelWorldParams(n=8, eps=0.15, p_switch=0.005), HeteroParams(), steps=5000, seed=0)
        self.assertGreater(r["regime_accuracy"], 0.95)
        self.assertGreater(r["hit_rate"], 0.9)
        self.assertGreaterEqual(r["detected"], int(0.8 * r["switches"]))
        self.assertLess(r["ambiguous_frac"], 0.05)

    def test_determinism(self):
        w = ChannelWorldParams(n=8, eps=0.15, p_switch=0.005, trickster=3)
        self.assertEqual(run_hetero(w, HeteroParams(), 2000, 4), run_hetero(w, HeteroParams(), 2000, 4))


if __name__ == "__main__":
    unittest.main()
