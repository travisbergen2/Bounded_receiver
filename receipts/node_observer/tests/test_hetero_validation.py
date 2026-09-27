import math
import random
import unittest

from node_observer.layered import ChannelWorldParams
from node_observer.hetero import HeteroObserver, HeteroParams


class TestHeteroValidation(unittest.TestCase):
    """Domain guards added 2026-09-27 after an outside review (Manus): endpoint noise and alpha values,
    a zero switch prior and a zero tau are refused instead of producing infinite or undefined odds."""

    def test_rejects_out_of_domain(self):
        rng = random.Random(0)
        for bad_world in (ChannelWorldParams(n=8, eps=0.0), ChannelWorldParams(n=8, eps=1.0),
                          ChannelWorldParams(n=8, eps=0.15, p_switch=0.0), ChannelWorldParams(n=8, eps=0.15, p_switch=1.0)):
            with self.assertRaises(ValueError):
                HeteroObserver(HeteroParams(), bad_world, rng)
        good = ChannelWorldParams(n=8, eps=0.15, p_switch=0.005)
        for bad in (HeteroParams(alpha=0.0), HeteroParams(alpha=1.0), HeteroParams(tau=0.0)):
            with self.assertRaises(ValueError):
                HeteroObserver(bad, good, rng)
        HeteroObserver(HeteroParams(), good, rng)  # the default is inside the domain

    def test_prior_mixing_bounds_the_odds(self):
        # the log-odds after prior mixing lie within ±log((1-p)/p) for any starting value
        w = ChannelWorldParams(n=8, eps=0.15, p_switch=0.005)
        cap = math.log((1 - w.p_switch) / w.p_switch)
        for L0 in (-50.0, -5.0, 0.0, 3.0, 40.0):
            obs = HeteroObserver(HeteroParams(), w, random.Random(0))
            obs.L = L0
            obs.estimate([])                     # prior mixing only (no channels)
            self.assertLessEqual(abs(obs.L), cap + 1e-9)


if __name__ == "__main__":
    unittest.main()
