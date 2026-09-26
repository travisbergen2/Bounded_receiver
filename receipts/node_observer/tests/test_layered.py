import random
import unittest

from node_observer.layered import (
    G, I, NEG_I, ZERO, ONE, PLUS, TIMES, Circuit, Gate, configuration_table, lens_report,
    ChannelWorld, ChannelWorldParams, SkepticObserver, SkepticParams, LayeredRunParams, run_layered,
)


class TestGaussian(unittest.TestCase):
    def test_units(self):
        self.assertEqual(I * I, G(-1, 0))
        self.assertEqual(I * NEG_I, ONE)
        self.assertEqual(I + NEG_I, ZERO)
        self.assertEqual((I + I), G(0, 2))
        self.assertEqual(str(G(1, -2)), "1-2i")
        self.assertEqual(G(0, 2).kind(), "+i")
        self.assertEqual(G(-3, 0).kind(), "-1")
        self.assertEqual(G(1, 1).kind(), "off")
        self.assertEqual(ZERO.kind(), "0")

    def test_configuration_table(self):
        rows = configuration_table()
        self.assertEqual([r["plus"] for r in rows], ["0", "2i", "-2i"])
        self.assertEqual([r["times"] for r in rows], ["1", "-1", "-1"])


class TestCircuits(unittest.TestCase):
    def test_plus_tree_counts(self):
        c = Circuit.tree(4, [PLUS, PLUS])
        for signs, out in c.truth_table():
            k = sum(1 for s in signs if s > 0)
            self.assertEqual(out, [G(0, 2 * k - 4)])

    def test_times_tree_is_parity_and_sign_blind(self):
        c = Circuit.tree(4, [TIMES, TIMES])
        tt = dict(c.truth_table())
        for signs, out in tt.items():
            minus = sum(1 for s in signs if s < 0)
            self.assertEqual(out, [G((-1) ** minus, 0)])
            flipped = tuple(-s for s in signs)
            self.assertEqual(out, tt[flipped])

    def test_mixed_trees(self):
        agree = Circuit.tree(4, [TIMES, PLUS])   # x1x2 + x3x4
        zeros = sum(out[0].is_zero() for _, out in agree.truth_table())
        self.assertEqual(zeros, 8)
        self.assertEqual({str(out[0]) for _, out in agree.truth_table()}, {"2", "0", "-2"})
        allagree = Circuit.tree(4, [PLUS, TIMES])  # (x1+x2)(x3+x4)
        zeros = sum(out[0].is_zero() for _, out in allagree.truth_table())
        self.assertEqual(zeros, 12)
        self.assertEqual({str(out[0]) for _, out in allagree.truth_table()}, {"4", "0", "-4"})

    def test_lens_report_direction_and_parity(self):
        plus = lens_report(Circuit.tree(8, [PLUS] * 3))[0]
        self.assertAlmostEqual(plus["MI_direction_bits"], plus["H_direction_bits"], places=3)
        self.assertAlmostEqual(plus["MI_sign_bits"], 1.0, places=6)
        self.assertFalse(plus["sign_blind"])
        times = lens_report(Circuit.tree(8, [TIMES] * 3))[0]
        self.assertTrue(times["sign_blind"])
        self.assertAlmostEqual(times["MI_sign_bits"], 0.0, places=6)   # never which way it leans
        self.assertGreater(times["MI_direction_bits"], 0.3)             # but it sees the tie (even parity)
        self.assertAlmostEqual(times["MI_parity_bits"], 1.0, places=6)
        self.assertEqual(times["distinct_values"], 2)

    def test_band_shapes(self):
        b = Circuit.band(8, [PLUS])
        self.assertEqual(b.widths(), [8, 8])
        vals = b.output([I] * 4 + [NEG_I] * 4)
        self.assertEqual({str(v) for v in vals}, {"2i", "0", "-2i"})
        with self.assertRaises(ValueError):
            Circuit(2, [[Gate(PLUS, 0, 5)]])


class TestWorld(unittest.TestCase):
    def test_noiseless_channels_follow_regime(self):
        w = ChannelWorld(ChannelWorldParams(n=6, eps=0.0, p_switch=0.0), random.Random(0))
        for _ in range(5):
            w.step()
            self.assertEqual(w.signs, [w.regime] * 6)

    def test_trickster_lies_only_while_lying(self):
        w = ChannelWorld(ChannelWorldParams(n=4, eps=0.0, p_switch=0.0, trickster=2, lie_on=1.0, lie_len=1e9),
                         random.Random(0))
        w.step()
        self.assertTrue(w.lying)
        self.assertEqual(w.signs, [1, 1, -1, 1])


class TestObserver(unittest.TestCase):
    def test_skeptic_burns_a_persistent_liar_and_nobody_else(self):
        rp = LayeredRunParams(
            steps=300,
            world=ChannelWorldParams(n=8, eps=0.0, p_switch=0.0, trickster=3, lie_on=1.0, lie_len=1e9),
            observer=SkepticParams(trust_mode="skeptic"))
        r = run_layered(rp, seed=0)
        self.assertEqual(r["true_burns"], [3])
        self.assertEqual(r["false_burns"], [])
        self.assertLessEqual(r["first_true_burn"], 10)

    def test_fool_once_under_noise_stops_at_two_survivors(self):
        # one-strike burning stops exactly when two witnesses remain: two can only contradict a
        # consensus they themselves form
        for seed in (0, 1, 2):
            rp = LayeredRunParams(steps=2000, world=ChannelWorldParams(n=8, eps=0.15),
                                  observer=SkepticParams(trust_mode="fool_once"))
            r = run_layered(rp, seed=seed)
            self.assertEqual(r["n_burned"], 6)

    def test_trusting_never_burns(self):
        rp = LayeredRunParams(steps=2000, world=ChannelWorldParams(n=8, eps=0.3, trickster=1),
                              observer=SkepticParams(trust_mode="trusting"))
        self.assertEqual(run_layered(rp, seed=0)["n_burned"], 0)

    def test_forgiving_recovers_after_a_burst(self):
        rng = random.Random(0)
        circ = Circuit.identity(4)
        obs = SkepticObserver(SkepticParams(trust_mode="forgiving", forgive_rate=0.2), circ, rng)
        for _ in range(20):                      # witness 0 contradicts a +1 consensus
            obs.update_trust([-1, 1, 1, 1], +1)
        self.assertLess(obs.weights()[0], 0.05)
        for _ in range(40):                      # then agrees again
            obs.update_trust([1, 1, 1, 1], +1)
        self.assertGreater(obs.weights()[0], 0.9)

    def test_times_band_witnesses_are_direction_blind(self):
        rp = LayeredRunParams(steps=500, circuit_ops=(TIMES,), observer=SkepticParams(witness_layer=1))
        r = run_layered(rp, seed=0)
        self.assertEqual(r["bets"], 0)
        self.assertEqual(r["counts"]["abstain-blind"], 500)

    def test_regime_switch_is_not_read_as_lying(self):
        # noiseless world, forced switch: all witnesses flip together, the consensus flips, no burn
        rp = LayeredRunParams(steps=400, world=ChannelWorldParams(n=8, eps=0.0, p_switch=0.02),
                              observer=SkepticParams(trust_mode="fool_once"))
        r = run_layered(rp, seed=1)
        self.assertGreater(r["switches"], 0)
        self.assertEqual(r["n_burned"], 0)

    def test_determinism(self):
        rp = LayeredRunParams(steps=1500, world=ChannelWorldParams(trickster=3))
        self.assertEqual(run_layered(rp, 5), run_layered(rp, 5))
        self.assertNotEqual(run_layered(rp, 5)["score"], run_layered(rp, 6)["score"])


if __name__ == "__main__":
    unittest.main()
