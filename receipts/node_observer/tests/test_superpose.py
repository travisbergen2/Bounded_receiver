import random
import unittest

from node_observer.model import World, WorldParams, default_network, PAIR_I
from node_observer.superpose import HardFilter, SoftFilter, SuperposeParams, run_superpose, factored_demo


class TestHardFilter(unittest.TestCase):
    def test_noiseless_collapse_and_empty_set_at_switch(self):
        rng = random.Random(1)
        net = default_network(4, 2)
        world = World(WorldParams(n=4, eps=0.0, p_switch=0.0), rng)
        f = HardFilter(4, net)
        f.observe(net.emit(world.phase))
        self.assertEqual(f.size, 2)                     # phase known (quadrature), regime open: the superposition
        f.predict(); world.step(); self.assertFalse(f.observe(net.emit(world.phase)))
        self.assertTrue(f.collapsed)
        self.assertEqual(f.answer(), (world.phase, world.regime))
        # force a regime switch: the prediction fails, the set empties, the filter resets
        world.regime = -world.regime
        f.predict(); world.step()
        self.assertTrue(f.observe(net.emit(world.phase)))
        self.assertEqual(f.resets, 1)
        self.assertEqual(f.size, 2)
        f.predict(); world.step(); f.observe(net.emit(world.phase))
        self.assertEqual(f.answer(), (world.phase, world.regime))

    def test_noiseless_run_has_no_false_alarms(self):
        r = run_superpose(SuperposeParams(steps=3000, n=4, eps=0.0, p_switch=0.02, mode="hard"), seed=0)
        # in a noiseless world nothing but a switch can refute a hypothesis: no spurious alarm, ever
        self.assertEqual(r["false_alarms"], 0)
        self.assertEqual(r["attributed"], r["alarms"])
        # every switch that happens while the filter is collapsed empties the set at once (delay 0);
        # a switch that happens while both regimes are still open is absorbed, not missed
        self.assertEqual(r["detected_while_collapsed"], r["switches_while_collapsed"])
        self.assertEqual(r["median_delay"], 0)
        self.assertEqual(r["answer_accuracy"], 1.0)

    def test_noisy_hard_filter_is_brittle(self):
        r = run_superpose(SuperposeParams(steps=3000, n=4, eps=0.15, p_switch=0.005, mode="hard"), seed=0)
        self.assertGreater(r["false_alarms"], 5 * max(1, r["switches"]))


class TestSoftFilter(unittest.TestCase):
    def test_posterior_normalises_and_collapses_in_a_quiet_world(self):
        rng = random.Random(2)
        net = default_network(4, 2)
        world = World(WorldParams(n=4, eps=0.05, p_switch=0.0), rng)
        f = SoftFilter(4, net, eps=0.05, p_switch=0.001)
        f.observe(net.emit(world.phase))
        for _ in range(30):
            f.predict(); world.step(); f.observe(net.emit(world.phase))
            self.assertAlmostEqual(sum(f.w.values()), 1.0, places=9)
        self.assertTrue(f.collapsed)
        self.assertEqual(f.top()[0], (world.phase, world.regime))

    def test_single_step_surprise_cannot_see_a_switch_but_the_posterior_flip_can(self):
        r = run_superpose(SuperposeParams(steps=4000, n=4, eps=0.15, p_switch=0.005, mode="soft"), seed=0)
        # P(obs | past) after a switch is about eps/n, the same as after a noise step: no single-step
        # surprise below 0.02 ever occurs (the pre-stated P4, refuted as written)
        self.assertEqual(r["surprises"], 0)
        self.assertGreater(r["min_p_obs"], 0.02)
        # the maximum-a-posteriori regime flipping sides is the emergent detector
        self.assertGreater(r["detected"], 0)
        self.assertGreaterEqual(r["detected"], r["switches"] // 2)
        self.assertLess(r["false_alarms"], r["alarms"])


class TestFactored(unittest.TestCase):
    def test_joint_is_product_held_at_sum(self):
        rows = factored_demo(4, 16, steps=8, seed=0)
        for row in rows:
            self.assertEqual(row["joint"], row["size1"] * row["size2"])
            self.assertEqual(row["held"], row["size1"] + row["size2"])
        self.assertEqual(rows[-1]["joint"], 1)


if __name__ == "__main__":
    unittest.main()
