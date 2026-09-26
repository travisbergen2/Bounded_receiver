import random
import unittest

from node_observer.model import (
    PLUS, TIMES, PAIR_I, PAIR_W, QuadraticPair, Node, Network, World, WorldParams,
    Observer, ObserverParams, RunParams, run, default_network, gaussian_table, eisenstein_table,
)


class TestPairs(unittest.TestCase):
    def test_i_pair_is_binary(self):
        self.assertEqual(PAIR_I.emit(PLUS), 0)    # (+i) + (-i) cancels to 0
        self.assertEqual(PAIR_I.emit(TIMES), 1)   # (+i) * (-i) combines to 1
        self.assertEqual(PAIR_I.gap, 1)
        self.assertFalse(PAIR_I.blind)

    def test_omega_pair(self):
        self.assertEqual(PAIR_W.trace, -1)
        self.assertEqual(PAIR_W.norm, 1)
        self.assertEqual(PAIR_W.gap, 2)

    def test_dist2_identity(self):
        # |alpha - 1|^2 = norm - trace + 1, checked against floating complex arithmetic
        import cmath
        for a in range(-3, 4):
            for b in range(-3, 4):
                g = QuadraticPair(a, b, "gauss")
                z = complex(a, b)
                self.assertAlmostEqual(abs(z - 1) ** 2, g.dist2_from_one, places=9)
                e = QuadraticPair(a, b, "eisen")
                w = cmath.exp(2j * cmath.pi / 3)
                z = a + b * w
                self.assertAlmostEqual(abs(z - 1) ** 2, e.dist2_from_one, places=9)

    def test_blind_iff_on_circle(self):
        for row in gaussian_table(3) + eisenstein_table(3):
            self.assertEqual(row["blind"], row["dist2"] == 1)

    def test_blind_lattice_points_are_one_plus_roots_of_unity(self):
        g_blind = sorted((r["a"], r["b"]) for r in gaussian_table(3) if r["blind"])
        # 1 + {1, i, -1, -i} = {2, 1+i, 0, 1-i}; the table keeps b >= 0 so 1-i is absent
        self.assertEqual(g_blind, [(0, 0), (1, 1), (2, 0)])
        e_blind = sorted((r["a"], r["b"]) for r in eisenstein_table(3) if r["blind"])
        # 1 + sixth roots of unity in the basis (1, w): 2, 1+(1+w)=2+w, 1+w, 0, 1-(1+w)=-w, 1-w
        self.assertEqual(e_blind, [(0, -1), (0, 0), (1, -1), (1, 1), (2, 0), (2, 1)])
        self.assertEqual(len(e_blind), 6)

    def test_decode_roundtrip_and_blind(self):
        for pair in (PAIR_I, PAIR_W, QuadraticPair(1, 2, "gauss")):
            for op in (PLUS, TIMES):
                self.assertEqual(pair.decode(pair.emit(op)), op)
        self.assertIsNone(QuadraticPair(1, 1, "gauss").decode(2))


class TestNetwork(unittest.TestCase):
    def test_quadrature_gray_code(self):
        net = default_network(4, 2)
        codes = [tuple(net.bits(h)) for h in range(4)]
        self.assertEqual(codes, [(1, 0), (1, 1), (0, 1), (0, 0)])
        for h in range(4):
            self.assertEqual(net.candidates(net.emit(h)), [h])

    def test_one_node_is_direction_blind(self):
        # forward and backward phase sequences give cyclic shifts of one another: 1100 vs 1001
        net = default_network(4, 1)
        fwd = [net.readout_bit(h % 4) for h in range(8)]
        bwd = [net.readout_bit((-h) % 4) for h in range(8)]
        self.assertEqual(fwd[:4], [1, 1, 0, 0])
        self.assertEqual(bwd[:4], [1, 0, 0, 1])
        self.assertIn(tuple(bwd[:4]), {tuple(fwd[k:k + 4]) for k in range(4)})
        for h in range(4):
            self.assertEqual(len(net.candidates(net.emit(h))), 2)

    def test_blind_nodes_carry_nothing(self):
        net = default_network(4, 2, QuadraticPair(1, 1, "gauss"))
        for h in range(4):
            self.assertEqual(net.candidates(net.emit(h)), [0, 1, 2, 3])

    def test_eisenstein_thermometer_code(self):
        net = default_network(6, 3, PAIR_W)
        codes = [tuple(net.bits(h)) for h in range(6)]
        self.assertEqual(len(set(codes)), 6)
        for h in range(6):
            self.assertEqual(net.candidates(net.emit(h)), [h])


class TestWorld(unittest.TestCase):
    def test_noiseless_world_turns_one_notch(self):
        w = World(WorldParams(n=4, eps=0.0, p_switch=0.0), random.Random(1))
        h0 = w.phase
        for t in range(1, 9):
            w.step()
            self.assertEqual(w.phase, (h0 + t) % 4)
            self.assertEqual(w.direction_of(w.last_step), +1)


class TestObserver(unittest.TestCase):
    def test_honest_zero_noise_observer_is_perfect_after_warmup(self):
        rp = RunParams(steps=2000, world=WorldParams(eps=0.0, p_switch=0.0))
        r = run(rp, seed=3)
        self.assertEqual(r["misses"], 0)
        self.assertEqual(r["alarms"], 0)
        self.assertGreater(r["hits"], 1900)

    def test_detector_fires_after_a_switch_and_resets_estimate(self):
        rng = random.Random(0)
        net = default_network(4, 2)
        obs = Observer(ObserverParams(TI=8, FT=3), net, rng)
        obs.E = 1.0
        fired = False
        for _ in range(20):
            fired = obs.detect(-1) or fired
            if fired:
                break
        self.assertTrue(fired)
        self.assertEqual(obs.E, 0.0)
        self.assertEqual(obs.seam, obs.p.seam_len)

    def test_detector_off_never_fires(self):
        rp = RunParams(steps=5000, observer=ObserverParams(detect_on=False))
        r = run(rp, seed=5)
        self.assertEqual(r["alarms"], 0)

    def test_one_node_arm_never_bets(self):
        rp = RunParams(steps=3000, n_nodes=1)
        r = run(rp, seed=7)
        self.assertEqual(r["bets"], 0)
        self.assertEqual(r["counts"]["abstain-undetermined"], 3000)

    def test_blind_pair_arm_never_bets(self):
        rp = RunParams(steps=1000, n_nodes=2, pair=QuadraticPair(1, 1, "gauss"))
        r = run(rp, seed=7)
        self.assertEqual(r["bets"], 0)

    def test_pure_ego_records_only_right_decisions(self):
        rp = RunParams(steps=20000, observer=ObserverParams(kappa=1.0))
        r = run(rp, seed=11)
        self.assertGreater(r["recorded"], 0)
        self.assertEqual(r["narrative_rate"], 1.0)

    def test_honest_narrative_matches_truth_within_tolerance(self):
        rp = RunParams(steps=40000, observer=ObserverParams(kappa=0.0))
        r = run(rp, seed=11)
        self.assertIsNotNone(r["ego_gap"])
        self.assertLess(abs(r["ego_gap"]), 0.06)

    def test_determinism(self):
        rp = RunParams(steps=3000)
        a, b = run(rp, seed=42), run(rp, seed=42)
        self.assertEqual(a, b)
        c = run(rp, seed=43)
        self.assertNotEqual(a["bits0"], c["bits0"])


if __name__ == "__main__":
    unittest.main()
