import random
import unittest

from node_observer.fourval import Held, NOTHING, PLUS, MINUS, BOTH, BASIC, from_sign, tables, travis_configurations


class TestFourValues(unittest.TestCase):
    def test_four_readouts_are_distinct_and_nothing_differs_from_superposition(self):
        self.assertEqual([v.readout() for v in (NOTHING, PLUS, MINUS, BOTH)], ["0", "+", "-", "S"])
        self.assertNotEqual(BOTH.readout(), NOTHING.readout())
        # the old collapse cannot tell them apart
        self.assertEqual(BOTH.collapse(), NOTHING.collapse())

    def test_travis_rows(self):
        rows = {r["pair"]: r for r in travis_configurations()}
        # his + row, with the one change he asked for: (+)+(−) = S instead of 0
        self.assertEqual((rows["(+,+)"]["hold"], rows["(-,-)"]["hold"], rows["(+,-)"]["hold"]), ("+", "-", "S"))
        self.assertEqual(rows["(+,-)"]["hold_old_collapse"], "0")
        # his × row exactly
        self.assertEqual((rows["(+,+)"]["compare"], rows["(-,-)"]["compare"], rows["(+,-)"]["compare"]), ("+", "+", "-"))

    def test_hold_table(self):
        t = tables()["HOLD"]
        self.assertEqual(t[("+", "-")], "S")
        self.assertEqual(t[("S", "+")], "S")
        self.assertEqual(t[("0", "-")], "-")
        self.assertEqual(t[("0", "0")], "0")

    def test_compare_table(self):
        t = tables()["COMPARE"]
        self.assertEqual(t[("+", "-")], "-")
        self.assertEqual(t[("-", "-")], "+")
        self.assertEqual(t[("S", "+")], "S")
        self.assertEqual(t[("S", "S")], "S")
        self.assertEqual(t[("0", "S")], "0")

    def test_channels_are_independent_ring_coordinates(self):
        rng = random.Random(0)
        for _ in range(500):
            x = Held(rng.randrange(6), rng.randrange(6))
            y = Held(rng.randrange(6), rng.randrange(6))
            h, c = x.hold(y), x.compare(y)
            self.assertEqual(h.presence, x.presence + y.presence)
            self.assertEqual(h.direction, x.direction + y.direction)
            self.assertEqual(c.presence, x.presence * y.presence)
            self.assertEqual(c.direction, x.direction * y.direction)
            self.assertEqual((x.presence + x.direction) % 2, 0)   # same parity: the group ring, not all of Z×Z

    def test_superposition_is_the_kernel_of_the_collapse(self):
        for n in range(1, 6):
            self.assertEqual(Held(n, n).readout(), "S")
            self.assertEqual(Held(n, n).collapse(), 0)
            self.assertGreater(Held(n, n).presence, 0)
        self.assertEqual(from_sign(0), NOTHING)

    def test_counts_grade_the_contest(self):
        self.assertEqual(Held(3, 1).readout(), "S")          # contested, leaning plus
        self.assertEqual(Held(3, 1).collapse(), 1)            # the old node would say plus and forget the 1
        self.assertEqual(Held(3, 1).presence, 4)


if __name__ == "__main__":
    unittest.main()
