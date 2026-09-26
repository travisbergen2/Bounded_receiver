import unittest

from node_observer.layered import PLUS, TIMES
from node_observer.opnet import (
    OpNet, OpNode, SILENT, PAIRS, PAIRS_REAL, response, response_table, chain_monoid, verify,
)


class TestRealUnitsTable(unittest.TestCase):
    """Travis's sign table (2026-09-25, 21:55): (+)+(+)=+, (+)×(+)=+, (−)+(−)=−, (−)×(−)=+, (+)+(−)=0,
    (+)×(−)=−. It is the real-unit table; the imaginary-unit table negates the × row (i² = −1)."""

    def test_plus_rows_agree_and_times_rows_are_negated(self):
        for t in ("C", "U", "D"):
            self.assertEqual(response(t, PLUS, PAIRS), response(t, PLUS, PAIRS_REAL))
            self.assertEqual(response(t, TIMES, PAIRS), -response(t, TIMES, PAIRS_REAL))

    def test_travis_table_verbatim(self):
        rows = {r["type"]: r for r in response_table(PAIRS_REAL)}
        self.assertEqual((rows["U"]["+ class"], rows["U"]["* class"]), (+1, +1))   # (+)+(+)=+, (+)×(+)=+
        self.assertEqual((rows["D"]["+ class"], rows["D"]["* class"]), (-1, +1))   # (−)+(−)=−, (−)×(−)=+
        self.assertEqual((rows["C"]["+ class"], rows["C"]["* class"]), (0, -1))    # (+)+(−)=0, (+)×(−)=−

    def test_one_wire_monoid_shrinks_to_six_without_a_swap(self):
        mon = chain_monoid(PAIRS_REAL)
        self.assertEqual(len(mon), 6)
        images = {(m[PLUS], m[TIMES]) for m in mon.values()}
        self.assertNotIn((TIMES, PLUS), images)          # no NOT on a single wire
        self.assertIn((PLUS, TIMES), images)             # D is the identity on one wire

    def test_not_needs_fan_in_under_real_units(self):
        net = OpNet(1, [[OpNode("C", (0,)), OpNode("U", (0,))], [OpNode("D", (0, 0, 1))]], units="real")
        self.assertEqual(net.output([PLUS])[0], TIMES)
        self.assertEqual(net.output([TIMES])[0], PLUS)
        self.assertEqual((net.size, net.depth), (3, 2))

    def test_majority_is_one_node_under_real_units(self):
        maj = OpNet(3, [[OpNode("D", (0, 1, 2))]], units="real")
        ok, _, fails = verify(maj, 3, (), lambda x, y, z: (x + y + z) >= 2)
        self.assertTrue(ok, fails)
        maj5 = OpNet(5, [[OpNode("D", (0, 1, 2, 3, 4))]], units="real")
        ok, _, fails = verify(maj5, 5, (), lambda *b: sum(b) >= 3)
        self.assertTrue(ok, fails)

    def test_constants_under_real_units(self):
        const_times = OpNet(1, [[OpNode("U", (0,))]], units="real")
        const_plus = OpNet(1, [[OpNode("U", (0,))], [OpNode("C", (0,))]], units="real")
        for op in (PLUS, TIMES):
            self.assertEqual(const_times.output([op])[0], TIMES)
            self.assertEqual(const_plus.output([op])[0], PLUS)


if __name__ == "__main__":
    unittest.main()
