import unittest

from node_observer.layered import PLUS, TIMES
from node_observer.opnet import (
    OpNet, OpNode, SILENT, response, response_table, wire_map, chain_monoid, describe_map,
    GATES, gate_threshold, gate_parity, verify,
)


class TestResponses(unittest.TestCase):
    def test_table_is_derived_from_the_arithmetic(self):
        self.assertEqual((response("C", PLUS), response("C", TIMES)), (0, +1))     # 0, 1
        self.assertEqual((response("U", PLUS), response("U", TIMES)), (+1, -1))    # 2i, -1
        self.assertEqual((response("D", PLUS), response("D", TIMES)), (-1, -1))    # -2i, -1
        rows = {r["type"]: r for r in response_table()}
        self.assertEqual(rows["C"]["+ value"], "0")
        self.assertEqual(rows["U"]["+ value"], "2i")
        self.assertEqual(rows["D"]["+ value"], "-2i")
        self.assertEqual(rows["U"]["* value"], "-1")

    def test_wire_maps(self):
        self.assertEqual(wire_map("C"), {SILENT: SILENT, PLUS: SILENT, TIMES: TIMES})   # filter
        self.assertEqual(wire_map("U"), {SILENT: SILENT, PLUS: TIMES, TIMES: PLUS})     # NOT
        self.assertEqual(wire_map("D"), {SILENT: SILENT, PLUS: PLUS, TIMES: PLUS})      # reset

    def test_chain_monoid_is_all_nine_wire_maps(self):
        mon = chain_monoid()
        self.assertEqual(len(mon), 9)
        self.assertTrue(all(len(w) <= 3 for w in mon))
        images = {(m[PLUS], m[TIMES]) for m in mon.values()}
        self.assertEqual(len(images), 9)
        # chains agree with the maps
        for word, m in mon.items():
            net = OpNet.chain(list(reversed(word))) if word else OpNet.identity(1)
            for op in (PLUS, TIMES):
                self.assertEqual(net.output([op])[0], m[op], f"{word} on {op}")


class TestNetwork(unittest.TestCase):
    def test_first_layer_plus_gives_travis_three_outcomes(self):
        net = OpNet(1, [[OpNode("C", (0,)), OpNode("U", (0,)), OpNode("D", (0,))]])
        self.assertEqual(net.output([PLUS]), [SILENT, TIMES, PLUS])   # 0 → silent, up → ×, down → +

    def test_fan_in_sums_sign_classes(self):
        net = OpNet(3, [[OpNode("U", (0, 1, 2))]])
        self.assertEqual(net.output([PLUS, PLUS, TIMES]), [TIMES])    # +1 +1 −1 = +1
        self.assertEqual(net.output([PLUS, TIMES, SILENT]), [SILENT])  # tie
        self.assertEqual(net.output([TIMES, TIMES, PLUS]), [PLUS])
        self.assertEqual(net.output([SILENT, SILENT, SILENT]), [SILENT])

    def test_pass_routing_forwards_a_common_op_on_tie(self):
        net = OpNet(2, [[OpNode("C", (0, 1))]], route="pass")
        self.assertEqual(net.output([PLUS, PLUS]), [PLUS])            # C gives 0 to both; pass forwards +
        strict = OpNet(2, [[OpNode("C", (0, 1))]])
        self.assertEqual(strict.output([PLUS, PLUS]), [SILENT])


class TestGates(unittest.TestCase):
    def test_all_named_gates_verify(self):
        for name, build in GATES.items():
            net, biases, fn = build()
            ok, patterns, fails = verify(net, net.n - len(biases), biases, fn)
            self.assertTrue(ok, f"{name}: {fails[:3]}")

    def test_xor_costs(self):
        net, biases, _ = GATES["XOR"]()
        self.assertEqual((net.depth, net.size, len(biases)), (5, 13, 2))

    def test_thresholds(self):
        for n, k in ((5, 3), (7, 1), (7, 7), (9, 6), (6, 4)):
            net, biases, fn = gate_threshold(n, k)
            ok, _, fails = verify(net, n, biases, fn)
            self.assertTrue(ok, f"THR {n},{k}: {fails[:3]}")
            self.assertEqual((net.size, net.depth), (2, 2))

    def test_parity_trees(self):
        for n in (2, 4, 8):
            net, biases, fn = gate_parity(n)
            ok, patterns, fails = verify(net, n, biases, fn)
            self.assertTrue(ok, f"PARITY {n}: {fails[:3]}")
            self.assertEqual(patterns, 2 ** n)
            self.assertEqual(net.depth, 5 * (n.bit_length() - 1))


if __name__ == "__main__":
    unittest.main()
