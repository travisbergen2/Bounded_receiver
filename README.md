# Room X · The Bounded Receiver

**Live:** https://fractalyouniverse.org/Bounded_receiver/ (also https://travisbergen2.github.io/Bounded_receiver/)

A receiver built from the arithmetic of the Fractal Youniverse collection: nodes that are conjugate pairs of
Gaussian integers, a hidden world that turns, and three coupled blocks — **estimate**, **detect**, **commit**
(IMM Paper 18, *Observer Requirements*) — whose prior starts blank and accumulates a narrative. Seven slow
witnesses and one sharp one decide who gets believed (Paper 11, *Collective Receiver Dynamics*).

Everything on the page is recomputed in the browser and checked against `receipts/twin.json`, written by a
Python twin that uses the same PRNG (mulberry32, bit-identical) and the same operation order. The page shows
its self-check count in the masthead (198 checks at build).

## Files

| file | what |
|---|---|
| `index.html` | the room — self-contained (core and twin data inlined); one external request (Google Fonts); no analytics |
| `core.js` | the computational core: PRNG, Gaussian-integer nodes, lens enumeration, the channel world, the uniform and heterogeneous receivers with the narrative prior, the tribe |
| `receipts/twin.py` → `receipts/twin.json` | the Python twin and its reference numbers |
| `receipts/node_check.js` | runs `core.js` headless under Node against the twin: decisions and counters exact, floats to 1e-9 |
| `receipts/index.template.html`, `receipts/build_page.py` | the page is the template with `core.js` and `twin.json` injected |
| `receipts/node_observer/` | the design record: the six model versions the room distils (v0.1 phase world and sensors; v0.2 arithmetic circuits and the optimistic skeptic; v0.3 the operation-propagating network; v0.4 the four-valued hold; v0.5 the superposition filters; v0.6 the heterogeneous observer and the tribe), 84 unit tests, every pre-stated expectation with its verdict including the refuted ones, and the seeded census results |

Rebuild:

```
python3 receipts/twin.py > receipts/twin.json
node receipts/node_check.js          # expect: 198 passed, 0 failed
python3 receipts/build_page.py
cd receipts/node_observer && python3 -m unittest discover -s tests   # 84 tests
```

## What the room computes

1. **Nodes.** The three configurations (+i,−i), (+i,+i), (−i,−i) under + (trace) and × (norm), exactly; a pair
   explorer; the circle |z − 1| = 1 on which a pair is blind (trace = norm ⟺ |α − 1|² = 1), with its lattice
   points {0, 2, 1 ± i}.
2. **Lenses.** Exact enumeration of two-layer four-input trees: patterns cancelled to zero, distinct values,
   entropy, mutual information with the direction (over non-tie patterns) and with the parity, sign-blindness.
   The flip-parity fact: one × layer anywhere makes the output sign-blind.
3. **Three blocks, one currency.** The uniform receiver (moving average / CUSUM / margin) against the
   heterogeneous one (exact posterior / side-switch / Wald band) on the same seeded world; the blank prior AR
   accumulating under an honest (κ = 0) or self-serving (κ → 1) attribution filter; an optional bursty liar.
4. **The tribe.** Seven slow witnesses and one sharp, judged by consensus-relative trust rules (one strike,
   threshold 4, threshold 8): who is burned, when, and what the group loses (nothing measurable).

## What it does not claim

Eight-bit toys; channels are not people. Nothing here bears on the Riemann Hypothesis, on any market, or on
any brain; nothing is an edge; nothing is registered as evidence for any claim in the papers. Exact where it
says exact, descriptive elsewhere. Room X's first version (a third-party trading scaffold run against
Paper 18) was withdrawn by the author on 2026-09-26; this room replaces it.

Built 2026-09-26 by Travis Bergen with the Riemann agent. Instruments, not proofs.
