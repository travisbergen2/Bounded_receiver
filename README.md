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

## Parameter domains (validated; out-of-domain inputs are refused, never coerced)

| parameter | domain | why |
|---|---|---|
| channels `n` | integer ≥ 1 | no channels → no observable majority |
| noise `eps` (world) | [0, 1] | a noiseless world is legitimate for the uniform receiver and the tribe |
| noise `eps` (heterogeneous receiver) | (0, 1) strictly | a deterministic channel has no finite likelihood ratio (1−ε)/ε |
| switch rate `pSwitch` (heterogeneous receiver) | (0, 1) strictly | the prior-mixing step contracts the odds into [p/(1−p), (1−p)/p] only then |
| Wald `alpha` | (0, 1) strictly | the band (1−α)/α must be finite and > 1 |
| `TI`, `FT`, `tau` | > 0 | 1/TI, the alarm level and the prior's softness must be finite |
| `UE` | (0, 1] | a learning rate |
| `steps`, `seed` | integer ≥ 0 | |
| tribe `rule` | exactly `"strike"` or `"cusum"` | a typo must not silently change the experiment |
| tribe `nSlow` | integer ≥ 1 | no silent default for 0 (the JavaScript `\|\|` pitfall was removed) |

## Validation and its limits

`node_check.js` proves that the browser core and the Python twin agree on the frozen cases (PRNG, node
tables, lenses, six observer runs including a 60,000-step small-noise run and a tie-heavy world, four
tribes) and that they reject the same seventeen out-of-domain parameter sets; it also checks the
odds-mixing bound. It does **not** prove the formulation is well defined outside the guarded domain — such
inputs are refused. Hit rate is `hits / (hits + misses)` — bets that were placed and met a non-tie outcome;
abstaining raises it, and the score (which gives abstention 0) does not. The tie-heavy boundary world (two
channels at ε = 0.5) exposed a genuine bug on 2026-09-27: the earlier denominator counted bets voided by a tie,
understating hit rates by the tie fraction (≈ 2% at eight channels; the fair coin read 0.23 in the tie world).
Fixed in the core, the twin and the design record; the correction is on display in both READMEs. Alarm
attribution is a stated heuristic (each alarm credited to the most recent switch since the previous alarm;
otherwise spurious); the median is the conventional one. The heterogeneous receiver is kept in **odds**
rather than log-odds so the two implementations agree bit for bit — its prior-mixing step is rational and
contracting, which bounds the odds for any 0 < pSwitch < 1 (receipt in `twin.json`, `boundary.mixing`).

## Outside review, 2026-09-27

At the author's request the AI agent Manus reviewed the room and the repository — "our research literally
says fool me once is not the way to go, so here is its second chance." It reproduced the 198/198 check and
the 84 design-record tests, and reported twelve findings. Verdicts: one genuine page wording error (the tribe
line "6 × 0.75 = 4.5 contradictions" — the code adds 1 per contradiction and drifts 0.25 down every step; the
net per contradicting step is 0.75, so the number was right and the words were wrong); valid parameter-domain
gaps (endpoint ε and α, TI = 0, τ = 0, negative or fractional steps, an unchecked tribe rule string, and
`nSlow || 7` rewriting 0 to 7); the conditional hit-rate denominator, the attribution heuristic and the
even-count median needing labels; the "exact posterior" needing its model assumptions displayed; and the
observation that the self-check is narrow (frozen cases). One recommendation was not adopted as stated —
moving the receiver to log-odds internally — because the odds form is what makes the twins bit-identical and
the mixing bound makes it safe inside the validated domain; the reason is now documented and receipted. All
other findings were folded in the same day. The test-command note (run from `receipts/node_observer/`, or set
`PYTHONPATH`) was also its catch.

## What it does not claim

Eight-bit toys; channels are not people. Nothing here bears on the Riemann Hypothesis, on any market, or on
any brain; nothing is an edge; nothing is registered as evidence for any claim in the papers. Exact where it
says exact, descriptive elsewhere. Room X's first version (a third-party trading scaffold run against
Paper 18) was withdrawn by the author on 2026-09-26; this room replaces it.

Built 2026-09-26 by Travis Bergen with the Riemann agent. Instruments, not proofs.
