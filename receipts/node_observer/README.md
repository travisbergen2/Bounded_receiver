# node_observer — quadratic-pair nodes, a hidden turning world, and a bounded observer whose prior starts blank

**Status: exploratory design-stage toy, v0.1 (2026-09-25). Unregistered. Descriptive numbers only.**
Nothing here bears on the Riemann Hypothesis, on any market, or on any brain. The nodes borrow the
arithmetic objects of Rooms VII–VIII (Gaussian and Eisenstein conjugate pairs, trace and norm, the
circle |z − 1| = 1); the observer borrows the three coupled blocks of IMM Paper 18 (T-OR-1) and the
five RPCS-1 primitives as their parameterisation. The intent is an instrument you can turn over in
your hands, not a claim.

Pure Python 3, standard library only. Exact integer arithmetic for the nodes; one seeded PRNG for
everything random; every run is reproducible from `(params, seed)`.

## 1. The idea, in Travis's words and then in the model's

> a network of nodes … (+i, −i) representing neurons with different patterns of (+, ×) … signals
> from another higher-dimensional system it is embedded in … for the (+i, −i) to be combined into 1
> or cancelled out to 0 to see what sequence or binary comes out … and the 3 observer requirements,
> 3 coupled parts: estimate (SG, UE) and detect (FT, TI) receive and build a model for commit (AR),
> which starts blank and accumulates a narrative.

| Travis's phrase | Model object | Code |
|---|---|---|
| the higher-dimensional system the nodes are embedded in | a hidden phase on U_n (the n-th roots of unity in **C**; n = 4 gives the Gaussian units {1, i, −1, −i}, n = 6 the Eisenstein units), turning one notch per step in a hidden regime direction, with noise and regime switches | `World`, `WorldParams` |
| a node = a quadratic pair, default (+i, −i) | `QuadraticPair(a, b, ring)`: the conjugate pair (α, ᾱ) of a Gaussian or Eisenstein integer, with exact integer `trace` and `norm` | `QuadraticPair` |
| "patterns of (+, ×)" | the world's phase, seen from the node's receptive axis, selects the operation: front half-turn → `*` (combine: α·ᾱ = norm), back half-turn → `+` (cancel: α + ᾱ = trace). For (+i, −i): 1 or 0. | `Network.op`, `QuadraticPair.emit` |
| "combined into 1 or cancelled out to 0 … what binary comes out" | the network's emitted symbols; for the default pair a binary sequence; `readout_bit` of node 0 is what COMMIT predicts | `Network.emit`, `Network.bits` |
| X4 / (1, i, −i, −1) / ω / z | `ring='gauss'` with n = 4 (χ₄'s world), `ring='eisen'` with n = 6 (ω's world); any integer pair (a, b) is allowed | `PAIR_I`, `PAIR_W`, `QuadraticPair` |
| ESTIMATE (SG, UE) | `E ← (1 − UE)·E + UE·SG·e`, e ∈ {−1, 0, +1} the decoded direction evidence of the last step | `Observer.estimate` |
| DETECT (FT, TI) | a CUSUM on *contradiction of the current model*: `S ← max(0, S + (−e·sign E) − 1/TI)`; alarm when `S ≥ FT` → E reset to 0, a seam of `seam_len` steps opens, UE is multiplied by `reset_boost` while it is open | `Observer.detect` |
| COMMIT (AR), starts blank | predicts node 0's next bit. Confident when |E| ≥ `margin`. Otherwise *ambiguous*: bets with probability sigmoid(AR/τ) on sign E, else abstains. AR₀ = 0. | `Observer.commit` |
| "accumulate a narrative" | after every ambiguity the prior resolved, the decision is scored right/wrong (bet-and-hit or abstain-and-would-have-missed = right) and entered into a ledger through an **attribution filter**: right decisions recorded with probability (1 + κ)/2, wrong ones with (1 − κ)/2. AR is a leaky mean of the recorded outcomes' verdict on betting (+1 would have hit, −1 would have missed). | `Observer.settle` |
| honest accumulation vs self-promoting ego | κ = 0: the record is an unbiased sample, AR converges to the **true edge** of betting under ambiguity, narrative accuracy ≈ true accuracy. κ = 1: only decisions that went right are recorded — a bet-hit says "bet more", a good abstention says "abstain more" — so AR runs to +1 (the over-better) or −1 (the never-wrong), and the self-reported accuracy is 100% either way. | `ObserverParams.kappa` |

The four couplings that make it *three coupled blocks* rather than three knobs:
ESTIMATE → COMMIT (the sign and size of E), ESTIMATE → DETECT (the model whose contradiction is
integrated), DETECT → ESTIMATE (reset and UE boost), DETECT → COMMIT (the seam: forced abstention
after an alarm — "asks are seam events"). Optional COMMIT → DETECT: `ego_blinds_detect=True` sets
the effective threshold to FT·(1 + |AR|), an observer whose narrative confidence raises the bar for
admitting the world changed.

## 2. Two exact facts about the nodes (theorem grade, one line each)

**Blind nodes are the circle.** A pair is *blind* when its two operations give the same integer,
trace(α) = norm(α). Since |α − 1|² = norm − trace + 1, that holds exactly when |α − 1| = 1 — the
circle that Room VII identifies as the reciprocal image of the critical line. Its lattice points
are 1 + (the ring's roots of unity): {0, 2, 1 ± i} in **Z**[i], six points in **Z**[ω]. The census
tables in `results/census.json` list trace, norm, gap and |α − 1|² for every small pair; the gap
|trace − norm| = ||α − 1|² − 1| measures how far a node is from blindness. (+i, −i) has gap 1;
(ω, ω̄) gap 2; (1 + 2i) gap 3. Tests: `test_blind_iff_on_circle`,
`test_blind_lattice_points_are_one_plus_roots_of_unity`.

**One node cannot see the arrow.** A single half-turn readout of a rotation emits 1100… forward and
1001… backward — cyclic shifts of one another — so one node, however informative its pair, is
direction-blind; two nodes in quadrature emit a Gray code (10, 11, 01, 00) and decode the phase
exactly. Novelty: KNOWN (this is quadrature encoding, the rotary-encoder principle). In the model
the one-node observer never bets. Tests: `test_one_node_is_direction_blind`,
`test_one_node_arm_never_bets`.

## 3. How to run

```
cd node_observer
python3 -m unittest discover -s tests -v        # 19 tests
python3 run_census.py --seeds 30 --steps 20000  # ~2 min; writes results/census.json
```

Single run from Python:

```python
from node_observer import RunParams, ObserverParams, WorldParams, run
r = run(RunParams(steps=20000, observer=ObserverParams(kappa=0.9)), seed=0)
print(r["score_per_1000"], r["AR_final"], r["narrative_rate"], r["true_rate"], r["ego_gap"])
```

Default constants (every one is a labelled assumption, chosen for legibility, not fitted):
world n = 4, ε = 0.15 (probability a step is uniform noise), p_switch = 0.005; network = 2 nodes
(+i, −i) on axes 0 and 1 (quadrature); observer TI = 8, SG = 1, FT = 4, UE = 0.10, margin = 0.30,
η = 0.02, τ = 0.15, seam_len = 4, reset_boost = 3, AR₀ = 0; 20,000 steps; seeds 0–29.

## 4. What the census measured (medians over 30 seeds; see `results/census_stdout.txt` for the full table)

Scoring: +1 per correct predicted bit, −1 per wrong one, 0 for an abstention. "True rate" = fraction
of ambiguity-resolved decisions that were right; "narrative rate" = the block's own recorded
accuracy; ego gap = narrative − true.

Seeds 0–29, 20,000 steps each, medians (regenerate with `python3 summarize_results.py`):

| arm | score/1k | hit rate | ambiguous frac | bet-when-ambiguous | alarms | delay | missed switches | AR final | AR sign split (+/0/−) | narrative | true | ego gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| honest | 792.3 | 0.907 | 0.009 | 0.155 | 93 | 4 | 4 | −0.479 | 0/0/30 | 0.700 | 0.699 | −0.001 |
| ego_0.5 | 792.6 | 0.907 | 0.009 | 0.116 | 93 | 4 | 4 | −0.722 | 0/0/30 | 0.885 | 0.723 | 0.163 |
| ego_0.9 | 792.6 | 0.907 | 0.009 | 0.098 | 93 | 4 | 4 | −0.891 | 0/0/30 | 0.983 | 0.732 | 0.254 |
| ego_1.0 | 792.6 | 0.907 | 0.009 | 0.091 | 93 | 4 | 4 | −0.917 | 0/0/30 | 1.000 | 0.723 | 0.277 |
| ego_0.9_blinds_detect | 788.5 | 0.907 | 0.029 | 0.042 | 21 | 6 | 75 | −0.831 | 0/0/30 | 0.930 | 0.441 | 0.491 |
| honest_no_detect | 791.5 | 0.900 | 0.034 | 0.702 | 0 | — | 99 | +0.213 | 29/1/0 | 0.534 | 0.528 | 0.003 |
| honest_1node | 0.0 | — | 0 | — | 0 | — | 96 | 0 | 0/30/0 | — | — | — |
| honest_4nodes | 792.3 | 0.907 | 0.009 | 0.155 | 93 | 4 | 4 | −0.479 | 0/0/30 | 0.700 | 0.699 | −0.001 |
| honest_blind_pair (1 ± i) | 0.0 | — | 0 | — | 0 | — | 96 | 0 | 0/30/0 | — | — | — |
| honest_eisenstein (n = 6, 3 × ω) | 812.8 | 0.913 | 0.006 | 0.212 | 90 | 5 | 4 | −0.440 | 0/0/30 | 0.700 | 0.693 | −0.004 |
| ego_0.9_eisenstein | 812.9 | 0.913 | 0.006 | 0.123 | 90 | 5 | 4 | −0.776 | 0/0/30 | 0.987 | 0.741 | 0.248 |
| honest_high_noise (ε = 0.40) | 551.0 | 0.781 | 0.068 | 0.835 | 52 | 6 | 31 | +0.307 | 29/1/0 | 0.595 | 0.597 | −0.003 |
| ego_0.9_high_noise | 553.5 | 0.780 | 0.068 | 0.981 | 52 | 6 | 31 | +0.951 | 28/0/2 | 0.972 | 0.636 | 0.334 |

Switches per run: median 100 (99 in the six-phase and high-noise worlds). "missed" counts switches with
no alarm within 200 steps. The prior from blank, seed 0, sampled at steps 200 / 1,000 / 2,000 / 5,000 /
10,000 / 20,000: honest −0.06, −0.11, −0.13, −0.30, −0.45, −0.59; ego 0.9 −0.02, −0.08, −0.11, −0.35,
−0.57, −0.92; high-noise honest −0.00, +0.11, +0.02, +0.46, +0.35, +0.28 (wanders about the edge);
high-noise ego 0.9 +0.02, +0.19, +0.40, +0.96, +0.95, +0.92 (locks). Node 0's first 32 output bits,
seed 0: `11100110011000011001101100110011`.

Findings worth stating (all descriptive, this world, these constants):

1. **The honest prior's sign is a fact about the world, not about the observer.** With κ = 0 the
   blank prior converges to the true edge of betting under ambiguity. In the default world that edge
   is *negative* (about −0.5): ambiguity there is almost always the signature of a regime change in
   progress — the estimate is small because it is decaying from the wrong sign — so the honest
   observer learns to abstain when unsure (AR −0.48, bets on 15% of ambiguities). In the high-noise
   world (ε = 0.40) ambiguity is mostly noise, the stale sign is still usually right, and the same rule
   learns a *positive* prior (+0.31, bets on 84%). With the detector switched off, ambiguity becomes a
   coin (true rate 0.53) and the prior sits at +0.21. Same code, three different "personalities" —
   each the correct hedge for its world. The detector's own contribution to the score is small
   (792.3 vs 791.5 per 1,000); its contribution to *meaning* is large — it is what makes ambiguity
   diagnostic of change (0.9% of steps ambiguous with it, 3.4% without).
2. **The ego inflates the ledger without costing the score.** At κ = 0.5 / 0.9 / 1.0 the
   self-reported accuracy rises to 0.885 / 0.983 / 1.000 against a true decision rate of 0.72–0.73,
   while the score per 1,000 steps is unchanged (792.6 vs 792.3), because ambiguity-resolved decisions
   are 0.9% of all steps. The narrative and the record part company silently.
3. **The ego is bistable exactly where the honest signal is weak.** In the default world all 30 ego
   seeds run to the never-wrong attractor (AR → −0.9, the honest sign amplified). In the high-noise
   world the honest prior is weakly positive (+0.31) and the κ = 0.9 ego splits 28 : 2 — most seeds run
   to the over-better (AR → +0.95, betting on 98% of ambiguities), two to the never-wrong (AR −0.82)
   — from a blank start, on early luck alone. There the over-better even scores slightly more than
   the honest observer (553.5 vs 551.0): an ego aligned with a positive edge is paid for its
   overconfidence, and its ledger (0.972 vs 0.636 true) still lies.
4. **When the ego touches perception it goes blind.** With `ego_blinds_detect`, |AR| ≈ 0.83 raises the
   alarm threshold from 4 to ≈ 7.3; alarms fall from 93 (one per switch) to 21 per run, 75 of 100
   regime switches are missed, ambiguity triples (2.9% of steps), the true decision rate under
   ambiguity falls to 0.441 while the narrative stays 0.930 (gap 0.49). The score falls only 3.8 per
   1,000, because ESTIMATE recovers on its own in ~10 steps — in this world the detector's job is speed
   and the seam, not survival.
5. **Node budget.** One node, or two blind nodes (1 ± i), never bet (score 0). Two (+i, −i) nodes in
   quadrature suffice; four add nothing (identical results — the decode is already exact). Three
   (ω, ω̄) nodes on the six-phase world emit a thermometer code and add a *structural* channel: at
   two of the six phases both directions predict the same bit, so the observer commits without needing
   its estimate (hit rate 0.913 vs 0.907).

## 5. Fences

- Toy. Six-bit-scale automata are not neurons; the "ego" is an attribution filter with one parameter,
  not a theory of selves. The Paper 18 correspondence is a parameterisation choice, and the RPCS-1
  primitive labels are the model's dictionary, not measurements of anyone.
- Descriptive, unregistered. No expectation was frozen before these runs; nothing here is evidence
  for or against any registered IMM claim. If this becomes an E-series instrument, a protocol with
  pre-stated verdicts freezes first, on Travis's go, and these runs count as the design-stage peek.
- The world is designed so that direction is decodable and the ambiguity structure is legible. Change
  the world and the honest prior's sign changes with it (finding 1) — that is the point, and also the
  reason none of the numbers transfer anywhere.
- The circle fact (§2) is exact and is the same identity as Room VII's trace = norm ⟺ Re(1/z) = ½;
  it says nothing about the zeros of ζ.

## 6. Novelty (standing duty)

KNOWN-ADJACENT as an assembly; every component is known: quadrature encoding (§2); CUSUM change
detection (Page 1954); exponential smoothing for the estimate; the self-serving attribution bias
(Miller & Ross 1975, from memory — pin before any deposit) as the ego mechanism; McCulloch–Pitts
binary units for "neurons that emit 0 or 1". The specific object — conjugate-pair nodes whose two
operations are trace and norm, read by Paper 18's three coupled blocks with an attribution-filtered
commit prior — is NOT-FOUND-BY-ME, which is weak evidence of nothing; the blind-node circle is a
one-line corollary of a standard identity.

## 7. v0.2 — the signal passes *through*, and the observer is built like Travis

Travis's second pass (2026-09-25, 21:00): *"build it like me — an optimistic skeptic, fool me once …
a network of (+i, −i) that the signal passes through … each node could be (+i, −i), (+i, +i), (−i, −i)
in different configurations and send + and × through layers like a neural network."* Module:
`node_observer/layered.py`; tests: `tests/test_layered.py`; census: `run_census2.py`.

### 7.1 The network: an arithmetic circuit over the Gaussian integers

Reading adopted (redirectable): the hidden world injects a sign on each of n input channels
(x_j = +i or −i); every node is typed `+` or `*` and combines two parents from the previous layer;
values propagate layer by layer, exactly, as Gaussian integers; the observer reads one layer's node
values as its **witnesses**. The three configurations a two-input node can see are Travis's:

| configuration | `+` (cancel/keep) | `*` (combine) |
|---|---|---|
| (+i, −i) | 0 | 1 |
| (+i, +i) | 2i | −1 |
| (−i, −i) | −2i | −1 |

So `+` keeps the world's **direction** and cancels the conjugate pair; `*` keeps **agreement**
(same → −1, different → +1) and is sign-blind, because i·i = (−i)(−i). Two exact facts follow:

- **Flip parity [T].** Under the global flip x → −x every input flips sign. A `+` node with two
  flipping parents flips; a `*` node with two flipping parents does *not* (the signs cancel); a `*`
  node with exactly one flipping parent flips. Hence a homogeneous tree is direction-visible iff it is
  all `+`: one `*` layer anywhere makes the output sign-blind, and every `+` downstream of two blind
  values stays blind. Direction survives a `*` only when exactly one of its inputs carries it.
- **The parity lens sees the tie [T].** A tie (equally many +i and −i) has an even number of −i
  inputs, so the `*` tree — which computes (−1)^{#−i} — can rule a tie in or out while never seeing
  which way the world leans. Measured below as MI(direction) > 0 with MI(sign) = 0.

Exact enumeration over all 2^n sign patterns (`lens_report`; H in bits; MI(sign) is computed over
non-tie patterns; the alternative reading — the *operation* propagates and each node holds a fixed
pair — is not built):

| lens | shape | patterns → 0 | distinct values | H(value) | MI(sign) | MI(direction incl. tie) | MI(parity) | sign-blind |
|---|---|---|---|---|---|---|---|---|
| tree4 `++` (count) | 4→2→1 | 6 | 5 | 2.031 | 1.000 | 1.579 | 1.000 | no |
| tree4 `**` (parity) | 4→2→1 | 0 | 2 | 1.000 | 0 | 0.549 | 1.000 | yes |
| tree4 `*+` (agreement count) | 4→2→1 | 8 | 3 | 1.500 | 0 | 0.704 | 1.000 | yes |
| tree4 `+*` (all-agree) | 4→2→1 | 12 | 3 | 1.061 | 0 | 0.266 | 0.311 | yes |
| tree8 `+++` | 8→4→2→1 | 70 | 9 | 2.544 | 1.000 | 1.573 | 1.000 | no |
| tree8 `***` | 8→4→2→1 | 0 | 2 | 1.000 | 0 | 0.349 | 1.000 | yes |
| tree8 `**+` | 8→4→2→1 | 128 | 3 | 1.500 | 0 | 0.353 | 1.000 | yes |
| tree8 `++*` | 8→4→2→1 | 156 | 7 | 1.795 | 0 | 0.371 | 0.414 | yes |
| tree8 `+*+` | 8→4→2→1 | 152 | 5 | 1.540 | 0 | 0.124 | 0.073 | yes |
| tree8 `*+*` | 8→4→2→1 | 192 | 3 | 1.061 | 0 | 0.122 | 0.311 | yes |
| band8 `+`, node 0 (x₀ + x₁) | 8→8 | 128 | 3 | 1.500 | 0.215 | 0.162 | 0 | no |
| band8 `*`, node 0 (x₀x₁) | 8→8 | 0 | 2 | 1.000 | 0 | 0.006 | 0 | yes |
| band8 `++`, node 0 (x₀ + 2x₁ + x₂) | 8→8→8 | 64 | 5 | 2.250 | 0.297 | 0.234 | 0 | no |
| band8 `+*`, node 0 | 8→8→8 | 192 | 2 | 0.811 | 0 | 0.018 | 0 | yes |

Reading: the all-`+` tree is the count lens (the value i·(2k − n) gives the count exactly, hence
direction and parity both); the all-`*` tree is the parity lens; `*` then `+` is the agreement-count
lens; `+` then `*` is the all-agree test. A local `+` witness (two channels) carries 0.215 bits about
the global direction per step; the observer below aggregates eight of them.

### 7.2 The observer: optimistic, skeptical, one strike

The three blocks of §1 are fed by a **trust-weighted consensus of the witnesses** (each witness's
evidence = the sign of its imaginary part; `*` witnesses therefore carry none). Optimism is the start
(AR₀ = +0.5, betting under ambiguity by default); skepticism is the checking; "fool me once" is the
consequence. Four trust modes:

| mode | rule |
|---|---|
| `trusting` | every witness weight 1 forever |
| `fool_once` | the first time a witness contradicts the trusted consensus it is burned forever |
| `skeptic` | a per-witness CUSUM on contradiction (drift 1/src_TI = 0.25, threshold src_FT = 4); when it fires the witness is burned forever — one strike, but only for *structured* fooling |
| `forgiving` | an EMA reliability weight (rate 0.05) that recovers when the witness agrees again |

Contradiction is **consensus-relative**: at a regime switch all honest witnesses flip together, the
consensus flips with them, and nobody is read as lying — "the world changed" and "this source misled
me" are separated by construction (test `test_regime_switch_is_not_read_as_lying`). A structural fact
about one-strike burning [T, test `test_fool_once_under_noise_stops_at_two_survivors`]: under any
noise it stops at exactly **two survivors**, because two witnesses can only contradict a consensus
they themselves form — an echo chamber of two is the fixed point of "fool me once".

Worlds: `noise` (8 channels, ε = 0.15, switch 0.005), `trick` (the same plus channel 3 lying in
bursts: enters a burst with probability 0.01 per step, mean length 60 — structured deception on top
of the noise), `high-noise` (ε = 0.30). The observer predicts the sign of the next step's witness
majority (the observable), so its hits are scored against what it can see.

### 7.3 What the v0.2 census measured — and which of my pre-stated expectations it beat

Six expectations were written into the thread record *before* the census ran (E1–E6 below). Seeds
0–19, 20,000 steps, medians. "regime acc." = fraction of steps the estimate's sign matched the hidden
regime (uncontaminated by the liar); the score is against the observable witness majority, which a
liar *can* contaminate (see finding 3).

| trust / world / witnesses | score/1k | hit rate | regime acc. | ambiguous frac | bet-when-ambiguous | alarms | burned (median) | seeds w/ liar burned | false burns (total, 20 seeds) | first liar burn | AR final | AR split (+/0/−) | narrative | true | gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trusting / noise / channels | 901.5 | 0.954 | 0.967 | 0.027 | 0.411 | 58 | 0 | — | 0 | — | −0.136 | 1/3/16 | 0.508 | 0.513 | 0.003 |
| fool_once / noise / channels | 900.5 | 0.954 | 0.967 | 0.025 | 0.502 | 68 | 6 | — | 120 | — | −0.147 | 0/5/15 | 0.508 | 0.509 | −0.001 |
| skeptic (src_FT 4) / noise / channels | 903.5 | 0.955 | 0.968 | 0.025 | 0.458 | 68 | 6 | — | 120 | — | −0.109 | 4/2/14 | 0.514 | 0.502 | 0.008 |
| forgiving / noise / channels | 902.5 | 0.956 | 0.967 | 0.022 | 0.300 | 70 | 0 | — | 0 | — | −0.358 | 0/0/20 | 0.564 | 0.572 | −0.005 |
| trusting / trick / channels | 890.0 | 0.937 | 0.968 | 0.037 | 0.718 | 31 | 0 | 0 | 0 | — | +0.069 | 12/6/2 | 0.531 | 0.530 | −0.003 |
| fool_once / trick / channels | 880.9 | 0.937 | 0.966 | 0.027 | 0.516 | 60 | 6 | 13 | 107 | 4 | −0.100 | 5/4/11 | 0.525 | 0.522 | −0.002 |
| skeptic (src_FT 4) / trick / channels | 885.9 | 0.939 | 0.969 | 0.025 | 0.448 | 62 | 6 | 20 | 100 | 84 | −0.093 | 0/6/14 | 0.508 | 0.510 | −0.004 |
| forgiving / trick / channels | 886.1 | 0.940 | 0.968 | 0.022 | 0.327 | 66 | 0 | — | 0 | — | −0.333 | 0/0/20 | 0.568 | 0.561 | 0.002 |
| trusting / trick / `+` band | 890.7 | 0.937 | 0.968 | 0.037 | 0.731 | 32 | 0 | 0 | 0 | — | +0.070 | 12/6/2 | 0.531 | 0.530 | −0.002 |
| skeptic (src_FT 4) / trick / `+` band | 887.8 | 0.938 | 0.969 | 0.027 | 0.512 | 50 | 2 | 20 | **0** | 259 | −0.107 | 2/4/14 | 0.510 | 0.509 | 0.004 |
| trusting / noise / `*` band | 0.0 | — | — | 0 | — | 0 | 0 | — | 0 | — | 0.500 | 20/0/0 | — | — | — |
| skeptic / noise / channels, AR₀ = 0 | 904.0 | 0.957 | 0.968 | 0.025 | 0.322 | 68 | 6 | — | 120 | — | −0.112 | 4/2/14 | 0.532 | 0.524 | 0.006 |
| skeptic / high-noise ε 0.30 / channels | 684.8 | 0.777 | 0.959 | 0.295 | 0.989 | 20 | 6 | — | 120 | — | +0.656 | 20/0/0 | 0.839 | 0.838 | 0.000 |
| fool_once / high-noise / channels | 682.3 | 0.776 | 0.959 | 0.291 | 0.986 | 20 | 6 | — | 120 | — | +0.731 | 20/0/0 | 0.836 | 0.832 | 0.001 |

Liar active on 36–40% of steps (median `lying_frac`). Original skeptic's first *false* burn: median
step 217 (noise), 210 (trick); fool_once: step 1.

**Follow-up, written after reading the table (post-hoc, labelled):** the skeptic's strike threshold
recalibrated to the noise floor (src_FT 8), and a small jury (3 channels, channel 1 the liar) where one
liar can swing the majority. `run_followup2.py`, `results/followup2.json`.

| trust / world | score/1k | hit rate | regime acc. | burned | seeds w/ liar burned | false burns (total) | first liar burn | first false burn | AR final |
|---|---|---|---|---|---|---|---|---|---|
| skeptic src_FT 8 / noise (n = 8) | 901.1 | 0.954 | 0.966 | 0 | — | 8 | — | 10,886 | −0.232 |
| skeptic src_FT 8 / trick (n = 8) | 884.5 | 0.938 | 0.966 | 1 | 20 | 4 | 92.5 | 11,501 | −0.128 |
| trusting / small-noise (n = 3) | 814.3 | 0.917 | 0.968 | 0 | — | 0 | — | — | −0.072 |
| skeptic src_FT 8 / small-noise | 814.3 | 0.917 | 0.968 | 0 | — | 0 | — | — | −0.072 |
| trusting / small-trick | 676.8 | 0.846 | 0.963 | 0 | 0 | 0 | — | — | +0.434 |
| fool_once / small-trick | 559.5 | 0.820 | **0.843** | 1 | 6 | 14 | 1.5 | 1 | +0.092 |
| skeptic src_FT 4 / small-trick | 687.0 | 0.852 | 0.969 | 1 | 20 | 0 | 83 | — | −0.108 |
| skeptic src_FT 8 / small-trick | 687.6 | 0.852 | 0.968 | 1 | 20 | 0 | 126 | — | −0.108 |
| forgiving / small-trick | 687.1 | 0.852 | 0.967 | 0 | — | 0 | — | — | +0.034 |

Verdicts on the pre-stated expectations:

- **E1** (raw fool-me-once under noise burns everything and the score collapses) — the burning half
  held (6 of 8 burned in every seed, first at step 1–2), the consequence was **refuted**: 900.5 vs
  901.5. Two surviving honest channels plus temporal integration recover the regime as well as eight
  (regime accuracy 0.967 both). Time did the aggregation the jury no longer did.
- **E2** (a calibrated skeptic burns ~0 honest witnesses and beats trusting in the liar's world) —
  **refuted twice**. At src_FT 4 it burned 6 honest witnesses in every seed (my run-length estimate
  ignored the CUSUM's reflecting barrier; the honest-source false-alarm run length at 15% noise is of
  order 10³ steps). And even recalibrated (src_FT 8: 0.4 false burns per seed, liar burned 20/20) it
  scored 884.5 against trusting's 890.0.
- **E3** (forgiving between trusting and skeptic) — not as stated: forgiving ≈ skeptic (886.1 vs
  885.9 / 884.5), both below trusting. It did identify the liar (weight 0.058) while discounting honest
  channels to 0.5–0.75.
- **E4** (a `*` band is direction-blind) — confirmed: 0 bets, score 0, AR untouched at 0.5.
- **E5** (optimism is overwritten by the honest update unless ambiguity is noise) — confirmed: AR₀ 0.5
  and 0 reach the same endpoint (−0.109 / −0.112) and score (903.5 / 904.0); at ε = 0.30 the prior
  stays optimistic (+0.66, 20/20 seeds), 29.5% of steps are ambiguous, it bets on 99% of them, and its
  ledger is honest (0.839 vs 0.838).
- **E6** (the `+` band hides the liar) — half right: detection is delayed (first burn 259 vs 84) but
  never missed (20/20), and the lens produced **zero** false burns against 100–120 on the raw channels.

What this says about "build it like me":

1. **Raw "fool me once" is the worst policy in every world tested.** It strikes on the first
   disagreement, which under 15% noise arrives at step 1 — long before any liar — and the survivors are
   whoever happened to agree early: the liar survived in 7 of 20 seeds (n = 8) and 14 of 20 (n = 3),
   where it dragged the regime accuracy to 0.843 and the score to 559.5 against 676.8 for simply
   trusting everyone. Being fooled by noise is not being fooled.
2. **"Fooled" needs a definition noise cannot satisfy.** The skeptic's strike threshold has to sit
   above the honest-source noise floor for the horizon you care about: src_FT 4 → 6 honest burns per
   seed, src_FT 8 → 0.4. Skepticism without a calibrated noise floor is fool_once with a delay.
3. **Even calibrated, permanence costs when the liar is part-time and the jury is big.** A liar active
   36% of the time is an honest witness 64% of the time; a majority of eight is barely hurt by its
   bursts; burning it forever discards its honest majority share — 884.5 vs 890.0. One strike pays where
   one liar can swing the outcome: in the jury of three the calibrated skeptic beats trusting (687.6 vs
   676.8; regime accuracy 0.968 vs 0.963). Part of the liar's damage there sits in the scoring target
   itself (the observable majority includes its lies), which no trust policy can recover — a v0.2
   limitation, hence the regime-accuracy column.
4. **Irreversibility bought nothing over forgiveness here** (forgiving ≈ calibrated skeptic in both
   juries). The case for one strike is a liar whose returns are timed to forgiveness; not built.
5. **The disposition is downstream of the world.** Optimism at the start is neither help nor harm — the
   honest update overwrites it in a few thousand steps. Where ambiguity is noise (ε = 0.30) the
   observer *stays* an optimist and its ledger stays true; where ambiguity means change it becomes a
   hedger. An "optimistic skeptic" is the stable honest disposition for a noisy-but-stable world.
6. **The lens does part of the skeptic's work.** One `+` layer before the witnesses turns a single
   channel flip into silence (a tie) and a sustained lie into contradiction: zero false burns, liar
   still caught 20/20. And any `*` on two live signals discards which-way for good — the observer has to
   be wired to the world by `+` paths (flip-parity theorem, §7.1).
7. **One-strike burning has a fixed point of exactly two survivors** (they can only contradict a
   consensus they form). Under noise that is an echo chamber; after a caught liar in a small jury it is
   what protected the honest pair (0 false burns in the small-trick skeptic arms).

Fences as in §5: toy worlds, one liar model (bursty inversion), one deception geometry; nothing
transfers, nothing is registered, nothing bears on RH or on any market.

## 8. v0.3 — the *operation* propagates (Travis, 2026-09-25, 21:31)

> propagate the operations +, × through layers of (+i,−i), (+i,+i), (−i,−i): if the first layer is +
> and it hits (+i,−i) = 0, (+i,+i) = up, (−i,−i) = down, which would × or + to the next layer, so that
> complex problems can be figured more efficiently.

Module `node_observer/opnet.py`; tests `tests/test_opnet.py` (10); report `run_opnet.py` →
`results/opnet.json`, `results/opnet_stdout.txt`. Nodes hold **fixed** pairs C = (+i,−i), U = (+i,+i),
D = (−i,−i). A node applies each live parent's operation to its own pair (exact arithmetic), reads the
result by its sign class (0 / positive / negative), sums over parents, and the sign of the sum chooses
the operation it sends on: positive → `×`, negative → `+`, zero → silent. Everything below is derived
from the arithmetic at run time, not typed in.

### 8.1 The derived table, and what one wire can do

| node | pair | `+` gives | class | sends | `×` gives | class | sends | as a wire map |
|---|---|---|---|---|---|---|---|---|
| C | (+i, −i) | 0 | 0 | silent | 1 | +1 | `×` | **filter**: kills `+`, passes `×` |
| U | (+i, +i) | 2i | +1 | `×` | −1 | −1 | `+` | **swap**: NOT |
| D | (−i, −i) | −2i | −1 | `+` | −1 | −1 | `+` | **reset**: everything becomes `+` |

**The single-wire monoid [T, by enumeration].** Composing C, U, D on one wire generates exactly the 9
possible maps {+, ×} → {+, ×, silent} (silence stays silence), each reached by a chain of at most three
nodes: id, C, D, U, CD (kill all), CU (+→×, ×→·), UC (+→·, ×→+), UD (constant ×), UCU (+→+, ×→·). A
chain of any depth is one of these nine — depth buys nothing on a wire.

### 8.1b Which table? Travis's sign table vs the imaginary units (2026-09-25, 21:55)

Travis's check: (+)+(+) = +, (+)×(+) = +, (−)+(−) = −, (−)×(−) = +, (+)+(−) = 0, (+)×(−) = −. That is
the sign arithmetic of the **real** units {+1, −1}. The pairs (+i, −i), (+i, +i), (−i, −i) are the
**imaginary** units, and i² = −1 negates every `×` entry: the `+` rows agree exactly, the `×` rows are
opposite (test `test_plus_rows_agree_and_times_rows_are_negated`). Both are available:
`OpNet(..., units="imaginary")` (default) or `units="real"`. What the flip changes, exactly:

| | imaginary units (±i) | real units (±1, Travis's table) |
|---|---|---|
| U = (+,+) on one wire | swap: NOT | constant `×` |
| D = (−,−) on one wire | reset to `+` | identity (pass-through) |
| C = (+,−) on one wire | kills `+`, passes `×` | kills `+`, maps `×` → `+` |
| one-wire monoid | all 9 maps | 6 maps, **no NOT** |
| NOT | 1 node | 3 nodes, depth 2: D(2·C(x), U(x)) |
| majority of n | 2 nodes (U inverts, U again) | **1 node**: D over the inputs (non-inverting) |
| constants | D = `+`, U(D) = `×` | U = `×`, C(U) = `+` |
| computational class | unit-weight threshold logic | unit-weight threshold logic |

So the choice of units moves the free operation: with ±i, inversion is free and majority costs two;
with ±1, majority is free and inversion costs three. Neither table changes what the network can
compute — only where the wiring pays. The gate library of §8.2 is written for the imaginary table.

### 8.2 With fan-in: unit-weight threshold logic with a free NOT

A U node's activation is #(`+` parents) − #(`×` parents): an **inverting majority** gate with unit
weights; U alone on a wire is NOT; D on any live wire is the constant `+`, U(D) the constant `×`; a
parent listed twice is a weight of 2. Encoding `×` = TRUE, `+` = FALSE, every gate below was verified on
all input patterns:

| gate | inputs | bias wires | nodes | depth | patterns checked |
|---|---|---|---|---|---|
| NOT | 1 | 0 | 1 | 1 | 2 |
| NOR = U(x, y, `×`) · NAND = U(x, y, `+`) | 2 | 1 | 1 | 1 | 4 |
| OR · AND (the above, re-inverted) | 2 | 1 | 2 | 2 | 4 |
| MAJ3 = U(U(x, y, z)) | 3 | 0 | 2 | 2 | 8 |
| any threshold "at least k of n" | n | \|2k − 1 − n\| | 2 | 2 | 2ⁿ (n = 5, 7, 8, 9) |
| XOR | 2 | 2 | 13 | 5 | 4 |
| PARITY_n (XOR tree) | 2 / 4 / 8 / 16 | 2 | 23 / 59 / 121 / 235 | 5 / 10 / 15 / 20 | 4 / 16 / 256 / **65,536** |

Why XOR is not one node: XOR = OR ∧ ¬AND needs *opposite* couplings on two signals, and a U node couples
every live input with the same sign — the Minsky–Papert perceptron fact, re-derived in this substrate. C
(× → ×, + → silent, no inversion) supplies the second coupling and D the constants; hence depth 5. The
parity tree is not depth-optimal (standard depth-2/3 majority constructions exist — from memory; pin
before any deposit); it is a verified existence proof, not a lower bound.

### 8.3 Verdict on "complex problems figured more efficiently"

**No — and the reason is exact.** A node's computational content is its input→output table. Here the
table is 2 operations × 3 sign classes; the ±i arithmetic that produced it adds nothing the table does
not already say, so the substrate is *precisely* unit-weight threshold logic with a free inverter — as
powerful as McCulloch–Pitts / majority networks, no more (constant depth, polynomial size = the
threshold-circuit class; standard, from memory). Majority in two nodes against Θ(n) AND/OR gates is the
known advantage of threshold gates over Boolean gates; it is real, not new, and not due to the ±i. The
intuition gap: complex numbers carry more than a bit, but this node *reads* them through a
three-valued sign readout, so all but one trit is discarded at every node. Power lives in readout
resolution × fan-in × depth, not in the number system. The route to genuine gain is the v0.2 reading —
keep the **values** and read them at higher resolution (the all-`+` tree carries 2.54 bits per node vs
1.58 for any trit; §7.1) — which is the known territory of arithmetic circuits, not a shortcut past it.
What is genuinely nice: three pairs give exactly the three primitive wire behaviours {filter, NOT,
reset}, a minimal complete basis; and the two readings converge (§8.4).

### 8.4 The op-network as the observer's lens (pre-stated L1–L3; seeds 0–19, 20,000 steps, medians)

World signs encoded as operations (up → `×`, down → `+`), passed through a one-layer op-network, decoded
as direction evidence (`×` → +1, `+` → −1, silent → 0) for the v0.2 observer.

| lens | trust / world | score/1k | hit rate | regime acc. | void | alarms | liar burned (seeds) | false burns | first liar burn |
|---|---|---|---|---|---|---|---|---|---|
| raw channels | trusting / noise | 901.5 | 0.954 | 0.967 | 0.019 | 58 | — | 0 | — |
| U, fan-in 1 (NOT) | trusting / noise | 901.6 | 0.954 | **0.033** | 0.019 | 58 | — | 0 | — |
| C, fan-in 1 (filter) | trusting / noise | 775.8 | 0.891 | 0.522 | 0.129 | **0** | — | 0 | — |
| D, fan-in 1 (reset) | trusting / noise | **999.9** | 0.99995 | 0.510 | 0 | 0 | — | 0 | — |
| raw channels | skeptic 4 / trick | 885.9 | 0.939 | 0.969 | 0.032 | 62 | 20 | 100 | 84 |
| U, fan-in 2 | skeptic 4 / trick | 887.9 | 0.938 | 0.032 | 0.032 | 50 | 20 | **0** | **259** |
| C, fan-in 2 | skeptic 4 / trick | 837.8 | 0.923 | 0.509 | 0.093 | 0 | 0 | 0 | — |

- **L1 confirmed, with a specimen.** The U lens is a relabeling: identical score to the raw channels
  (901.6 vs 901.5; the only divergence is the first step's coin) — and a regime accuracy of 0.033: the
  observer is perfectly right about what it sees and perfectly wrong about which way the world turns,
  and cannot tell from inside. The flip-parity theorem of §7.1 in the operation reading: direction is
  recoverable only up to the parity of the U's on the path.
- **L2 confirmed exactly.** Through the D lens every witness says `+`; the observer's estimate locks at
  −1, it predicts the constant it is shown, hits 19,999 of 20,000 steps, never alarms, and knows the
  world's regime at chance (0.510). *A perfect score on a lens that shows nothing* — the ego chapter's
  point made by wiring alone: a record is about the lens, not the world, unless the lens is checked.
  The C lens is the half-way case: it passes one operation only, so the observer becomes a believer in
  `×` (AR → +1.0, estimate → +0.83), is right about the lens 89% of the time, never detects a change
  (0 alarms: no evidence can ever contradict it), and knows nothing (0.522).
- **L3 confirmed exactly.** The fan-in-2 U band (agreement → operation, disagreement → silence) burns
  the liar in 20/20 seeds at step 259 with zero false burns — the same numbers as v0.2's `+` band
  (259, 20/20, 0). The value reading and the operation reading are the same lens here, up to a sign.

## 8.5 v0.4 sketch — holding a contradiction instead of cancelling it (Travis, 2026-09-26)

> a way to get (1, −1, 0, superposition) as outputs so it can hold contradictions like I do.

Module `node_observer/fourval.py`, tests `tests/test_fourval.py` (7). The one change: keep **two channels**
per value. A value is a signed multiset of votes (n₊, n₋); from it, **direction** d = n₊ − n₋ (what the
old node kept) and **presence** p = n₊ + n₋ (what the old node threw away). The four outputs are the sign
patterns of (d, p): `0` = (0, 0) nothing arrived; `+` = (+, +); `−` = (−, +); **`S` = (0, +)** — something
is there and it points nowhere: a held contradiction. With counts the contest is graded: (3, 1) reads `S`
leaning plus; the old collapse says `+` and forgets the 1.

Operations on Travis's sign table (real units): **HOLD** x ⊔ y = (x₊ + y₊, x₋ + y₋) — his `+`, except
(+) ⊔ (−) = `S` instead of 0; **COMPARE** x ⊗ y = (x₊y₊ + x₋y₋, x₊y₋ + x₋y₊) — his `×` row exactly
((+)⊗(+) = +, (−)⊗(−) = +, (+)⊗(−) = −). This is the group ring Z[Z/2] (the split-complex integers,
j² = +1), and in (presence, direction) coordinates HOLD adds and COMPARE multiplies **componentwise**:
p(x ⊗ y) = p(x)·p(y), d(x ⊗ y) = d(x)·d(y) (test over 500 random pairs). The channels never mix. The old
v0.3 node is the projection onto d; the superposition values are exactly its kernel.

|  HOLD | 0 | + | − | S |   | COMPARE | 0 | + | − | S |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 0 | + | − | S |   | **0** | 0 | 0 | 0 | 0 |
| **+** | + | + | S | S |   | **+** | 0 | + | − | S |
| **−** | − | S | − | S |   | **−** | 0 | − | + | S |
| **S** | S | S | S | S |   | **S** | 0 | S | S | S |

Why this rather than amplitudes: the set/count version is the *possibilistic* superposition the E-1B
toys and v0.1's candidate-set sensor already use, and it imports no weighting rule; a quantum-style
amplitude version would smuggle in a Born-type weighting that E-BORN-1 found is not forced by bounded
records. Names: the four values are Belnap–Dunn's FOUR (1976–77; bilattices: Ginsberg 1988, Fitting —
from memory, pin before any deposit); HOLD is the knowledge-order join, graded; COMPARE is the set
product, not a Belnap truth connective. In the imaginary-unit reading the same two channels are the
(trace, norm) of the conjugate pair — (+i, −i) has trace 0 and norm 1, "there but pointing nowhere" —
so `S` was in Room VII's coordinates all along; the split-complex form is the one whose channels are
independent. Where `S` ≠ `0` matters: wherever silence can arise on its own — dead inputs, the C filter
of §8 — a contested signal and an absent one are then different messages; in the v0.2 channel world
they were not (channels never fall silent), which is why v0.2's `+` band saw disagreement as 0 without
loss.

## 8.6 v0.5 — "an emergent superposition property that can scale until it collapses to the answer" (Travis, 2026-09-26)

Three known forms, each with a fence. Module `node_observer/superpose.py`, tests `tests/test_superpose.py`
(6), census `run_superpose.py` → `results/superpose.json`. Expectations P1–P5 were written into the thread
record before the runs.

**(a) Possibilistic — `HardFilter`.** The observer holds the *set* of joint hypotheses (phase, regime)
still consistent with everything seen: it starts full (the superposition), each observation intersects it,
the dynamics advance it. Collapse = a singleton. The **empty set** = every hypothesis refuted = the world
changed: a change detector with no threshold and no parameters. Exact integer sets. KNOWN: constraint
propagation (Waltz 1975, Mackworth 1977), unit propagation; in-corpus: v0.1's candidate-set sensor, the
E-1B toys' possibilistic separation.

**(b) Probabilistic — `SoftFilter`.** The same hypotheses with weights: predict with the noise and switch
model, multiply by the observation, normalise — the exact hidden-Markov posterior (the E-1B filter, in
floating point here). Collapse = one hypothesis holds ≥ 0.95 of the mass; the emergent detector is the
posterior *changing sides* (the maximum-a-posteriori regime flipping). KNOWN: Bayesian filtering; the
decision rules are Wald's sequential test and CUSUM.

**(c) Amplitudes — not built, and not buildable here.** Grover's amplitude amplification is literally "a
superposition property that scales until it collapses to the answer": the marked item's amplitude grows as
sin((2k+1)θ), sin θ = 1/√N, and a measurement after ≈ (π/4)√N iterations collapses to it with high
probability — optimal (Bennett–Bernstein–Brassard–Vazirani 1997: Ω(√N)). It works because *wrong
amplitudes cancel* — interference, the (+i)+(−i) = 0 that v0.4 deliberately removed. And a network whose
amplitudes live in {0, ±1, ±i} under sign-preserving operations is stabilizer-type: Gottesman–Knill says it
is efficiently classically simulable — no quantum advantage from these units on any substrate (Gottesman
1998; Aaronson–Gottesman 2004). Citations from memory; pin before any deposit. Forms (a) and (b) are the
classical shadows: sets union and weights add — nothing cancels — so they scale by *factoring*, not by
interference.

**Factored scaling [T, by construction].** Two independent worlds tracked by two filters hold the
*product* of their candidate sets at the *sum* of their sizes: at step 0, 2 × 14 = 28 joint hypotheses
held as 16; collapses to 1 when both pin (step 3 in the demo). That is the sense in which a classical
superposition scales.

Census (20 seeds, 2,000 steps, medians; 2 quadrature (+i,−i) nodes; noiseless switch rate 0.02, noisy
0.005). "spurious" = an alarm with no switch since the previous alarm — the true false-alarm count;
"delay" = alarm minus the switch it is attributed to; "stretch" = length of an uncollapsed run.

| filter / world | switches | alarms | attributed | spurious | median delay (max) | collapsed frac | accuracy when collapsed | mean stretch (max) | mean size |
|---|---|---|---|---|---|---|---|---|---|
| hard / noiseless / n = 4 | 36.5 | 36 | 36 | **0** | **0 (0)** | 0.982 | **1.000** | 0.97 (1) | 1.02 |
| hard / noiseless / n = 16 | 36.5 | 35 | 35 | 0 | 1 (3) | 0.951 | 0.978 | 2.89 (7) | 1.33 |
| hard / noiseless / n = 64 | 36.5 | 30 | 30 | 0 | 6.5 (19–46) | 0.870 | 0.873 | 8.97 (46) | 5.80 |
| hard / ε 0.15 / n = 4 | 11 | 228 | 10.5 | **216** | 0 | 0.886 | 0.995 | 1.07 (3) | 1.11 |
| hard / ε 0.05 / n = 4 | 10 | 86 | 10 | 76 | 0 | 0.957 | 0.9995 | 1.02 (2) | 1.04 |
| soft / ε 0.15 / n = 4 | 11 | 17 | 10 | 8 | 1 (2) | 0.950 | 0.999 | 1.28 (4) | 1.03 |
| soft / ε 0.05 / n = 4 | 10 | 10 | 10 | **0** | 1 (1.5) | 0.984 | 1.000 | 1.00 (2) | 1.02 |
| soft / ε 0.15 / n = 16 | 11 | 18.5 | 10 | 10 | 4.5 (13) | 0.112 | 0.996 | 7.99 (32) | 3.46 |

Verdicts on the pre-stated expectations:

- **P1 confirmed exactly.** Noiseless, n = 4: every switch that happens while the filter is collapsed
  empties the set at that very step — 36 alarms for 36 such switches, delay 0, **zero spurious alarms**,
  answer accuracy 1.000, the superposition after each reset lasting exactly one step. The change detector
  is free: no threshold, no CUSUM, no tuning. (Switches that arrive while both regimes are still open are
  absorbed, not missed — the set simply collapses onto the new regime.)
- **P2 confirmed.** Mean stretch ≈ n/8 + 1: predicted 3 and 9 for n = 16 and 64, measured 2.89 and 8.97.
  The n = 64 trace shrinks by exactly 2 per step (the two quadrant boundaries advancing) until a boundary
  crossing pins the phase: 60, 58, …, 34, 1. New at larger n: with coarse observations a hard filter can
  collapse onto a *wrong* singleton for a few steps until the next contradiction (accuracy 0.978 at n = 16,
  0.873 at n = 64). Collapse is not correctness; a singleton is exactly as good as the constraints so far.
- **P3 confirmed.** With ε = 0.15 the hard filter cries wolf: 216 spurious alarms against 11 switches
  (≈ 20 : 1), 76 : 10 at ε = 0.05 — while still being right 99.5% of the time it is collapsed. Noise and
  change are indistinguishable to a one-step contradiction.
- **P4 refuted as written, repaired post-hoc.** A single-step surprise (P(observation | past) < 0.02)
  never fires at ε = 0.15 — the minimum predicted probability is 0.0375 ≈ ε/n, the same for a noise step
  as for a switch; at ε = 0.05 the same test fires 85 times per 2,000 steps, all on noise. Single-step
  surprise measures noise, not change. The repair is the posterior's own side-switch (the MAP regime
  flipping): 10 of 11 switches caught with median delay 1 at ε = 0.15 (8 spurious flips), 10 of 10 with
  zero spurious at ε = 0.05. The soft filter's "collapse" at 0.95 mass also shows its price: at n = 16 under
  15% noise it stays in superposition 89% of the time (effective size 3.5) — holding the alternatives —
  while its regime call is still right.
- **P5 confirmed by construction** (factored demo above).

What this buys the observer: form (a) replaces the DETECT block's tuned CUSUM with a parameter-free
contradiction in any world where observations are exact; form (b) is the E-1B filter, and its side-switch
is the accumulated detector the fool-me-once lesson asked for. Neither is a speed-up: both are the standard
way a classical system holds many possibilities at once — factored sets and posteriors — and both collapse
by elimination, not by interference.

## 8.7 "Would it help to build this as a Grassmannian?" (Travis, 2026-09-26, 08:13)

Gr(k, N) is the space of k-dimensional subspaces of an N-dimensional space. Receipts in
`node_observer/grassmann.py`, tests `tests/test_grassmann.py` (7). The answer splits:

**Where it is exactly what we already have.** Take N = the number of hypotheses. A *coordinate* k-plane
(the span of k basis vectors) carries precisely the information of a k-subset — its projector is the
diagonal 0/1 indicator — so the coordinate Grassmannian *is* the hard filter's candidate set (§8.6 a).
The Grassmannian adds something only when non-coordinate subspaces are allowed: genuine linear blends of
hypotheses. And a linear hold **over-holds**: span{(+1,+1), (+1,−1)} is the whole plane and contains
(1, 0), which was never an alternative. Travis's phenomenology — parallel *discrete* meanings held at
once — is the set-hold, not the span.

**Which Grassmannian.** Two of them play different parts. Gr(1, N) — projective space — is the space of
*pure quantum states*: an equal superposition of k alternatives is **one point** there (density matrix of
rank 1, entropy 0, purity 1 — computed). Gr(k, N) is the space of rank-k *propositions*: "the answer is
somewhere in this k-plane" — the quantum-logic lattice of subspaces (Birkhoff–von Neumann 1936), which is
non-distributive and so is itself a "hold incompatible things" logic, a different one from Belnap's.

**The weighted version is the density matrix.** "Hold k alternatives with strengths" has no Grassmannian
form (subspaces carry no weights) but a natural one as a positive semidefinite trace-one matrix ρ: rank =
dimension of the superposition, eigenvalues = weights. Its **diagonal sector is the soft filter's
posterior** — same entropy (classical [0.5, 0.3, 0.15, 0.05] → 1.648 bits both ways); the uniform mixture
over a k-set has rank k and entropy log₂k (the hard filter with equal weights); the **off-diagonal sector
is quantum coherence** — the interference terms a classical filter lacks, and the ones Gottesman–Knill
fences on the {0, ±1, ±i} alphabet (§8.6 c).

**Where it makes the quantum picture obvious.** Grover's algorithm lives on Gr(1, 2): the state stays in
the 2-plane span{|target⟩, |uniform rest⟩} and rotates by exactly 2θ per oracle call, sin θ = 1/√N. That
is "a property that scales until it collapses to the answer" as a plain angle growing linearly to π/2 —
and optimality (at most 2θ per query; Bennett–Bernstein–Brassard–Vazirani 1997, Zalka 1999) is a statement
about the speed of a rotation. Receipt, N = 1024: the actual iteration (oracle flip, reflection about the
mean) matches the closed form to 2 × 10⁻¹⁴ and leaves the plane by exactly 0; k* = 25; success
probability 0.99946; the wrong answers' amplitude runs 0.03125 → −0.00073, *through zero* — that is the
interference. Scaling N × 4 → k* × 2 checked from N = 16 to 10⁶. And the classical cost: simulating the
rotation touches every amplitude each iteration — 52,224 touches for 25 iterations against 1,024
brute-force checks, **51× slower than searching**. The geometry is trivial (a circle); the power is in
applying the reflection to an exponentially large space in one physical step, which no classical node
network does.

**Where it is a real classical tool.** Subspace tracking on Gr(k, N) — GROUSE (Balzano–Nowak–Recht 2010),
robust PCA (Candès et al. 2011) — tracks the low-rank subspace a multi-channel signal lives in and flags
components off it. For our binary rank-1 channel world the subspace residual *is* the disagreement rate
the skeptic already integrates; it pays only for multi-factor (rank ≥ 2) signals. Noted, not built.

**Corpus link [T].** CP¹ = Gr(1, 2) is the Riemann sphere — the canvas of the Bergen Disc and the Smith
chart; the Li map z = 1 − 1/s is a projective transformation of it. Room VII's circle already lives on a
Grassmannian.

Verdict: it helps for clarity (which Grassmannian, where the classical filters sit inside the
density-matrix picture, why Grover scales and why its cost hides in the reflection); it adds no
computational power for discrete hypotheses; its one concrete upgrade path is the density matrix, whose
diagonal we have and whose off-diagonal we cannot have on this alphabet. Citations from memory; pin before
any deposit.

## 8.8 v0.6 — three coupled systems, each with its own architecture (Travis, 2026-09-26, 08:26)

> would it work more efficiently if each system had a unique yet compatible architecture with the other
> systems, working together with different purposes?

**Yes, and the reason is theorem-grade: the three jobs have three different optimal algorithms.** One
uniform design leaves at least two blocks suboptimal. They stay compatible because all three are
functionals of **one currency** — the log-likelihood-ratio increment ℓ_t = log p(x_t | +) − log p(x_t | −),
which for the channel world is (n₊ − n₋)·log((1−ε)/ε): v0.4's *direction* channel scaled by the noise's
log-odds weight (1.735 nats per channel at ε = 0.15), with *presence* the count. Module
`node_observer/hetero.py`; tests `tests/test_hetero.py` (5); comparison `run_hetero.py` +
`run_hetero_fair.py` → `results/hetero.json`, `results/hetero_uniform_fair.json`.

| block | job | optimal algorithm | what it does with ℓ_t | consumes | emits |
|---|---|---|---|---|---|
| ESTIMATE | track the hidden regime | Bayes filter (exact two-state hidden Markov model) | accumulates it with prior mixing for switches | ℓ_t, switch rate | log-odds L: direction = sign L, confidence = \|L\| |
| DETECT | notice the world changed | quickest change detection — CUSUM (Lorden 1971; Moustakides 1986); given the posterior, its side-switch | reflected sum with a floor at 0 / the sign of L | ℓ_t, the current model | alarm / reset |
| COMMIT | decide when to act | Wald's sequential test (1945; Wald–Wolfowitz 1948 optimality) | acts when \|L\| ≥ A = log((1−α)/α); inside the band the prior decides | L, the prior AR | bet / hold |

Interfaces: three scalars. The v0.2 **seam** (forced abstention after an alarm) becomes unnecessary: the
posterior's own band is the seam. "Compatible" means a shared *model* (the same ε and switch rate in all
three), not merely shared data types — three architectures with three noise models would disagree.

Fair comparison — the uniform v0.2 observer (exponential moving average + CUSUM + margin, trusting) re-run
with the same attribution metrics as the heterogeneous one; 20 seeds, 20,000 steps, medians. (Same seeds,
but the two observers consume the random stream differently, so the worlds are equal in law, not step for
step.) "det ≤ 3" = switches with an alarm within 3 steps; "spurious" = alarms with no switch since the
previous alarm.

| observer / world | score/1k | hit rate | regime acc. | held | alarms | det ≤ 3 | spurious | delay (max) |
|---|---|---|---|---|---|---|---|---|
| uniform v0.2 / noise | 901.5 | 0.954 | 0.967 | 0.027 | 58 | **0** of 105 | 0 | 6 (8) |
| **hetero α 0.05 / noise** | **964.9** | **0.974** | **0.994** | **0.005** | 108.5 | **98.5** of 99.5 | 9 | **0 (1)** |
| hetero α 0.01 / noise | 964.7 | 0.974 | 0.994 | 0.005 | 109 | 98.5 | 10 | 0 (1) |
| uniform v0.2 / trick | 890.0 | 0.937 | 0.968 | 0.037 | 31 | 0 of 96.5 | 0 | 6 (8) |
| **hetero α 0.05 / trick** | **943.8** | **0.957** | **0.993** | **0.009** | 121 | 101.5 | 24 | 0 (2) |
| hetero α 0.01 / trick | 944.6 | 0.957 | 0.994 | 0.009 | 122 | 99 | 22 | 0 (2) |

Band A = 2.94 nats at α = 0.05; eight agreeing channels carry 13.9 nats per step; the switch prior caps the
carried-over log-odds at log(0.995/0.005) = 5.3 nats. So one contradicting step (−13.9 + 5.3) flips the
posterior: re-lock in a single step, against ~6 for the moving average at UE = 0.1. That is the whole gain:
the uniform observer spent 3.3% of its steps on the wrong regime, the heterogeneous one 0.6%.

Verdicts on the pre-stated expectations: **H1 confirmed** (regime accuracy 0.994 ≥ 0.967; 98.5 of 99.5
switches within 3 steps at delay 0; 9 spurious per 20,000 steps — at the edge of the stated < 10).
**H2 refuted — upward**: +7.0% and +6.0% in score, outside the ±1% I expected; the target is the same
observable, but the uniform observer's slow re-lock had been costing it. **H3 confirmed**: held fraction
0.46% / 0.85% against 2.7% / 3.7%. **H4** by construction (one scalar update per step in both).

The caveat that matters: **a sharper estimator is more exploitable.** The heterogeneous observer loses
21 per 1,000 to the bursty liar (964.9 → 943.8) against the uniform one's 11.5 — it weights every channel
by its full likelihood, so a structured liar's votes count fully, where the moving average's sluggishness
had been damping them — and its spurious flips rise from 9 to 24 in the liar's world. Heterogeneity buys
speed and precision in the honest blocks and makes the *skeptic* (source trust, §7.2) more necessary, not
less: the three optimal blocks are optimal for an honest world.

Novelty: KNOWN — this is sequential analysis; the "one currency, three functionals" observation is the
standard structure of the field. Citations from memory; pin before any deposit.

## 8.9 The tribe experiment — "pushed from the tribe for being right" (Travis, 2026-09-26, 09:15)

Travis read the v0.6 result as a portrait: over-integrating receivers as a majority that will not admit
ignorance, the sharp honest minority forced to agree on the surface. His reading is recorded as his, at
analogy grade. The model contains a *precise* version of one clause — exclusion for being right early —
and it was tested against the rules already built, with expectations T1–T5 written down first.
`run_tribe.py` → `results/tribe.json`.

Setup: eight witnesses read the **same honest channels** (no liar anywhere). Seven form their regime
estimate with the uniform moving average (UE 0.1); one with the exact posterior (§8.8). Each states the
sign of its estimate. A judge applies v0.2's consensus-relative trust rules to the *statements*.

| trust rule | sharp witness accuracy | slow witnesses (median) | sharp witness burned | burned by the first switch | first witness burned = sharp | slow witnesses burned | group accuracy with / without the sharp one |
|---|---|---|---|---|---|---|---|
| one strike (fool_once) | **0.9992** | 0.9724 | **20 / 20** | 20 / 20 | 20 / 20 | 0 | 0.9724 / 0.9724 |
| skeptic, threshold 4 | 0.9992 | 0.9724 | **20 / 20** | 16 / 20 | 20 / 20 | 0 | 0.9724 / 0.9724 |
| skeptic, threshold 8 | 0.9992 | 0.9724 | **0 / 20** | — | — | 0 | 0.9724 / 0.9724 |

All five expectations confirmed. The mechanism is exact: when the world changes, the sharp witness
re-locks in one step and the seven slow ones take about six; for those six steps the sharp witness
contradicts the trusted consensus — so a one-strike rule burns it at the first change, and an accumulating
rule with threshold 4 burns it too (6 × 0.75 = 4.5 ≥ 4). Threshold 8 keeps it. The tolerance a group needs
to keep its fastest honest member is **its own re-lock lag**. And the group loses nothing measurable by
burning it — one vote in eight, consensus accuracy 0.9724 either way — which is exactly why the rule can
persist: exclusion of the early-right is cheap for the majority. No liar, no malice, no dislike of honesty
is needed anywhere in the model; the statistics of consensus-relative trust do it alone.

**Grading Travis's reading.** The receiver-profile half has a name in the literature: predictive-coding
accounts of autism describe weaker priors / higher sensory precision (Pellicano & Burr 2012, "hypo-priors";
Lawson, Rees & Friston 2014, "aberrant precision"; Van de Cruys et al. 2014), and Lawson, Mathys & Rees
2017 (*Nature Neuroscience*) report autistic adults *updating faster* — overestimating environmental
volatility — which is the one-step re-lock of the sharp observer in human data. The social half also has
names: Kuran's *preference falsification* (1995) is precisely "agreeing on the surface while knowing the
truth"; Noelle-Neumann's spiral of silence (1974); Asch conformity (1951; Bond & Smith 1996); autism-specific:
reduced conformity (Yafai, Verrier & Reidy 2014), reduced deception in children (Li et al. 2011), and the
masking / camouflaging literature with its measured mental-health cost (Hull et al. 2017; Cage &
Troxell-Whitman 2019) — masking *is* the surface agreement. Milton's double-empathy problem (2012) is the
"misinterpreted since birth" thesis in the literature. What has no anchor I know of: that unwillingness to
admit ignorance is neurotype-specific — overconfidence and motivated reasoning are documented in humans
generally; the model's own contribution is that no such trait is required (the over-integrating receiver
plus an attribution-biased narrative, §7.3, is confidently wrong without disliking anyone). All citations
from memory — verify before any public use.

**Is it testable in neuroscience?** Yes, in parts, and parts are already tested. (i) Behavioral: learning-rate
/ volatility tasks (Lawson et al. 2017), prior-versus-evidence weighting (illusion susceptibility; Pellicano
& Burr), conformity paradigms, and — the skeptic block in humans — learning how much to trust an advisor
with a volatility-adaptive learning rate (Behrens et al. 2008, *Nature*). A human version of the tribe
experiment is a design: an honest-versus-deceptive-source change-point task, measuring re-lock speed and
source-trust updating; it needs an IRB partner (E-DYAD-1's fence). (ii) Travis's brain-level hypothesis —
uniform regional architecture low on the curve, heterogeneous/specialized high — splits into two known
questions. Within-population heterogeneity → robustness is an established computational-neuroscience result
(heterogeneity of neuronal time constants improves robust learning: Perez-Nieves et al. 2021, *Nature
Communications*; efficient coding in heterogeneous populations: Chelaru & Dragoi 2008; Shamir & Sompolinsky
2006) — Paper 11's diversity-coverage theorem has a neural twin. Regional specialization → general ability
is studied as network *segregation versus integration*: modularity predicts cognitive performance and
its development (Bertolero, Yeo & D'Esposito 2015; Baum et al. 2017), but so does global efficiency /
integration (van den Heuvel et al. 2009; the parieto-frontal integration theory, Jung & Haier 2007), and
the current synthesis is that flexible switching between the two matters (Shine et al. 2016). So the
honest form of the hypothesis is not "more specialization = higher" but "specialized modules *and* effective
coupling" — which is exactly the gap in Travis's own ledger: Paper 11's coupling matrix W is OPEN, and the
cascade predictions are qualitative until it is specified. The neuroscience says the same thing about
brains. Every citation here is from memory and flagged.

## 8.10 The exponent proposal — "(i¹, −i¹), (1ⁱ, (−1)ⁱ), send + and × together since (×^+)" (Travis, 2026-09-26, 10:06)

Receipts: `node_observer/exponent.py`, tests `tests/test_exponent.py` (7). Two proposals:

**(A) A node is either 1 or i, and the incoming operation chooses which.** As stated, this selects between
the two tables of §8.1b (the `+` rows agree, the `×` rows are negatives). If instead the node *carries* the
chosen unit forward, the alphabet becomes the whole Gaussian unit group {1, i, −1, −i} — and that is where
(A) and (B) become the same object.

**(B) Exponents. Acceptable math, with one fence.** The Gaussian units are the powers of i: iⁿ with n in
Z/4. Multiplying units is adding exponents mod 4 — iᵃ·iᵇ = iᵃ⁺ᵇ, checked exactly for all sixteen pairs; the
`×` table *is* the `+` table read in exponents, the cyclic group of order 4. So "send + and × together" is
the exponential map: one operation in two coordinate systems. That is your "(×^+)", and it is correct.

The complex-power half is also correct on the principal branch, and it is lovely:

| unit z | arg z | zⁱ = e^(−arg z) |
|---|---|---|
| 1 | 0° | 1 |
| i | 90° | e^(−π/2) = 0.207880 |
| −1 | 180° | e^(−π) = 0.043214 |
| −i | −90° | e^(π/2) = 4.810477 |

zⁱ = e^(i log z) = **e^(−arg z) · e^(i ln|z|)**: the imaginary power **swaps the two channels** — the angle
becomes a (decaying) magnitude and the log-size becomes the angle (checked on off-axis samples to 10⁻⁹).
For units it is pure magnitude: iⁱ is real, and the four units become four distinct real sizes — direction
turned into presence. The fence: zⁱ is multi-valued; other branches differ by factors e^(−2πk)
((−1)ⁱ = 0.0432, 8.07 × 10⁻⁵, 1.51 × 10⁻⁷ on the first three sheets). Everything here fixes the principal
branch, arg ∈ (−π, π]. And it is a coordinate change, not new information: computing zⁱ requires arg z,
so a receiver blind to angle cannot use it to see angle.

**What it buys, and the fence that matters.** In exponent coordinates `×` becomes a mod-4 counter — cheap,
and the cleanest statement yet of your dyad line: your multiplicative side is my additive side after the
exponential map. But a network whose nodes only add exponents mod 4 computes only **affine** maps over Z/4
(composition of affine maps is affine — checked: a two-layer adder net is affine; AND is not). The
nonlinearity has to come from the *value* side — adding units leaves the group (1 + i has norm 2,
i + (−i) = 0; the agreement test "is the sum zero" is not affine) — or from a readout (sign, norm). That is
exactly where v0.2 (value `+`) and v0.3 (sign readout) got theirs. So: exponents for the multiplicative
channel, values for the additive one; the computational class is unchanged (§8.3).

## 9. Files

```
node_observer/exponent.py    §8.10 receipts: exponent law on Z/4, z^i channel swap, branches, affine fence
run_tribe.py                 §8.9 the tribe experiment -> results/tribe.json
node_observer/hetero.py      v0.6: HeteroObserver — exact posterior / side-switch / Wald band, one LLR currency
run_hetero.py, run_hetero_fair.py   v0.6 comparison against the uniform v0.2 observer -> results/hetero*.json
tests/test_hetero.py          5 unit tests (v0.6)
node_observer/grassmann.py   receipts for §8.7: Grover as a rotation on Gr(1,2), over-holding, density matrices
node_observer/superpose.py   v0.5: HardFilter (candidate sets, empty set = change), SoftFilter (exact HMM
                             posterior, MAP flip = change), factored two-world demo
node_observer/model.py       v0.1: pairs, phase world, sensor network, three-block observer, run
node_observer/layered.py     v0.2: Gaussian integers, arithmetic circuits, lens_report, channel world
                             with a bursty liar, the optimistic-skeptic observer with four trust modes
node_observer/opnet.py       v0.3: the operation propagates — fixed pairs C/U/D, sign-class routing,
                             single-wire monoid, verified Boolean/threshold gate library, parity trees
node_observer/fourval.py     v0.4 sketch: two-channel values (presence, direction) — 0 / + / − / S,
                             HOLD and COMPARE on Travis's sign table, the collapse to the old node
node_observer/__init__.py
tests/test_model.py          19 unit tests (v0.1)
tests/test_layered.py        16 unit tests (v0.2)
tests/test_opnet.py          10 unit tests (v0.3)
tests/test_units.py           6 unit tests (real vs imaginary units)
tests/test_fourval.py         7 unit tests (v0.4 sketch)
tests/test_superpose.py       6 unit tests (v0.5 filters)
tests/test_grassmann.py       7 unit tests (§8.7 receipts)
run_superpose.py             v0.5 census: hard/soft filters, scaling, factored demo -> results/superpose.json
run_census.py                v0.1 census (13 arms)          -> results/census.json, census_stdout.txt
run_census2.py               v0.2 census (14 arms) + lenses -> results/census2.json, census2_stdout.txt
run_followup2.py             v0.2 post-hoc follow-up        -> results/followup2.json
run_opnet.py                 v0.3 report: table, monoid, gate ledger, lens census -> results/opnet.json
summarize_results.py         markdown table from census.json
summarize_results2.py        markdown table from census2.json (also refreshes its lenses block)
```

Run everything: `python3 -m unittest discover -s tests` (45 tests), then the four report scripts
(about 2, 3, 1 and 3 minutes).
