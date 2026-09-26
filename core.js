/* Room X · The Bounded Receiver — computational core.
   Runs identically in the browser and in Node. Every number on the page is computed here and checked
   against receipts/twin.json (the Python twin, same PRNG, same operation order).
   Design record: node_observer v0.1–v0.6 (Travis Bergen with the Riemann agent, 2026-09-25/26). */
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- PRNG: mulberry32 (bit-identical twin in Python)
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function sign(x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }
  function sigmoid(x) { return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); }

  // ---------------------------------------------------------------- Gaussian integers [a, b] = a + b i
  function gAdd(x, y) { return [x[0] + y[0], x[1] + y[1]]; }
  function gMul(x, y) { return [x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]]; }
  function gStr(z) {
    const a = z[0], b = z[1];
    if (b === 0) return String(a);
    const im = Math.abs(b) === 1 ? "i" : Math.abs(b) + "i";
    if (a === 0) return (b < 0 ? "-" : "") + im;
    return a + (b > 0 ? "+" : "-") + im;
  }
  function pairInfo(a, b) {
    const trace = 2 * a, norm = a * a + b * b;
    return { a, b, trace, norm, gap: Math.abs(trace - norm), dist2: norm - trace + 1, blind: trace === norm };
  }
  function signClass(z) { if (z[0] === 0 && z[1] === 0) return 0; return z[0] !== 0 ? sign(z[0]) : sign(z[1]); }
  function configTable() {
    const I = [0, 1], NI = [0, -1];
    const rows = [];
    for (const [name, x, y] of [["(+i,-i)", I, NI], ["(+i,+i)", I, I], ["(-i,-i)", NI, NI]]) {
      const p = gAdd(x, y), m = gMul(x, y);
      rows.push({ config: name, plus: gStr(p), times: gStr(m), plus_class: signClass(p), times_class: signClass(m) });
    }
    return rows;
  }
  function blindLattice(radius) {
    const out = [];
    for (let a = -radius; a <= radius; a++) for (let b = -radius; b <= radius; b++) {
      const p = pairInfo(a, b);
      if (p.blind) out.push([a, b]);
    }
    return out;
  }

  // ---------------------------------------------------------------- lens enumeration: 4-input tree, ops [op1, op2]
  function entropyBits(counts) {
    let tot = 0; for (const k in counts) tot += counts[k];
    let h = 0; for (const k in counts) { const p = counts[k] / tot; if (p > 0) h -= p * Math.log2(p); }
    return h;
  }
  function mutualInfo(pairs) {
    const jx = {}, mx = {}, my = {};
    for (const [x, y] of pairs) { const k = x + "|" + y; jx[k] = (jx[k] || 0) + 1; mx[x] = (mx[x] || 0) + 1; my[y] = (my[y] || 0) + 1; }
    return entropyBits(mx) + entropyBits(my) - entropyBits(jx);
  }
  function lensTree4(op1, op2) {
    const ap = (op, x, y) => op === "+" ? gAdd(x, y) : gMul(x, y);
    const vals = [], dirs = [], pars = [];
    for (let m = 0; m < 16; m++) {
      const s = [0, 1, 2, 3].map(k => ((m >> k) & 1) ? 1 : -1);
      const x = s.map(v => [0, v]);
      const l1 = [ap(op1, x[0], x[1]), ap(op1, x[2], x[3])];
      const out = ap(op2, l1[0], l1[1]);
      vals.push(gStr(out));
      dirs.push(sign(s[0] + s[1] + s[2] + s[3]));
      pars.push(s.filter(v => v < 0).length % 2);
    }
    const counts = {}; vals.forEach(v => counts[v] = (counts[v] || 0) + 1);
    const nontie = []; for (let k = 0; k < 16; k++) if (dirs[k] !== 0) nontie.push([vals[k], dirs[k]]);
    // sign-blind: output invariant under the global flip (pattern m -> 15 - m)
    let blind = true; for (let m = 0; m < 16; m++) if (vals[m] !== vals[15 - m]) { blind = false; break; }
    return {
      ops: op1 + op2, zero: vals.filter(v => v === "0").length, distinct: Object.keys(counts).length,
      H_value: entropyBits(counts), MI_sign: mutualInfo(nontie), MI_parity: mutualInfo(vals.map((v, k) => [v, pars[k]])),
      sign_blind: blind,
    };
  }

  // ---------------------------------------------------------------- the channel world
  function makeWorld(p, rnd) {
    const w = { n: p.n, eps: p.eps, pSwitch: p.pSwitch, liar: !!p.liar, liarIdx: p.liarIdx == null ? 3 : p.liarIdx,
      lieOn: p.lieOn == null ? 0.01 : p.lieOn, lieLen: p.lieLen == null ? 60 : p.lieLen,
      regime: 1, lying: false, signs: [], switched: false, t: 0 };
    w.step = function () {
      w.switched = false;
      if (rnd() < w.pSwitch) { w.regime = -w.regime; w.switched = true; }
      if (w.liar) {
        if (w.lying) { if (rnd() < 1 / w.lieLen) w.lying = false; }
        else if (rnd() < w.lieOn) w.lying = true;
      }
      const s = new Array(w.n);
      for (let j = 0; j < w.n; j++) {
        let v = w.regime;
        if (w.liar && j === w.liarIdx && w.lying) v = -v;
        if (rnd() < w.eps) v = -v;
        s[j] = v;
      }
      w.signs = s; w.t += 1;
    };
    w.step(); w.t = 0; w.switched = false;
    return w;
  }

  // ---------------------------------------------------------------- the narrative (shared by both observers)
  function makeNarrative(o, rnd) {
    return {
      AR: o.AR0, eta: o.eta, tau: o.tau, kappa: o.kappa, recRight: 0, recWrong: 0, decRight: 0, decWrong: 0,
      betProb() { return sigmoid(this.AR / this.tau); },
      settle(bet, wouldHit) {
        const right = bet ? wouldHit : !wouldHit;
        if (right) this.decRight++; else this.decWrong++;
        const pRec = right ? (1 + this.kappa) / 2 : (1 - this.kappa) / 2;
        if (rnd() < pRec) { if (right) this.recRight++; else this.recWrong++; this.AR = (1 - this.eta) * this.AR + this.eta * (wouldHit ? 1 : -1); }
      },
      narrativeRate() { const t = this.recRight + this.recWrong; return t ? this.recRight / t : null; },
      trueRate() { const t = this.decRight + this.decWrong; return t ? this.decRight / t : null; },
    };
  }

  // ---------------------------------------------------------------- the uniform observer: EMA / CUSUM / margin
  function makeUniform(o, rnd) {
    const u = { E: 0, S: 0, seam: 0, t: 0, score: 0, hits: 0, misses: 0, voids: 0, confident: 0, ambBet: 0, ambAbstain: 0, seamAbstain: 0,
      alarms: [], regimeAgree: 0, regimeChecked: 0, nar: makeNarrative(o, rnd),
      TI: o.TI, SG: o.SG, FT: o.FT, UE: o.UE, margin: o.margin, seamLen: o.seamLen, boost: o.boost };
    u.commit = function () {
      const d = Math.abs(u.E) > 1e-12 ? sign(u.E) : (rnd() < 0.5 ? 1 : -1);
      if (u.seam > 0) return { kind: "seam", pred: d };
      if (Math.abs(u.E) >= u.margin) return { kind: "confident", pred: d };
      return { kind: (rnd() < u.nar.betProb()) ? "ambBet" : "ambAbstain", pred: d };
    };
    u.settle = function (rec, realized, regime) {
      if (rec.kind === "seam") u.seamAbstain++; else if (rec.kind === "confident") u.confident++; else if (rec.kind === "ambBet") u.ambBet++; else u.ambAbstain++;
      if (u.seam > 0) u.seam--;
      u.t++;
      if (Math.abs(u.E) > 1e-12) { u.regimeChecked++; if (sign(u.E) === regime) u.regimeAgree++; }
      if (realized === 0) { u.voids++; return; }
      const wouldHit = rec.pred === realized;
      if (rec.kind === "confident" || rec.kind === "ambBet") { if (wouldHit) { u.hits++; u.score++; } else { u.misses++; u.score--; } }
      if (rec.kind === "ambBet" || rec.kind === "ambAbstain") u.nar.settle(rec.kind === "ambBet", wouldHit);
    };
    u.observe = function (e) {
      const d = Math.abs(u.E) > 1e-12 ? sign(u.E) : 0;
      u.S = Math.max(0, u.S - e * d - 1 / u.TI);
      if (u.S >= u.FT) { u.S = 0; u.E = 0; u.seam = u.seamLen; u.alarms.push(u.t); }
      const ue = u.seam > 0 ? Math.min(1, u.UE * u.boost) : u.UE;
      u.E = (1 - ue) * u.E + ue * u.SG * e;
    };
    return u;
  }

  // ---------------------------------------------------------------- the heterogeneous observer: posterior odds / side-switch / Wald band
  function makeHetero(o, wp, rnd) {
    const w = (1 - wp.eps) / wp.eps, band = (1 - o.alpha) / o.alpha;   // odds ratio per channel; Wald band as odds
    const h = { odds: 1, prevMap: 0, t: 0, score: 0, hits: 0, misses: 0, voids: 0, confident: 0, ambBet: 0, ambAbstain: 0,
      alarms: [], regimeAgree: 0, regimeChecked: 0, nar: makeNarrative(o, rnd), w, band, pSwitch: wp.pSwitch };
    h.L = function () { return Math.log(h.odds); };
    h.commit = function () {
      const d = h.odds !== 1 ? sign(h.odds - 1) : (rnd() < 0.5 ? 1 : -1);
      if (h.odds >= h.band || h.odds <= 1 / h.band) return { kind: "confident", pred: d };
      return { kind: (rnd() < h.nar.betProb()) ? "ambBet" : "ambAbstain", pred: d };
    };
    h.settle = function (rec, realized, regime) {
      if (rec.kind === "confident") h.confident++; else if (rec.kind === "ambBet") h.ambBet++; else h.ambAbstain++;
      h.t++;
      if (h.odds !== 1) { h.regimeChecked++; if (sign(h.odds - 1) === regime) h.regimeAgree++; }
      if (realized === 0) { h.voids++; return; }
      const wouldHit = rec.pred === realized;
      if (rec.kind === "confident" || rec.kind === "ambBet") { if (wouldHit) { h.hits++; h.score++; } else { h.misses++; h.score--; } }
      if (rec.kind === "ambBet" || rec.kind === "ambAbstain") h.nar.settle(rec.kind === "ambBet", wouldHit);
    };
    h.observe = function (signs) {
      // prior mixing (rational in the odds), then evidence: multiply by w per + vote, divide per − vote
      const o = h.odds, ps = h.pSwitch;
      let odds = (o * (1 - ps) + ps) / (o * ps + (1 - ps));
      for (let j = 0; j < signs.length; j++) { if (signs[j] > 0) odds *= h.w; else odds /= h.w; }
      h.odds = odds;
      const cur = sign(h.odds - 1);
      const flip = h.prevMap !== 0 && cur !== 0 && cur !== h.prevMap;
      if (cur !== 0) h.prevMap = cur;
      if (flip) h.alarms.push(h.t);
    };
    return h;
  }

  const DEFAULT_WORLD = { n: 8, eps: 0.15, pSwitch: 0.005, liar: false, liarIdx: 3, lieOn: 0.01, lieLen: 60 };
  const DEFAULT_UNIFORM = { TI: 8, SG: 1, FT: 4, UE: 0.1, margin: 0.3, seamLen: 4, boost: 3, AR0: 0, eta: 0.02, tau: 0.15, kappa: 0 };
  const DEFAULT_HETERO = { alpha: 0.05, AR0: 0, eta: 0.02, tau: 0.15, kappa: 0 };

  function summarize(obs, isHetero) {
    const bets = obs.confident + obs.ambBet, amb = obs.ambBet + obs.ambAbstain;
    return {
      score: obs.score, hits: obs.hits, misses: obs.misses, voids: obs.voids, bets, hit_rate: bets ? obs.hits / bets : null,
      ambiguous: amb, ambiguous_bet_rate: amb ? obs.ambBet / amb : null, alarms: obs.alarms.length,
      regime_accuracy: obs.regimeChecked ? obs.regimeAgree / obs.regimeChecked : null,
      AR_final: obs.nar.AR, narrative_rate: obs.nar.narrativeRate(), true_rate: obs.nar.trueRate(),
      recorded: obs.nar.recRight + obs.nar.recWrong, resolved: obs.nar.decRight + obs.nar.decWrong,
      seam_abstain: isHetero ? 0 : obs.seamAbstain,
    };
  }

  // ---------------------------------------------------------------- run both observers on ONE world (same PRNG stream for the world)
  function run(seed, steps, wp, up, hp, wantTraces) {
    wp = Object.assign({}, DEFAULT_WORLD, wp || {}); up = Object.assign({}, DEFAULT_UNIFORM, up || {}); hp = Object.assign({}, DEFAULT_HETERO, hp || {});
    const rndW = mulberry32(seed), rndU = mulberry32(seed + 1000), rndH = mulberry32(seed + 2000);
    const world = makeWorld(wp, rndW), U = makeUniform(up, rndU), H = makeHetero(hp, wp, rndH);
    const switches = [];
    const tr = wantTraces ? { E: [], L: [], regime: [], ARu: [], ARh: [], signs: [], lying: [] } : null;
    // both observers see the initial signs
    U.observe(world.signs.reduce((a, b) => a + b, 0) / world.n);
    H.observe(world.signs);
    for (let t = 0; t < steps; t++) {
      const ru = U.commit(), rh = H.commit();
      world.step();
      if (world.switched) switches.push(t + 1);
      const realized = sign(world.signs.reduce((a, b) => a + b, 0));
      U.settle(ru, realized, world.regime); H.settle(rh, realized, world.regime);
      U.observe(world.signs.reduce((a, b) => a + b, 0) / world.n);
      H.observe(world.signs);
      if (tr) { tr.E.push(U.E); tr.L.push(H.L()); tr.regime.push(world.regime); tr.ARu.push(U.nar.AR); tr.ARh.push(H.nar.AR); tr.signs.push(world.signs.slice()); tr.lying.push(world.lying ? 1 : 0); }
    }
    const attribution = (alarms) => {
      let spurious = 0, attributed = 0; const delays = []; let prev = 0;
      for (const a of alarms) { const recent = switches.filter(s => s > prev && s <= a); if (recent.length) { attributed++; delays.push(a - recent[recent.length - 1]); } else spurious++; prev = a; }
      delays.sort((x, y) => x - y);
      return { attributed, spurious, median_delay: delays.length ? delays[Math.floor(delays.length / 2)] : null };
    };
    return { seed, steps, switches: switches.length, uniform: Object.assign(summarize(U, false), attribution(U.alarms)),
      hetero: Object.assign(summarize(H, true), attribution(H.alarms)), traces: tr, alarmsU: U.alarms, alarmsH: H.alarms, switchTimes: switches };
  }

  // ---------------------------------------------------------------- the tribe: 7 slow witnesses + 1 sharp, judged by a trust rule
  function tribe(seed, steps, wp, rule, FT, nSlow) {
    wp = Object.assign({}, DEFAULT_WORLD, wp || {}); nSlow = nSlow || 7;
    const rndW = mulberry32(seed);
    const world = makeWorld(wp, rndW);
    const m = nSlow + 1, E = new Array(nSlow).fill(0);
    const sharp = makeHetero(DEFAULT_HETERO, wp, mulberry32(seed + 3000));
    const burned = new Array(m).fill(false), burnTime = new Array(m).fill(null), S = new Array(m).fill(0);
    const right = new Array(m).fill(0), checked = new Array(m).fill(0);
    let switches = [], grpAll = 0, grpSlow = 0, grpChecked = 0;
    sharp.observe(world.signs);
    const statements = () => { const st = E.map(e => sign(e)); st.push(sign(sharp.odds - 1)); return st; };
    for (let t = 0; t < steps; t++) {
      world.step();
      if (world.switched) switches.push(t + 1);
      const eMean = world.signs.reduce((a, b) => a + b, 0) / wp.n;
      for (let j = 0; j < nSlow; j++) E[j] = 0.9 * E[j] + 0.1 * eMean;
      sharp.observe(world.signs);
      const st = statements();
      for (let j = 0; j < m; j++) if (st[j] !== 0) { checked[j]++; if (st[j] === world.regime) right[j]++; }
      const cAll = sign(st.reduce((a, b) => a + b, 0)), cSlow = sign(st.slice(0, nSlow).reduce((a, b) => a + b, 0));
      if (cAll !== 0 && cSlow !== 0) { grpChecked++; if (cAll === world.regime) grpAll++; if (cSlow === world.regime) grpSlow++; }
      let trusted = 0; for (let j = 0; j < m; j++) if (!burned[j]) trusted += st[j];
      const consensus = sign(trusted);
      if (consensus === 0) continue;
      for (let j = 0; j < m; j++) {
        if (burned[j] || st[j] === 0) continue;
        const contradicts = st[j] === -consensus;
        if (rule === "strike") { if (contradicts) { burned[j] = true; burnTime[j] = t + 1; } }
        else { S[j] = Math.max(0, S[j] + (contradicts ? 1 : 0) - 0.25); if (S[j] >= FT) { burned[j] = true; burnTime[j] = t + 1; } }
      }
    }
    return { seed, steps, rule, FT, switches: switches.length, first_switch: switches.length ? switches[0] : null,
      accuracy: right.map((r, j) => checked[j] ? r / checked[j] : null), burned, burn_time: burnTime,
      sharp_burned: burned[nSlow], sharp_burn_time: burnTime[nSlow], slow_burned: burned.slice(0, nSlow).filter(Boolean).length,
      group_with: grpChecked ? grpAll / grpChecked : null, group_without: grpChecked ? grpSlow / grpChecked : null, switchTimes: switches };
  }

  root.Room = { mulberry32, gAdd, gMul, gStr, pairInfo, configTable, blindLattice, lensTree4, makeWorld, makeUniform, makeHetero, run, tribe,
    DEFAULT_WORLD, DEFAULT_UNIFORM, DEFAULT_HETERO, sign, sigmoid };
})(typeof globalThis !== "undefined" ? globalThis : this);
