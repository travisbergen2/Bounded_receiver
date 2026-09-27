// Headless check of core.js against twin.json (the Python twin). Decisions/counters exact; floats to 1e-9;
// parameter guards must reject the same cases; the odds-mixing bound must hold on both sides.
//    node receipts/node_check.js
const fs = require("fs"), path = require("path");
require(path.join(__dirname, "..", "core.js"));
const twin = JSON.parse(fs.readFileSync(path.join(__dirname, "twin.json"), "utf8"));
const R = globalThis.Room;
let pass = 0, fail = 0; const fails = [];
function eq(name, a, b, tol) {
  let ok;
  if (typeof a === "number" && typeof b === "number") ok = Math.abs(a - b) <= (tol || 0) * Math.max(1, Math.abs(b));
  else ok = JSON.stringify(a) === JSON.stringify(b);
  if (ok) pass++; else { fail++; fails.push(`${name}: got ${JSON.stringify(a)} want ${JSON.stringify(b)}`); }
}
// PRNG
const r1 = R.mulberry32(1); for (let k = 0; k < 5; k++) eq(`prng[${k}]`, r1(), twin.prng_first5[k], 0);
// nodes
R.configTable().forEach((row, i) => { const w = twin.config_table[i]; for (const k of ["plus", "times", "plus_class", "times_class"]) eq(`config ${row.config} ${k}`, row[k], w[k]); });
eq("blind lattice", R.blindLattice(3), twin.blind_lattice);
// lenses
[["+", "+"], ["*", "*"], ["*", "+"], ["+", "*"]].forEach(([a, b], i) => {
  const got = R.lensTree4(a, b), w = twin.lenses[i];
  eq(`lens ${a}${b} zero`, got.zero, w.zero); eq(`lens ${a}${b} distinct`, got.distinct, w.distinct);
  eq(`lens ${a}${b} H`, got.H_value, w.H_value, 1e-9); eq(`lens ${a}${b} MI_sign`, got.MI_sign, w.MI_sign, 1e-9);
  eq(`lens ${a}${b} MI_parity`, got.MI_parity, w.MI_parity, 1e-9); eq(`lens ${a}${b} blind`, got.sign_blind, w.sign_blind);
});
// observer runs
const INT = ["score", "hits", "misses", "voids", "bets", "ambiguous", "alarms", "recorded", "resolved", "seam_abstain", "attributed", "spurious"];
const FLT = ["hit_rate", "ambiguous_bet_rate", "regime_accuracy", "AR_final", "narrative_rate", "true_rate", "median_delay"];
for (const rr of twin.runs) {
  const p = rr.params, got = R.run(p.seed, p.steps, p.wp, p.up, p.hp, false), w = rr.result;
  eq(`run${p.seed} switches`, got.switches, w.switches);
  for (const side of ["uniform", "hetero"]) {
    for (const k of INT) eq(`run${p.seed} ${side} ${k}`, got[side][k], w[side][k]);
    for (const k of FLT) { if (w[side][k] === null) eq(`run${p.seed} ${side} ${k}`, got[side][k], null); else eq(`run${p.seed} ${side} ${k}`, got[side][k], w[side][k], 1e-9); }
  }
  eq(`run${p.seed} hetero odds_finite`, got.hetero.odds_finite, w.hetero.odds_finite);
  eq(`run${p.seed} hetero odds_final`, got.hetero.odds_final, w.hetero.odds_final, 1e-9);
}
// tribes
for (const tt of twin.tribes) {
  const p = tt.params, got = R.tribe(p.seed, p.steps, null, p.rule, p.FT, p.nSlow === undefined ? null : p.nSlow), w = tt.result;
  const tag = `tribe ${p.rule}${p.FT}${p.nSlow ? "n" + p.nSlow : ""}`;
  eq(`${tag} switches`, got.switches, w.switches);
  eq(`${tag} burned`, got.burned, w.burned);
  eq(`${tag} burn_time`, got.burn_time, w.burn_time);
  got.accuracy.forEach((a, j) => eq(`${tag} acc${j}`, a, w.accuracy[j], 1e-9));
  eq(`${tag} group_with`, got.group_with, w.group_with, 1e-9);
  eq(`${tag} group_without`, got.group_without, w.group_without, 1e-9);
}
// boundary: the same invalid parameter sets must be rejected here
for (const c of twin.boundary.invalid_cases) {
  let threw = false;
  try {
    const a = c.case.args;
    if (c.case.kind === "run") R.run(1, a.steps === undefined ? 100 : a.steps, a.wp, a.up, a.hp, false);
    else R.tribe(1, 100, null, a.rule === undefined ? "strike" : a.rule, a.FT === undefined ? 0 : a.FT, a.nSlow === undefined ? null : a.nSlow);
  } catch (e) { threw = true; }
  eq(`rejects ${JSON.stringify(c.case)}`, threw, true); eq(`twin rejects ${JSON.stringify(c.case)}`, c.raises, true);
}
// boundary: the odds-mixing bound
for (const row of twin.boundary.mixing) {
  const m = R.mixOdds(row.odds, row.ps);
  eq(`mix ${row.ps} ${row.odds}`, m, row.mixed, 1e-12); eq(`mix inside ${row.ps} ${row.odds}`, row.inside, true);
  const lo = row.ps / (1 - row.ps), hi = (1 - row.ps) / row.ps; eq(`mix bound js ${row.ps} ${row.odds}`, m >= lo - 1e-12 && m <= hi + 1e-12, true);
}
// median convention
eq("median odd", R.median([5, 1, 3]), 3); eq("median even", R.median([4, 1, 3, 2]), 2.5); eq("median empty", R.median([]), null);
console.log(`node_check: ${pass} passed, ${fail} failed`);
if (fail) { console.log(fails.slice(0, 20).join("\n")); process.exit(1); }
