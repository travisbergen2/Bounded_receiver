// Headless check of core.js against twin.json (the Python twin). Decisions/counters exact; floats to 1e-9.
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
const FLT = ["hit_rate", "ambiguous_bet_rate", "regime_accuracy", "AR_final", "narrative_rate", "true_rate"];
for (const rr of twin.runs) {
  const p = rr.params, got = R.run(p.seed, p.steps, p.wp, p.up, p.hp, false), w = rr.result;
  eq(`run${p.seed} switches`, got.switches, w.switches);
  for (const side of ["uniform", "hetero"]) {
    for (const k of INT) eq(`run${p.seed} ${side} ${k}`, got[side][k], w[side][k]);
    for (const k of FLT) { if (w[side][k] === null) eq(`run${p.seed} ${side} ${k}`, got[side][k], null); else eq(`run${p.seed} ${side} ${k}`, got[side][k], w[side][k], 1e-9); }
    eq(`run${p.seed} ${side} median_delay`, got[side].median_delay, w[side].median_delay);
  }
}
// tribes
for (const tt of twin.tribes) {
  const p = tt.params, got = R.tribe(p.seed, p.steps, null, p.rule, p.FT), w = tt.result;
  eq(`tribe ${p.rule}${p.FT} switches`, got.switches, w.switches);
  eq(`tribe ${p.rule}${p.FT} burned`, got.burned, w.burned);
  eq(`tribe ${p.rule}${p.FT} burn_time`, got.burn_time, w.burn_time);
  got.accuracy.forEach((a, j) => eq(`tribe ${p.rule}${p.FT} acc${j}`, a, w.accuracy[j], 1e-9));
  eq(`tribe ${p.rule}${p.FT} group_with`, got.group_with, w.group_with, 1e-9);
  eq(`tribe ${p.rule}${p.FT} group_without`, got.group_without, w.group_without, 1e-9);
}
console.log(`node_check: ${pass} passed, ${fail} failed`);
if (fail) { console.log(fails.slice(0, 20).join("\n")); process.exit(1); }
