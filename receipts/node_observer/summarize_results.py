"""Print a markdown table and a few diagnostics from results/census.json.

    python3 summarize_results.py results/census.json
"""
import json
import sys


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main(path):
    d = json.load(open(path))
    print(f"seeds {len(d['seeds'])}, steps {d['steps']}\n")
    print("| arm | score/1k | hit rate | ambiguous frac | bet-when-ambiguous | alarms | delay | missed | AR final | AR sign split (+/0/−) | narrative | true | ego gap |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, arm in d["arms"].items():
        s = arm["summary"]
        m = lambda k, nd=3: fmt(s[k]["median"], nd)
        split = s["AR_sign_split"]
        print(f"| {name} | {m('score_per_1000',1)} | {m('hit_rate')} | {m('ambiguous_frac')} | {m('ambiguous_bet_rate')} | "
              f"{m('alarms',0)} | {m('median_delay',0)} | {m('missed',0)} | {m('AR_final')} | "
              f"{split['positive']}/{split['near_zero']}/{split['negative']} | {m('narrative_rate')} | {m('true_rate')} | {m('ego_gap')} |")
    print()
    for name in ("honest", "ego_0.9", "honest_high_noise", "ego_0.9_high_noise", "ego_0.9_blinds_detect"):
        runs = d["arms"][name]["runs"]
        r0 = runs[0]
        tr = r0["AR_trace"]
        pts = [tr[i] for i in (0, 4, 9, 24, 49, 99) if i < len(tr)]
        print(f"{name}: seed 0 AR at steps 200/1000/2000/5000/10000/20000 -> {pts}; bits0[:32] = {''.join(map(str, r0['bits0'][:32]))}")
        ars = sorted(r["AR_final"] for r in runs)
        print(f"   AR_final over seeds: min {ars[0]:.3f} max {ars[-1]:.3f}; resolved decisions median "
              f"{sorted(r['resolved'] for r in runs)[len(runs)//2]}; switches median {sorted(r['switches'] for r in runs)[len(runs)//2]}")
    print()
    print("Gaussian pairs (b >= 0), radius 2: alpha, trace, norm, gap, |alpha-1|^2, blind")
    for row in d["gaussian_table"]:
        if row["blind"] or row["gap"] <= 3:
            print(f"  {row['alpha']:>6}  t={row['trace']:>3} n={row['norm']:>3} gap={row['gap']:>2} d2={row['dist2']:>2} {'BLIND' if row['blind'] else ''}")
    print("Eisenstein blind pairs, radius 2:")
    for row in d["eisenstein_table"]:
        if row["blind"]:
            print(f"  {row['alpha']:>6}  t={row['trace']:>3} n={row['norm']:>3}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/census.json")
