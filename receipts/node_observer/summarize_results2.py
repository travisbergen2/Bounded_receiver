"""Markdown table + diagnostics from results/census2.json; also refreshes its 'lenses' block.

    python3 summarize_results2.py results/census2.json
"""
import json
import sys

from run_census2 import lenses


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main(path):
    d = json.load(open(path))
    d["lenses"] = lenses()  # refresh with the sign-only measure
    json.dump(d, open(path, "w"), indent=1)
    print(f"seeds {len(d['seeds'])}, steps {d['steps']}\n")
    print("| arm (trust / world / witnesses) | score/1k | hit rate | regime acc. | ambiguous frac | bet-when-ambiguous | alarms | burned | seeds w/ true burn | false burns (total) | first true burn | AR final | AR split (+/0/−) | narrative | true | gap |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, arm in d["arms"].items():
        s = arm["summary"]
        m = lambda k, nd=3: fmt(s[k]["median"], nd)
        sp = s["AR_sign_split"]
        print(f"| {name} | {m('score_per_1000',1)} | {m('hit_rate')} | {m('regime_accuracy')} | {m('ambiguous_frac')} | "
              f"{m('ambiguous_bet_rate')} | {m('alarms',0)} | {m('n_burned',0)} | {s['seeds_with_true_burn']} | "
              f"{s['total_false_burns']} | {m('first_true_burn',0)} | {m('AR_final')} | {sp['positive']}/{sp['near_zero']}/{sp['negative']} | "
              f"{m('narrative_rate')} | {m('true_rate')} | {m('ego_gap')} |")
    print()
    for name in ("skeptic/trick/channels", "forgiving/trick/channels", "fool_once/noise/channels",
                 "skeptic/noise/channels", "skeptic/trick/plus-band", "skeptic/noise/channels/AR0=0",
                 "skeptic/high-noise/channels", "fool_once/high-noise/channels"):
        if name not in d["arms"]:
            continue
        runs = d["arms"][name]["runs"]
        r0 = runs[0]
        tr = r0["AR_trace"]
        pts = [tr[i] for i in (0, 4, 9, 24, 49, 99) if i < len(tr)]
        burns = sorted(r["n_burned"] for r in runs)
        print(f"{name}: seed0 AR@200/1k/2k/5k/10k/20k {pts}; burned per seed min/med/max {burns[0]}/{burns[len(burns)//2]}/{burns[-1]}; "
              f"seed0 burned {r0['burned']} true {r0['true_burns']} false {r0['false_burns']} firstTrue {r0['first_true_burn']} firstFalse {r0['first_false_burn']}; "
              f"lying frac {r0['lying_frac']}; forgiving weights {r0['forgiving_weights']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/census2.json")
