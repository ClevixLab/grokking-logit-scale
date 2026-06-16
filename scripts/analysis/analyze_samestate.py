r"""analyze_samestate.py — reproduce the §4.6 same-state numbers (Table: same-state arms).

Reads metrics/samestate/{p59,p97}_{free,clamp_same,clamp_raised,clamp_lowered}.npz and reports, per
modulus: the median delay-from-fork of each arm, the decisive clamp_raised/clamp_same ratio with a paired
bootstrap 95% CI over the 12 seeds, and the per-seed monotonicity count (lowered < same < raised).

Run:
    python scripts/analysis/analyze_samestate.py            # from repo root
    python scripts/analysis/analyze_samestate.py --metrics metrics/samestate
"""
import argparse, glob, os, json, numpy as np

def load(metdir, p, arm):
    d = np.load(os.path.join(metdir, f"{p}_{arm}.npz"), allow_pickle=True)
    tg = d["T_grok_per_seed"].astype(float); tf = int(d["t_fork"])
    return tg, tf

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--metrics", default="metrics/samestate")
    ap.add_argument("--out", default="reproduced_samestate.json"); a = ap.parse_args()
    rng = np.random.default_rng(0); arms = ["clamp_lowered", "free", "clamp_same", "clamp_raised"]
    out = {}
    for p in ["p59", "p97"]:
        if not os.path.exists(os.path.join(a.metrics, f"{p}_free.npz")):
            continue
        D = {arm: load(a.metrics, p, arm) for arm in arms}
        tf = D["free"][1]
        delays = {arm: (tg - tf) for arm, (tg, tf_) in D.items()}
        meds = {arm: float(np.median(delays[arm][D[arm][0] > 0])) for arm in arms}
        same, raised, lowered = D["clamp_same"][0], D["clamp_raised"][0], D["clamp_lowered"][0]
        both = (same > 0) & (raised > 0); idx = np.where(both)[0]
        ds, dr, dl = delays["clamp_same"], delays["clamp_raised"], delays["clamp_lowered"]
        point = np.median(dr[idx]) / np.median(ds[idx])
        boots = [np.median(dr[rng.choice(idx, len(idx), replace=True)]) /
                 np.median(ds[rng.choice(idx, len(idx), replace=True)]) for _ in range(4000)]
        # paired resample (same indices for num and den)
        boots = []
        for _ in range(4000):
            bi = rng.choice(idx, len(idx), replace=True)
            boots.append(np.median(dr[bi]) / np.median(ds[bi]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        mono = int((((dl < ds) & (ds < dr)) & both).sum())
        out[p] = dict(t_fork=tf, delays=meds, ratio_raised_over_same=round(float(point), 3),
                      ci=[round(float(lo), 3), round(float(hi), 3)],
                      monotonic_seeds=f"{mono}/{int(both.sum())}",
                      all_grok={arm: int((D[arm][0] > 0).sum()) for arm in arms})
        print(f"\n=== {p} (fork t={tf}) ===")
        for arm in arms:
            print(f"  {arm:14s} delay={meds[arm]:>7.0f}  grok={out[p]['all_grok'][arm]}/12")
        print(f"  RATIO raised/same = {point:.2f}  95% CI [{lo:.2f},{hi:.2f}]  monotonic {mono}/{int(both.sum())}")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\nWrote {a.out}")

if __name__ == "__main__":
    main()
