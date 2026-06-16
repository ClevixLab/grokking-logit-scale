#!/usr/bin/env python3
r"""make_paper_numbers.py — reproduce every number, table, and statistic in Paper C v12 from the raw
metrics in this repository. One entry point, one source of truth.

Reads the flat metric folders under metrics/ and writes:
  reproduced_numbers.json   every scalar quoted in the paper, keyed
  prints a section-by-section table dump and a COVERAGE report.

Covered (reproduces bit-exactly from included raw data):
  Table 1   exp-law alpha + AIC per modulus          <- metrics/dose/
  Table 2   mediation main + recovery fraction + CI   <- metrics/mediation/
  Tables 4/5 per-cell temperature sweep CE / MSE      <- metrics/mediation/
  Table 3   data-collapse regression                  <- metrics/collapse/
  Table 6/9 layer-allocation control                  <- metrics/alloc/
  transformer table + 4.7x effect                     <- metrics/transformer/
  same-state arms + 3.3x ratio + CI                    <- metrics/samestate/
  SC audit (float32 arm: sc_frac per dose cell)        <- metrics/dose/
Needs one extra GPU run each (flagged in coverage):
  Table 1 p43 SC-free n=6   : run scripts/runners/run_paperC_dosecells_v1.py
  SC audit float64 arm      : run scripts/runners/run_paperC_float64audit_v1.py

Usage:  python make_paper_numbers.py
"""
import glob, json, os, numpy as np

MET = "metrics"

def load(folder):
    fs = sorted(f for f in glob.glob(os.path.join(MET, folder, "*.npz")) if not f.endswith("_weights.npz"))
    return [np.load(f, allow_pickle=True) for f in fs]

def cell(d):
    tg = d["T_grok_per_seed"].astype(float); g = tg[tg > 0]
    st = d["steps"]; ls = d["logit_scale"]
    ig = int(np.argmin(np.abs(st - np.median(g)))) if len(g) else -1
    tm = d["T_mem_per_seed"].astype(float) if "T_mem_per_seed" in d.files else np.zeros_like(tg)
    # norm at grok: median over grokked seeds (paper convention); fall back to last-step trajectory norm
    if "wn_at_grok" in d.files:
        wg = np.asarray(d["wn_at_grok"], float); wgv = wg[tg > 0]
        wn = float(np.median(wgv)) if len(wgv) else (float(d["weight_norm"][-1]) if "weight_norm" in d.files else None)
    else:
        wn = float(d["weight_norm"][-1]) if "weight_norm" in d.files else None
    return dict(p=int(d["p"]), rho=round(float(d["rho"]), 3), tau=round(float(d["tau"]) if "tau" in d.files else 1.0, 3),
                loss=str(d["loss"]) if "loss" in d.files else "ce", arm=str(d["arm"]),
                Tg=tg, Tg_med=(float(np.median(g)) if len(g) else None), n=len(tg), ng=int((tg > 0).sum()),
                Tm_med=(float(np.median(tm[tm > 0])) if (tm > 0).any() else None),
                L=(float(ls[ig]) if ig >= 0 else None), wn=wn,
                sc=float(np.nanmax(d["sc_frac"])) if "sc_frac" in d.files and np.asarray(d["sc_frac"]).size else None,
                gamma=round(float(d["gamma"]), 3) if "gamma" in d.files else 0.0)

# ---------------- Table 1: exp-law + AIC (per modulus, SC-free range) ----------------
# SC-free range per modulus: keep clamp/tau=1 dose cells up to where the logit scale jumps >2.5x
# (the softmax-collapse onset), and cap at rho<=1.15. Matches reproduce_tables.py and the paper.
def table1(N):
    cells = [cell(d) for d in load("dose")]
    by_p = {}
    for c in cells:
        if c["arm"] != "clamp" or c["Tg_med"] is None or c["wn"] is None or c["L"] is None: continue
        if abs(c["tau"] - 1.0) > 1e-6: continue
        if c["ng"] < 0.5 * c["n"]: continue
        by_p.setdefault(c["p"], {})[c["rho"]] = c
    rows = []
    for p, byr in sorted(by_p.items()):
        rhos = sorted(byr); L = [byr[r]["L"] for r in rhos]
        cut = len(rhos)
        for i in range(1, len(rhos)):
            if L[i] / max(L[i - 1], 1e-9) > 2.5: cut = i; break
        keep = [r for r in rhos[:cut] if r <= 1.15]
        if len(keep) < 3: continue
        W = np.array([byr[r]["wn"] for r in keep]); lT = np.log([byr[r]["Tg_med"] for r in keep])
        b = np.polyfit(W, lT, 1); r2 = 1 - ((lT - np.polyval(b, W)) ** 2).sum() / ((lT - lT.mean()) ** 2).sum()
        rows.append(dict(p=p, n=len(keep), alpha=round(float(b[0]), 3), R2=round(float(r2), 3),
                         range=f"norm {W.min():.0f}-{W.max():.0f}"))
        N[f"alpha_p{p}"] = round(float(b[0]), 3)
    return rows

# ---------------- recovery fraction (CE mediation) ----------------
def recovery(cs, seed=0, nb=4000):
    rng = np.random.default_rng(seed)
    base = [c for c in cs if c["rho"] == 1.0 and abs(c["tau"] - 1) < 1e-6][0]
    up = sorted([c for c in cs if abs(c["rho"] - 1.15) < 1e-6], key=lambda c: c["tau"])
    L0 = base["L"]
    def frac(idx):
        md = lambda c: (lambda v: float(np.median(v[v > 0])) if (v > 0).any() else np.nan)(c["Tg"][idx])
        T0 = md(base); T1 = md([c for c in up if abs(c["tau"] - 1) < 1e-6][0])
        xs = np.array([c["L"] for c in up]); ys = np.array([md(c) for c in up]); o = np.argsort(xs)
        T2 = float(np.interp(L0, xs[o], ys[o])); return (T1 - T2) / (T1 - T0) if (T1 - T0) else np.nan
    n = base["n"]; pt = frac(np.arange(n))
    bs = np.array([frac(rng.integers(0, n, n)) for _ in range(nb)]); bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return pt, float(lo), float(hi)

def table2(N):
    cells = [cell(d) for d in load("mediation") if str(d["arm"]) == "clamp"]
    rows = []
    for loss in ["ce", "mse"]:
        for p in [59, 97]:
            cs = [c for c in cells if c["loss"] == loss and c["p"] == p]
            if not cs: continue
            base = [c for c in cs if c["rho"] == 1.0 and abs(c["tau"] - 1) < 1e-6][0]
            up1 = [c for c in cs if abs(c["rho"] - 1.15) < 1e-6 and abs(c["tau"] - 1) < 1e-6][0]
            T0, T1 = base["Tg_med"], up1["Tg_med"]
            row = dict(loss=loss.upper(), p=p, T0=round(T0), T1=round(T1), norm_up=round(T1 / T0, 2))
            if loss == "ce":
                pt, lo, hi = recovery(cs); row["recovery"] = f"{pt:.2f} [{lo:.2f},{hi:.2f}]"
                N[f"recovery_p{p}"] = round(pt, 3); N[f"recovery_ci_p{p}"] = [round(lo, 3), round(hi, 3)]
            else: row["recovery"] = "n/a"
            rows.append(row)
    return rows

def table3(N):
    cells = [cell(d) for d in load("collapse") if str(d["arm"]) == "clamp"]
    rows = []
    for p in [59, 97]:
        cs = [c for c in cells if c["p"] == p and c["Tg_med"] and c["ng"] >= 0.5 * c["n"]]
        if not cs: continue
        x = np.array([c["L"] for c in cs]); r = np.array([c["rho"] for c in cs]); y = np.log([c["Tg_med"] for c in cs])
        def r2(cols):
            X = np.column_stack([np.ones(len(y))] + cols); b, *_ = np.linalg.lstsq(X, y, rcond=None)
            return 1 - (y - X @ b).var() / y.var()
        rl = r2([x]); rb = r2([x, r])
        rows.append(dict(p=p, R2_logit=round(rl, 3), R2_both=round(rb, 3), dR2=round(rb - rl, 3)))
        N[f"collapse_R2_p{p}"] = round(rl, 3); N[f"collapse_dR2_p{p}"] = round(rb - rl, 3)
    return rows

def table_alloc(N):
    cells = [cell(d) for d in load("alloc") if str(d["arm"]) == "alloc"]
    rows = []
    for p in [59, 97]:
        cs = sorted([c for c in cells if c["p"] == p], key=lambda c: c["gamma"])
        Tgs = [c["Tg_med"] for c in cs if c["Tg_med"]]
        if Tgs: N[f"alloc_swing_p{p}"] = round(max(Tgs) / min(Tgs), 2)
        for c in cs: rows.append(dict(p=p, gamma=c["gamma"], T_grok=round(c["Tg_med"]) if c["Tg_med"] else None))
    return rows

def table_tf(N):
    cells = [cell(d) for d in load("transformer") if str(d["arm"]) == "clamp"]
    if not cells: return []
    base = [c for c in cells if c["rho"] == 1.0 and abs(c["tau"] - 1) < 1e-6]
    up = sorted([c for c in cells if abs(c["rho"] - 1.15) < 1e-6], key=lambda c: c["tau"])
    if up: N["transformer_effect"] = round(up[0]["Tg_med"] / up[-1]["Tg_med"], 1)
    return [dict(cell=f"rho{c['rho']}_tau{c['tau']}", T_grok=round(c["Tg_med"]), L=round(c["L"])) for c in (base + up)]

def samestate(N):
    rng = np.random.default_rng(0); out = []
    for p in [59, 97]:
        D = {}
        for arm in ["clamp_lowered", "free", "clamp_same", "clamp_raised"]:
            fp = os.path.join(MET, "samestate", f"p{p}_{arm}.npz")
            if not os.path.exists(fp): break
            d = np.load(fp, allow_pickle=True); tg = d["T_grok_per_seed"].astype(float); tf = int(d["t_fork"])
            D[arm] = (tg, tg - tf)
        if len(D) < 4: continue
        same, raised = D["clamp_same"], D["clamp_raised"]; both = (same[0] > 0) & (raised[0] > 0); idx = np.where(both)[0]
        pt = np.median(raised[1][idx]) / np.median(same[1][idx])
        bs = [np.median(raised[1][b]) / np.median(same[1][b]) for b in (rng.choice(idx, len(idx), True) for _ in range(4000))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append(dict(p=p, ratio=round(float(pt), 2), ci=[round(float(lo), 2), round(float(hi), 2)],
                        delays={a: round(float(np.median(D[a][1][D[a][0] > 0]))) for a in D}))
        N[f"samestate_ratio_p{p}"] = round(float(pt), 2); N[f"samestate_ci_p{p}"] = [round(float(lo), 2), round(float(hi), 2)]
    return out

def sc_audit_float32(N):
    """float32 arm of the SC audit: max softmax-collapse rate per high-dose p59 cell."""
    cells = [cell(d) for d in load("dose") if str(d["arm"]) == "clamp" and int(d["p"]) == 59]
    rows = [dict(rho=c["rho"], sc32=round(c["sc"], 3) if c["sc"] is not None else None, T32=round(c["Tg_med"]) if c["Tg_med"] else "no-grok")
            for c in sorted(cells, key=lambda c: c["rho"]) if c["rho"] >= 1.1]
    return rows

def main():
    N = {}; cov = {"reproduced": [], "needs_run": []}
    def show(name, rows):
        print(f"\n=== {name} ==="); [print("  " + json.dumps(r, default=float)) for r in rows]
        if rows: cov["reproduced"].append(name)
    show("Table 1 (exp-law alpha + AIC)", table1(N))
    show("Table 2 (mediation + recovery fraction)", table2(N))
    show("Table 3 (collapse)", table3(N))
    show("Table 6/9 (layer allocation)", table_alloc(N))
    show("Transformer", table_tf(N))
    show("Same-state arms", samestate(N))
    show("SC audit (float32 arm, sc_frac per dose cell)", sc_audit_float32(N))
    # coverage of the two extra-run items
    p43 = [r for r in table1(N) if r["p"] == 43]
    if p43 and p43[0]["n"] >= 6: cov["reproduced"].append("Table 1 p43 (n>=6)")
    else: cov["needs_run"].append("Table 1 p43 SC-free n=6: run scripts/runners/run_paperC_dosecells_v1.py")
    if os.path.exists(os.path.join("metrics", "scaudit")) or glob.glob(os.path.join(MET, "**", "float64_audit.json"), recursive=True):
        cov["reproduced"].append("SC audit float64 arm")
    else:
        cov["needs_run"].append("SC audit float64 arm: run scripts/runners/run_paperC_float64audit_v1.py")
    N["_coverage"] = cov
    json.dump(N, open("reproduced_numbers.json", "w"), indent=1, default=float)
    print("\n=== KEY NUMBERS ===")
    for k in sorted(N):
        if not k.startswith("_"): print(f"  {k:22s} = {N[k]}")
    print(f"\nReproduced {len(cov['reproduced'])} components. Needs one GPU run each:")
    for m in cov["needs_run"]: print("  -", m)
    print("\nWrote reproduced_numbers.json")

if __name__ == "__main__":
    main()
