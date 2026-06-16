r"""make_figures.py — regenerate the paper figures from the npz metrics.

    python scripts/make_figures.py --metrics metrics --out figures

Currently regenerates:
  fig1_explaw.pdf    — dose response ln T_grok vs ||W|| per modulus (the exponential law)
  fig_collapse.pdf   — data collapse: T_grok vs effective logit scale across the rho x tau grid
The mediation (fig_mediation) and memorization-control (fig_memctrl) figures follow the same
pattern from the tempmed metrics (per-cell T0/T1/tau sweep, and T_mem/T_grok split); see MANIFEST.
"""
import argparse, os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def med_logit_at_grok(d):
    s = np.asarray(d["steps"]).ravel(); ls = np.asarray(d.get("logit_scale", [])).ravel()
    tg = np.asarray(d["T_grok_per_seed"]).ravel(); tg = tg[tg > 0]
    if not len(tg) or not len(ls): return np.nan
    return float(ls[int(np.argmin(np.abs(s - np.median(tg))))])
def med_norm(d):
    w = np.asarray(d.get("wn_at_grok", [])).ravel(); w = w[w > 0]; return float(np.median(w)) if len(w) else np.nan
def med_T(d):
    tg = np.asarray(d["T_grok_per_seed"]).ravel(); tg = tg[tg > 0]
    return (float(np.median(tg)), int(len(tg)), int(np.asarray(d["T_grok_per_seed"]).size))

def fig_explaw(metrics, out):
    by_p = {}
    for f in glob.glob(os.path.join(metrics, "**", "*.npz"), recursive=True):
        b = os.path.basename(f)
        if "_E_t" in b or "_struct" in b or "tau" in b: continue
        m = re.search(r"p(\d+).*clamp_rho([0-9.]+)(?:_addlow)?\.npz", b)
        if not m: continue
        d = np.load(f, allow_pickle=True)
        if str(d.get("loss")) != "ce" or str(d.get("noise", "full")) != "full": continue
        T, ng, S = med_T(d)
        if ng < S / 2: continue
        p = int(m.group(1)); rho = round(float(m.group(2)), 2)
        if rho > 1.15: continue
        by_p.setdefault(p, []).append((med_norm(d), T))
    plt.figure(figsize=(5, 4))
    for p in sorted(by_p):
        pts = sorted(by_p[p]); W = np.array([x[0] for x in pts]); T = np.array([x[1] for x in pts])
        plt.scatter(W, T, s=18, label=f"p={p}")
        a, c = np.polyfit(W, np.log(T), 1); xs = np.linspace(W.min(), W.max(), 50)
        plt.plot(xs, np.exp(a * xs + c), lw=1, alpha=0.6)
    plt.yscale("log"); plt.xlabel(r"$\|W\|$ (held norm)"); plt.ylabel(r"$T_{\mathrm{grok}}$")
    plt.title("Fixed-norm exponential law"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig1_explaw.pdf")); plt.close()
    print("wrote fig1_explaw.pdf")

def fig_collapse(metrics, out):
    cells = []
    for f in glob.glob(os.path.join(metrics, "**", "*.npz"), recursive=True):
        m = re.search(r"p(\d+)_grid_rho([0-9.]+)_tau([0-9.]+)\.npz", os.path.basename(f))
        if not m: continue
        d = np.load(f, allow_pickle=True); T, ng, S = med_T(d)
        if ng < S / 2: continue
        cells.append((int(m.group(1)), float(m.group(2)), med_logit_at_grok(d), T))
    plt.figure(figsize=(5, 4))
    for p, mk in [(59, "o"), (97, "^")]:
        cs = [c for c in cells if c[0] == p]
        if not cs: continue
        L = [c[2] for c in cs]; T = [c[3] for c in cs]; rho = [c[1] for c in cs]
        sc = plt.scatter(L, T, c=rho, marker=mk, s=28, cmap="viridis", label=f"p={p}")
    plt.yscale("log"); plt.xlabel("effective logit scale at grok"); plt.ylabel(r"$T_{\mathrm{grok}}$")
    plt.colorbar(label=r"norm dose $\rho$"); plt.title("Data collapse onto logit scale")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_collapse.pdf")); plt.close()
    print("wrote fig_collapse.pdf")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--metrics", default="metrics"); ap.add_argument("--out", default="figures_regenerated")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    fig_explaw(a.metrics, a.out); fig_collapse(a.metrics, a.out)

if __name__ == "__main__":
    main()
