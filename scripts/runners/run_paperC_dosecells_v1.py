r"""run_paperC_dosecells_v1.py — add the missing low-rho p43 dose cells (SC-free range -> n=7).

WHY: the SC-free refit of Table 1 leaves p=43 with only 4 cells (rho 0.90-1.05) once the
softmax-collapsed cells (rho>=1.10) are excluded. This adds rho in {0.75, 0.80, 0.85} at p=43
so the SC-free range has n=7, restoring model-selection (AIC) power.

COMPARABILITY: reuses run_config from run_paperC_confound_v1 (the v7-matched runner) and PINS
  wc = 50.51311 (the pscan p43 calibration, free wd=1.0 norm-at-grok), so held norm = rho*wc
  matches the existing pscan p43 cells exactly. seeds=8, t_int=500, max_steps=15000, full-batch,
  CE, MLP -- all matched to pscan p43.

BUILT-IN CHECK: it also re-runs rho=0.90 (which pscan already has: T_grok~3612, logit@grok~19.08).
  If this reproduces within seed noise, the 3 NEW cells are trustworthy. If it does NOT, prefer
  re-running with your ORIGINAL pscan/dose runner at rho in {0.75,0.80,0.85} instead.

RUN (put NEXT TO run_paperC_confound_v1.py):
  python run_paperC_dosecells_v1.py --device cuda
  # resume:
  python run_paperC_dosecells_v1.py --device cuda --resume_dir "./runs/paperC_dose_p43_<ts>"

Then send the metrics/ folder; analysis (refit Table 1 p43 with n=7) is done offline.
"""
import argparse, os, json, datetime
import numpy as np
import torch
import run_paperC_confound_v1 as C   # must be importable next to this file

WC_P43 = 50.51311111450195            # pscan p43 calibration (free wd=1.0 norm@grok). DO NOT change.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.75, 0.80, 0.85, 0.90])  # 0.90 = check cell
    ap.add_argument("--seeds", type=int, default=8)            # match pscan p43
    ap.add_argument("--t_int", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=15000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_root", default="./runs")
    ap.add_argument("--resume_dir", default=None)
    a = ap.parse_args()
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"

    if a.resume_dir:
        run_dir = a.resume_dir
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(a.out_root, f"paperC_dose_p43_{ts}")
    paths = {"metrics": os.path.join(run_dir, "metrics"), "Temp": os.path.join(run_dir, "Temp")}
    for q in (run_dir, paths["metrics"], paths["Temp"]):
        os.makedirs(q, exist_ok=True)

    base = dict(loss="ce", noise="full", p=43, d_model=128, H=256, alpha=0.40,
                seeds=a.seeds, lr=1e-3, lam=1.0, t_int=a.t_int,
                log_every=200, eval_every=25, seed0=0, batch_size=739,
                init_scale=1.0, arm="clamp", gamma=0.0, tau=1.0, base_frac=[1/3**0.5]*3)

    ckpt_path = os.path.join(run_dir, "checkpoint.json")
    ckpt = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {"done": [], "summary": []}
    C.log(run_dir, f"start dose p43: rhos={a.rhos} wc(pinned)={WC_P43:.4f} seeds={a.seeds} dev={device}")

    for rho in a.rhos:
        cid = f"mlp_ce_full_p43_clamp_rho{rho:.2f}_addlow"
        if cid in ckpt["done"]:
            continue
        cfg = dict(base, rho=float(rho), max_steps=a.max_steps, config_id=cid)
        ng, tg, out = C.run_config(cfg, WC_P43, [1/3**0.5]*3, paths, device, 5000)
        wn = np.asarray(out["wn_at_grok"]).ravel(); wn = wn[wn > 0]
        held = float(np.median(wn)) if len(wn) else float("nan")
        logit = C._median_logit_at_grok(out)
        ckpt["done"].append(cid)
        ckpt["summary"].append(dict(rho=rho, held_norm=held, logit_at_grok=logit,
                                    Tgrok_med=tg, grok=ng))
        C.atomic_json(ckpt_path, ckpt)
        C.atomic_json(os.path.join(run_dir, "results_summary.json"), ckpt["summary"])
        chk = ""
        if abs(rho - 0.90) < 1e-6:
            chk = "  <-- CHECK vs pscan (expect T~3612, logit~19.08)"
        C.log(run_dir, f"  rho{rho:.2f} held={held:.2f} logit@grok={logit:.2f} T={tg} grok={ng}/{a.seeds}{chk}")

    print("\n  rho   held_norm   logit@grok   T_grok   grok", flush=True)
    for s in sorted(ckpt["summary"], key=lambda s: s["rho"]):
        tag = "  <-- CHECK (pscan: T=3612 logit=19.08)" if abs(s["rho"] - 0.90) < 1e-6 else ""
        print(f"  {s['rho']:.2f}  {s['held_norm']:>9.2f}  {s['logit_at_grok']:>9.2f}  "
              f"{str(s['Tgrok_med']):>7}  {s['grok']}/{a.seeds}{tag}", flush=True)
    print("\nSend the metrics/ folder. If rho=0.90 reproduces pscan, the 3 new cells finalize Table 1 (n=7).", flush=True)
    C.log(run_dir, "DONE dose p43")


if __name__ == "__main__":
    main()
