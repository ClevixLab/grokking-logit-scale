r"""run_paperC_float64audit_v1.py — reproduce the softmax-collapse audit (Table 4).

Re-runs the two highest-dose CE cells at p=59 in BOTH float32 and float64 at the SAME absolute
held norm, and records the softmax-collapse rate (fraction of train points whose correct-class
softmax probability reaches 1.0 in working precision) and the grok time. Paper Table 4:
  rho=1.15: T32~12.75k, T64~12.16k, sc32=0.00, sc64=0.00  -> precision-robust
  rho=1.25: T32~25.4k (sc32=0.31), T64 not grokked by 60k  -> float32 SC-confounded

Reuses model/clamp/data from run_paperC_confound_v1 (must be importable next to this file) so the
float32 arm matches the paper exactly; the float64 arm casts the same graph to double.

RUN (put NEXT TO run_paperC_confound_v1.py):
  python run_paperC_float64audit_v1.py --device cuda
"""
import argparse, os, json, datetime, time
import numpy as np
import torch
import torch.nn.functional as F
import run_paperC_confound_v1 as C

WC_P59 = 54.49          # paper calibration for p=59 (clamp wc). held norm = rho*wc.

def train_prec(p, rho, wc, dtype, seeds, max_steps, t_int, device, eval_every=50):
    torch.manual_seed(0)
    A, Bv, Y = C.make_pairs(p, device); N = A.shape[0]
    pool, train_m, test_m, ntr = C.per_seed_train_pool(seeds, N, 0.40, 0, device)
    all_idx = torch.arange(N, device=device).unsqueeze(0).expand(seeds, N)
    P = C.init_mlp(seeds, p, 128, 256, 0, device, 1.0)
    P = {k: v.to(dtype) for k, v in P.items()}
    for v in P.values(): v.requires_grad_(True)
    keys = list(P.keys()); m = {k: torch.zeros_like(P[k]) for k in keys}; v = {k: torch.zeros_like(P[k]) for k in keys}
    target = rho * wc
    T_grok = np.full(seeds, -1, np.int64); sc_max = np.zeros(seeds)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, max_steps + 1):
        Ai, Bi = C.egidx(A, all_idx), C.egidx(Bv, all_idx)
        logits = C.flogits(P, Ai, Bi, 1.0)
        le = C.loss_per_elem(logits, C.egidx(Y, all_idx) if hasattr(C, 'egidx') else Y[None].expand(seeds, N), p, "ce")
        L = ((le * train_m).sum(1) / train_m.sum(1)).sum()
        for k in keys: P[k].grad = None
        L.backward()
        with torch.no_grad():
            for k in keys:
                g = P[k].grad
                m[k].mul_(b1).add_(g, alpha=1 - b1); v[k].mul_(b2).addcmul_(g, g, value=1 - b2)
                mh = m[k] / (1 - b1 ** t); vh = v[k] / (1 - b2 ** t)
                P[k].addcdiv_(mh, vh.sqrt().add_(eps), value=-1e-3)
                if k in C.WMATS: P[k].add_(P[k], alpha=-1e-3 * 1.0)
            if t >= t_int:                                   # global clamp to fixed absolute norm
                C.clamp_global(P, target)
            if t % eval_every == 0:
                lg = C.flogits(P, C.egidx(A, all_idx), C.egidx(Bv, all_idx), 1.0)
                pr = lg.softmax(-1)
                pred = lg.argmax(-1)
                te = ((pred == Y[None]) & test_m).sum(1).float() / test_m.sum(1).float()
                T_grok[(T_grok < 0) & (te >= 0.90).cpu().numpy()] = t
                pcorr = pr.gather(-1, Y[None].expand(seeds, N).unsqueeze(-1)).squeeze(-1)
                sc = (((pcorr >= 1.0) & train_m).sum(1).float() / ntr).cpu().numpy()
                sc_max = np.maximum(sc_max, sc)
        if (T_grok > 0).all(): break
    tg = T_grok[T_grok > 0]
    return (float(np.median(tg)) if len(tg) else None), float(np.mean(sc_max)), int((T_grok > 0).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=59); ap.add_argument("--rhos", type=float, nargs="+", default=[1.15, 1.25])
    ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--t_int", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=60000); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_root", default="./runs")
    a = ap.parse_args()
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(a.out_root, f"paperC_float64audit_p{a.p}_{ts}"); os.makedirs(run_dir, exist_ok=True)
    rows = []
    print(f"\n{'rho':>5} {'held':>6} {'T32':>8} {'T64':>10} {'sc32':>6} {'sc64':>6}  verdict", flush=True)
    for rho in a.rhos:
        held = rho * WC_P59
        T32, sc32, g32 = train_prec(a.p, rho, WC_P59, torch.float32, a.seeds, a.max_steps, a.t_int, device)
        T64, sc64, g64 = train_prec(a.p, rho, WC_P59, torch.float64, a.seeds, a.max_steps, a.t_int, device)
        verdict = "precision-robust" if (sc32 < 0.05 and T64 and T32 and abs(np.log((T64 or 1)/(T32 or 1))) < np.log(1.3)) \
                  else "float32 SC-confounded"
        rows.append(dict(rho=rho, held=held, T32=T32, T64=T64, sc32=sc32, sc64=sc64,
                         grok32=g32, grok64=g64, verdict=verdict))
        print(f"{rho:>5.2f} {held:>6.1f} {str(T32):>8} {str(T64):>10} {sc32:>6.2f} {sc64:>6.2f}  {verdict}", flush=True)
    json.dump(rows, open(os.path.join(run_dir, "float64_audit.json"), "w"), indent=1, default=float)
    print(f"\nSaved -> {run_dir}/float64_audit.json . Send it; it fills Table 4.", flush=True)

if __name__ == "__main__":
    main()
