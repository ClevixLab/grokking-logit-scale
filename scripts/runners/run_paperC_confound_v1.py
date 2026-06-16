r"""Paper C — clamp-confound control (Arm A: layer allocation; Arm B: temperature mediation).

WHY THIS EXISTS (read before touching):
  The fixed-norm clamp rescales E, W1, W2 by ONE scalar to hold ||W|| fixed. That single
  operation does two things at once: it sets the scalar norm, AND it changes the function
  (chiefly the logit scale, since logits grow with the readout norm). So "raising the held
  norm lengthens the delay" does not yet isolate ||W|| (the scalar) from the readout-norm /
  logit-scale that rescaling drags along. This script closes that confound by two
  interventions that move the *internals* while holding the *scalar* ||W|| fixed.

  ARM A  (--mode alloc):  at fixed total ||W|| = wc (rho = 1.0), shift mass between E and W2
     (gamma sweep). Logit scale ~ ||W2||, so moving E<->W2 at fixed TOTAL norm changes the
     logit scale WITHOUT changing ||W||.
        If the SCALAR ||W|| gates the delay  -> T_grok is flat in gamma.
        If readout-norm / logit-scale gates  -> T_grok varies with gamma.

  ARM B  (--mode tempmed):  hold ||W|| via the global clamp, and add a NON-trainable
     temperature tau that divides the logits before the softmax/loss. tau is NOT counted in
     ||W|| (it acts only in the forward), so at a CLAMPED ||W|| it tunes the *effective*
     logit scale independently of the norm. Mediation design:
        baseline    : rho=1.00, tau=1            -> T0
        norm-up     : rho=1.15, tau=1            -> T1  (the known effect)
        compensated : rho=1.15, tau=tau* (>1)    -> T2  (logit scale matched back to baseline)
        If norm effect is MEDIATED by logit scale -> T2 ~ T0.
        If ||W|| matters BEYOND logit scale       -> T2 ~ T1.
     We sweep a small tau grid at rho=1.15 and let analysis pick the tau matching baseline
     logit scale; we report T2 by interpolation on measured logit scale.

PRE-REGISTRATION (frozen before running; also dumped to preregistration.json):
  H1: at fixed total ||W||, T_grok is invariant to gamma (Arm A) and, after matching logit
      scale, the norm-up delay is NOT recovered by tau-compensation (Arm B): T2 stays near T1.
  Falsifiers (either => the scalar ||W|| is NOT the load-bearing variable; the mediator is
      readout-norm / logit-scale):
        (A) median T_grok changes by > 2x across the gamma sweep at fixed ||W||; or
        (B) the logit-scale-matched compensated cell returns within 1.3x of baseline T0
            (i.e. |log T2 - log T0| < log 1.3 while |log T1 - log T0| > log 2).
  Both outcomes are reportable and honest. H1 holds -> "norm causal even when logit scale and
  layer allocation are controlled" (closes the confound). H1 fails -> identifies the mediator
  (logit scale), reframes the claim around it, and pre-empts the reviewer.

ENGINEERING (matches run_paperC_v7): timestamped dir under --out_root; atomic writes; resume
  EXACTLY at the interrupted step (config-level checkpoint.json + within-config Temp/state.pt);
  never overwrite; npz metric keys compatible with the v7 outputs. MLP only (the paper's main
  setting); CE loss (the regime where logit scale / softmax curvature is in play).

Run (single CUDA GPU, Python 3.11):
  # Arm A (layer allocation) — two moduli:
  python run_paperC_confound_v1.py --mode alloc   --p 59 --device cuda
  python run_paperC_confound_v1.py --mode alloc   --p 97 --device cuda
  # Arm B (temperature mediation):
  python run_paperC_confound_v1.py --mode tempmed --p 59 --device cuda
  python run_paperC_confound_v1.py --mode tempmed --p 97 --device cuda
"""
import argparse, json, os, sys, math, time, datetime, glob
import numpy as np
import torch
import torch.nn.functional as F

# ============================ infra ===========================================
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def atomic_bytes(path, fn):
    with open(path + ".tmp", "wb") as f: fn(f)
    os.replace(path + ".tmp", path)

def atomic_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)

def log(run_dir, msg):
    line = f"[{now()}] {msg}"; print(line, flush=True)
    with open(os.path.join(run_dir, "run.log"), "a") as f: f.write(line + "\n")

# ============================ data ============================================
def make_pairs(p, device):
    a = torch.arange(p, device=device).repeat_interleave(p)
    b = torch.arange(p, device=device).repeat(p)
    return a, b, (a + b) % p

def per_seed_train_pool(S, N, alpha, seed0, device):
    ntr = int(alpha * N)
    pool = torch.empty(S, ntr, dtype=torch.long, device=device)
    train_m = torch.zeros(S, N, dtype=torch.bool, device=device)
    for s in range(S):
        g = torch.Generator().manual_seed(1000 + seed0 + s)
        perm = torch.randperm(N, generator=g)[:ntr]
        pool[s] = perm.to(device); train_m[s, perm.to(device)] = True
    return pool, train_m, (~train_m), ntr

def sample_minibatch(pool, B, seed_s, t):
    S, ntr = pool.shape
    out = torch.empty(S, B, dtype=torch.long, device=pool.device)
    for s in range(S):
        g = torch.Generator().manual_seed((seed_s + s) * 1_000_003 + t)
        pos = torch.randint(0, ntr, (B,), generator=g)
        out[s] = pool[s, pos.to(pool.device)]
    return out

# ============================ params / arch (MLP) =============================
WMATS = ["E", "W1", "W2"]
GROUPS = {"E": ["E"], "W1": ["W1"], "W2": ["W2"]}

def init_mlp(S, p, d, H, seed, device, iscale=1.0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *sh, fan: (iscale * torch.randn(*sh, generator=g) / math.sqrt(fan)).to(device)
    P = dict(E=rn(S, p, d, fan=d), W1=rn(S, 2 * d, H, fan=2 * d),
             b1=torch.zeros(S, H, device=device), W2=rn(S, H, p, fan=H),
             b2=torch.zeros(S, p, device=device))
    for k in P: P[k].requires_grad_(True)
    return P

def grp_norm(P, keys):
    sq = sum((P[k] ** 2).flatten(1).sum(1) for k in keys); return torch.sqrt(sq)

def total_wnorm(P): return grp_norm(P, WMATS)

def egather(E, idx):
    S, M = idx.shape; d = E.shape[-1]
    return torch.gather(E, 1, idx.unsqueeze(-1).expand(S, M, d))

def egidx(vec, idx): return vec[idx]

def fwd_mlp(P, Ai, Bi):
    x = torch.cat([egather(P["E"], Ai), egather(P["E"], Bi)], -1)
    h = F.gelu(torch.einsum("smk,skh->smh", x, P["W1"]) + P["b1"][:, None])
    return torch.einsum("smh,shp->smp", h, P["W2"]) + P["b2"][:, None]

# --- temperature wrapper: divides logits by TAU (Arm B). TAU=1.0 => identical to v7. ---
# TAU is NOT a parameter and is NOT in ||W||; it only rescales the forward logits.
def flogits(P, Ai, Bi, tau):
    lg = fwd_mlp(P, Ai, Bi)
    return lg if tau == 1.0 else lg / tau

def loss_per_elem(logits, Yi, p, loss):
    if loss == "ce":
        lp = F.log_softmax(logits, -1)
        return -lp.gather(-1, Yi.unsqueeze(-1)).squeeze(-1)
    return ((logits - F.one_hot(Yi, p).float()) ** 2).sum(-1)

def struct_measure(P, p):                                # top-5 Fourier power fraction (S,)
    E = P["E"]
    pw = torch.fft.rfft(E, dim=1).abs().pow(2).sum(-1)[:, 1:]      # drop DC
    return torch.topk(pw, min(5, pw.shape[1]), dim=1).values.sum(1) / (pw.sum(1) + 1e-12)

# ============================ clamps ==========================================
def alloc_targets(total, gamma, base_frac):
    """Per-layer target norms at FIXED total ||W|| = `total`, shifting E<->W2 by gamma.
    base_frac = (fE, fW1, fW2) with fE^2+fW1^2+fW2^2 = 1 (measured at the control grok).
    W1 stays at its base share; the E+W2 budget is re-split, giving W2 a fraction
    theta(gamma) = clip(theta0*(1+gamma), 0.02, 0.98) of the E+W2 power, where
    theta0 = fW2^2/(fE^2+fW2^2). gamma>0 => more mass in W2 => higher logit scale."""
    fE, fW1, fW2 = base_frac
    nW1 = fW1 * total
    rem2 = max(total ** 2 - nW1 ** 2, 1e-8)
    theta0 = (fW2 ** 2) / (fE ** 2 + fW2 ** 2 + 1e-12)
    theta = min(max(theta0 * (1.0 + gamma), 0.02), 0.98)
    nW2 = math.sqrt(theta * rem2)
    nE = math.sqrt((1.0 - theta) * rem2)
    return {"E": nE, "W1": nW1, "W2": nW2}

@torch.no_grad()
def clamp_global(P, target_total):
    sc = target_total / (total_wnorm(P) + 1e-12)
    for k in WMATS: P[k].mul_(sc.view(-1, *([1] * (P[k].dim() - 1))))

@torch.no_grad()
def clamp_alloc(P, targets):                              # per-layer projection (same target all seeds)
    for k in WMATS:
        cur = torch.sqrt((P[k] ** 2).flatten(1).sum(1))   # (S,)
        sc = targets[k] / (cur + 1e-12)
        P[k].mul_(sc.view(-1, *([1] * (P[k].dim() - 1))))

# ============================ one config ======================================
def run_config(cfg, wc, base_frac, paths, device, ckpt_every):
    loss = cfg["loss"]; p, d, H = cfg["p"], cfg["d_model"], cfg["H"]
    S, alpha, lr, lam = cfg["seeds"], cfg["alpha"], cfg["lr"], cfg["lam"]
    arm, rho, gamma, tau, t_int = cfg["arm"], cfg["rho"], cfg["gamma"], cfg["tau"], cfg["t_int"]
    steps, log_every, eval_every, seed0, B = (cfg["max_steps"], cfg["log_every"],
                                              cfg["eval_every"], cfg["seed0"], cfg["batch_size"])
    noise = cfg["noise"]
    state = os.path.join(paths["Temp"], cfg["config_id"] + "_state.pt")

    A, Bv, Y = make_pairs(p, device); N = A.shape[0]
    pool, train_m, test_m, ntr = per_seed_train_pool(S, N, alpha, seed0, device)
    all_idx = torch.arange(N, device=device).unsqueeze(0).expand(S, N)
    Bfull = ntr if noise == "full" else min(B, ntr)

    total_target = rho * wc
    alloc_tg = alloc_targets(total_target, gamma, base_frac) if arm == "alloc" else None

    P = init_mlp(S, p, d, H, seed0, device, cfg.get("init_scale", 1.0))
    keys = list(P.keys())
    m = {k: torch.zeros_like(P[k]) for k in keys}; v = {k: torch.zeros_like(P[k]) for k in keys}
    t0 = 0; T_mem = np.full(S, -1, np.int64); T_grok = np.full(S, -1, np.int64)
    nb = np.zeros(S, np.float32); na = np.zeros(S, np.float32)
    BUFK = ["steps", "train_loss", "test_loss", "train_acc", "test_acc", "weight_norm",
            "logit_scale", "frac_saturated", "structure", "sc_frac"]
    buf = {k: [] for k in BUFK}
    for gk in GROUPS: buf[f"wn_{gk}"] = []

    if os.path.exists(state):                              # exact within-config resume
        st = torch.load(state, map_location=device, weights_only=False)
        for k in keys:
            P[k].data = st["P"][k].to(device); m[k] = st["m"][k].to(device); v[k] = st["v"][k].to(device)
        t0, T_mem, T_grok, buf, nb, na = (st["t"], st["T_mem"], st["T_grok"], st["buf"], st["nb"], st["na"])

    b1e, b2e, eps = 0.9, 0.999, 1e-8; tw = time.time()
    for t in range(t0 + 1, steps + 1):
        if noise == "full":
            Ai, Bi, Yi = egidx(A, all_idx), egidx(Bv, all_idx), egidx(Y, all_idx); tmask = train_m
        else:
            mb = sample_minibatch(pool, Bfull, seed0, t)
            Ai, Bi, Yi = egidx(A, mb), egidx(Bv, mb), egidx(Y, mb); tmask = None
        logits = flogits(P, Ai, Bi, tau)
        le = loss_per_elem(logits, Yi, p, loss)
        L = (((le * tmask).sum(1) / tmask.sum(1)).sum() if tmask is not None else le.mean(1).sum())
        for k in keys: P[k].grad = None
        L.backward()
        with torch.no_grad():
            for k in keys:
                g = P[k].grad
                m[k].mul_(b1e).add_(g, alpha=1 - b1e); v[k].mul_(b2e).addcmul_(g, g, value=1 - b2e)
                mh = m[k] / (1 - b1e ** t); vh = v[k] / (1 - b2e ** t)
                P[k].addcdiv_(mh, vh.sqrt().add_(eps), value=-lr)
                if k in WMATS: P[k].add_(P[k], alpha=-lr * lam)
            if arm in ("clamp", "alloc") and t == t_int: nb = total_wnorm(P).cpu().numpy().copy()
            if arm == "clamp" and t >= t_int: clamp_global(P, total_target)
            if arm == "alloc" and t >= t_int: clamp_alloc(P, alloc_tg)
            if arm in ("clamp", "alloc") and t == t_int: na = total_wnorm(P).cpu().numpy().copy()

            if t % eval_every == 0 or t == 1:
                lg = flogits(P, egidx(A, all_idx), egidx(Bv, all_idx), tau)
                pred = lg.argmax(-1)
                tr = ((pred == Y[None]) & train_m).sum(1).float() / ntr
                te = ((pred == Y[None]) & test_m).sum(1).float() / test_m.sum(1).float()
                T_mem[(T_mem < 0) & (tr >= 0.99).cpu().numpy()] = t
                T_grok[(T_grok < 0) & (te >= 0.90).cpu().numpy()] = t
            if t % log_every == 0 or t == 1:
                lg = flogits(P, egidx(A, all_idx), egidx(Bv, all_idx), tau)
                pr = lg.softmax(-1) if loss == "ce" else None
                le_full = loss_per_elem(lg, Y[None].expand(S, N), p, loss)
                pred = lg.argmax(-1)
                tr = ((pred == Y[None]) & train_m).sum(1).float() / ntr
                te = ((pred == Y[None]) & test_m).sum(1).float() / test_m.sum(1).float()
                buf["steps"].append(t)
                buf["train_loss"].append(((le_full * train_m).sum(1) / ntr).mean().item())
                buf["test_loss"].append(((le_full * test_m).sum(1) / test_m.sum(1)).mean().item())
                buf["train_acc"].append(tr.mean().item()); buf["test_acc"].append(te.mean().item())
                buf["weight_norm"].append(total_wnorm(P).mean().item())
                buf["logit_scale"].append(lg.norm(dim=-1).mean().item())   # KEY readout for Arm B
                buf["frac_saturated"].append((pr.max(-1).values > 0.999).float().mean().item() if pr is not None else 0.0)
                if pr is not None:
                    pcorr = pr.gather(-1, Y[None].expand(S, N).unsqueeze(-1)).squeeze(-1)
                    buf["sc_frac"].append((((pcorr >= 1.0) & train_m).sum(1).float() / ntr).mean().item())
                else:
                    buf["sc_frac"].append(0.0)
                buf["structure"].append(struct_measure(P, p).median().item())
                for gk, gks in GROUPS.items(): buf[f"wn_{gk}"].append(grp_norm(P, gks).mean().item())
                if t % (log_every * 20) == 0 or t == 1:
                    print(f"   .. {cfg['config_id']} t={t} test_acc={buf['test_acc'][-1]:.2f} "
                          f"norm={buf['weight_norm'][-1]:.1f} logit={buf['logit_scale'][-1]:.1f} "
                          f"wnW2={buf['wn_W2'][-1]:.1f}", flush=True)
            if t % ckpt_every == 0:
                torch.save(dict(P={k: P[k].detach().cpu() for k in keys},
                                m={k: m[k].cpu() for k in keys}, v={k: v[k].cpu() for k in keys},
                                t=t, T_mem=T_mem, T_grok=T_grok, buf=buf, nb=nb, na=na),
                           state + ".tmp"); os.replace(state + ".tmp", state)
        if (T_grok > 0).all(): break

    wf = total_wnorm(P).detach().cpu().numpy()
    # per-layer norms at grok (median over seeds) for base_frac calibration on the control cell
    pl = {gk: float(np.median([buf[f"wn_{gk}"][-1]])) for gk in GROUPS} if buf["steps"] else {}
    out = dict(p=p, alpha=alpha, d_model=d, H=H, lam=lam, lr=lr, arm=arm, rho=rho, gamma=gamma, tau=tau,
               t_int=t_int, wc_used=float(wc), max_steps=steps, n_train=ntr, n_test=int(N - ntr),
               loss=loss, arch="mlp", noise=noise, batch_size=Bfull,
               diverged=bool(not np.isfinite(wf).all()), norm_before_int=nb, norm_after_int=na,
               wn_at_grok=wf.astype(np.float32), T_mem_per_seed=T_mem, T_grok_per_seed=T_grok,
               steps_run=t, wall_s=time.time() - tw,
               base_frac=np.array(cfg.get("base_frac", [0, 0, 0]), dtype=np.float32))
    for k, val in buf.items(): out[k] = np.array(val, dtype=np.float32)
    atomic_bytes(os.path.join(paths["metrics"], cfg["config_id"] + ".npz"),
                 lambda f: np.savez_compressed(f, **out))
    if os.path.exists(state): os.remove(state)
    ng = int((T_grok > 0).sum()); tg = T_grok[T_grok > 0]
    return ng, (float(np.median(tg)) if len(tg) else float("nan")), out

# ============================ analysis ========================================
def _median_logit_at_grok(d):
    """median logit_scale near the grok step (or last logged) for a cell."""
    st = d["steps"]; ls = d["logit_scale"]
    tg = d["T_grok_per_seed"]; tg = tg[tg > 0]
    if not len(st): return float("nan")
    if len(tg):
        ig = int(np.argmin(np.abs(st - np.median(tg))))
    else:
        ig = len(st) - 1
    return float(ls[ig])

def analyze(paths, run_dir, mode):
    cells = []
    for f in sorted(glob.glob(os.path.join(paths["metrics"], "*.npz"))):
        d = np.load(f, allow_pickle=True)
        tg = d["T_grok_per_seed"]; tg = tg[tg > 0]
        ng = int((d["T_grok_per_seed"] > 0).sum()); S = d["T_grok_per_seed"].shape[0]
        cells.append(dict(config_id=os.path.basename(f)[:-4], arm=str(d["arm"]),
                          rho=float(d["rho"]), gamma=float(d["gamma"]), tau=float(d["tau"]),
                          n_grok=ng, n_seeds=S,
                          Tgrok_med=(float(np.median(tg)) if len(tg) else None),
                          logit_at_grok=_median_logit_at_grok(d),
                          wnW2_last=float(d["wn_W2"][-1]) if len(d["wn_W2"]) else None,
                          wn_last=float(d["weight_norm"][-1]) if len(d["weight_norm"]) else None,
                          sc_frac_last=float(d["sc_frac"][-1]) if len(d["sc_frac"]) else None))
    res = {"mode": mode, "cells": cells}

    if mode == "alloc":
        a = sorted([c for c in cells if c["arm"] == "alloc" and c["Tgrok_med"]], key=lambda c: c["gamma"])
        if len(a) >= 3:
            T = [c["Tgrok_med"] for c in a]
            res["alloc_gamma"] = [c["gamma"] for c in a]
            res["alloc_Tgrok"] = T
            res["alloc_logit_scale"] = [c["logit_at_grok"] for c in a]
            res["alloc_T_ratio_maxmin"] = float(max(T) / min(T))
            res["FALSIFIER_A_triggered"] = bool(max(T) / min(T) > 2.0)
            res["verdict"] = ("mediator: T_grok varies with layer allocation at fixed ||W|| "
                              "(scalar norm NOT sole driver)") if res["FALSIFIER_A_triggered"] else \
                             ("H1 supported: T_grok ~flat across allocation at fixed ||W|| "
                              "(scalar norm robust to E<->W2 reallocation)")

    if mode == "tempmed":
        base = [c for c in cells if c["arm"] == "clamp" and abs(c["rho"] - 1.00) < 1e-6 and abs(c["tau"] - 1.0) < 1e-6 and c["Tgrok_med"]]
        up = [c for c in cells if c["arm"] == "clamp" and abs(c["rho"] - 1.15) < 1e-6 and c["Tgrok_med"]]
        if base and up:
            T0 = base[0]["Tgrok_med"]; L0 = base[0]["logit_at_grok"]
            up_sorted = sorted(up, key=lambda c: c["tau"])
            T1 = [c for c in up_sorted if abs(c["tau"] - 1.0) < 1e-6]
            T1 = T1[0]["Tgrok_med"] if T1 else up_sorted[0]["Tgrok_med"]
            # compensated: tau whose logit_at_grok is closest to baseline L0
            comp = min(up_sorted, key=lambda c: abs(c["logit_at_grok"] - L0))
            T2 = comp["Tgrok_med"]
            res["tempmed"] = dict(T0=T0, T1=T1, T2=T2, L0=L0, tau_star=comp["tau"],
                                  logit_comp=comp["logit_at_grok"],
                                  up_curve=[dict(tau=c["tau"], T=c["Tgrok_med"], logit=c["logit_at_grok"]) for c in up_sorted])
            import math as _m
            res["FALSIFIER_B_triggered"] = bool(abs(_m.log(T2) - _m.log(T0)) < _m.log(1.3)
                                                 and abs(_m.log(T1) - _m.log(T0)) > _m.log(2.0))
            res["verdict"] = ("mediator: logit-scale compensation recovers baseline (norm effect "
                              "mediated by logit scale)") if res["FALSIFIER_B_triggered"] else \
                             ("H1 supported: matching logit scale does NOT recover the norm-up delay "
                              "(||W|| load-bearing beyond logit scale)")
    atomic_json(os.path.join(run_dir, "analysis.json"), res)
    return res

PRE_REGISTRATION = dict(
    H1=("at fixed total ||W||, T_grok invariant to gamma (Arm A); after matching logit scale, "
        "norm-up delay NOT recovered by tau-compensation (Arm B), T2 stays near T1"),
    falsifier_A="median T_grok changes > 2x across gamma at fixed ||W|| (=> mediator, not scalar norm)",
    falsifier_B="logit-matched compensated cell within 1.3x of baseline T0 while T1 > 2x T0 (=> mediated by logit scale)",
    note="both outcomes reportable; H1 holds => confound closed; H1 fails => identifies mediator and reframes")

# ============================ main ============================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["alloc", "tempmed"], required=True)
    ap.add_argument("--p", type=int, default=59); ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--loss", choices=["ce", "mse"], default="ce")
    ap.add_argument("--noise", choices=["full", "mini"], default="full")
    ap.add_argument("--alpha", type=float, default=0.40); ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--init_scale", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=512); ap.add_argument("--t_int", type=int, default=500)
    ap.add_argument("--gammas", type=float, nargs="+", default=[-0.6, -0.3, 0.0, 0.3, 0.6],
                    help="Arm A: E<->W2 mass shift at fixed total ||W|| (rho=1.0)")
    ap.add_argument("--taus", type=float, nargs="+", default=[1.0, 1.15, 1.3, 1.5, 1.7],
                    help="Arm B: logit temperatures at rho=1.15 (tau>1 lowers logit scale)")
    ap.add_argument("--budget_base", type=int, default=20000)
    ap.add_argument("--budget_growth", type=float, default=1.9)
    ap.add_argument("--max_steps_cap", type=int, default=120000)
    ap.add_argument("--log_every", type=int, default=200); ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=int, default=5000); ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out_root", default="./runs")
    ap.add_argument("--resume_dir", default=None)
    a = ap.parse_args()
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"

    if a.resume_dir: run_dir = a.resume_dir
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(a.out_root, f"paperC_confound_{a.mode}_p{a.p}_{ts}")
    paths = {"metrics": os.path.join(run_dir, "metrics"), "Temp": os.path.join(run_dir, "Temp")}
    for q in (run_dir, paths["metrics"], paths["Temp"]): os.makedirs(q, exist_ok=True)
    atomic_json(os.path.join(run_dir, "preregistration.json"), PRE_REGISTRATION)   # frozen BEFORE results

    base = dict(loss=a.loss, noise=a.noise, p=a.p, d_model=a.d_model, H=a.H, alpha=a.alpha,
                seeds=a.seeds, lr=a.lr, lam=a.lam, t_int=a.t_int, log_every=a.log_every,
                eval_every=a.eval_every, seed0=a.seed0, batch_size=a.batch_size, init_scale=a.init_scale,
                gamma=0.0, tau=1.0)
    cid = lambda tag: f"mlp_{a.loss}_{a.noise}_p{a.p}_{tag}"

    # control cell (arm none) calibrates wc AND base_frac (per-layer shares at grok)
    grid = [dict(base, arm="none", rho=1.0,
                 max_steps=min(a.max_steps_cap, a.budget_base * 3), config_id=cid("ctrl"))]
    if a.mode == "alloc":
        for gm in a.gammas:
            grid.append(dict(base, arm="alloc", rho=1.0, gamma=float(gm),
                             max_steps=min(a.max_steps_cap, int(a.budget_base * 2.5)),
                             config_id=cid(f"alloc_g{gm:+.2f}")))
    else:  # tempmed
        grid.append(dict(base, arm="clamp", rho=1.00, tau=1.0,
                         max_steps=min(a.max_steps_cap, int(a.budget_base * 2.0)),
                         config_id=cid("base_rho1.00_tau1.00")))
        for tau in a.taus:
            bud = int(min(a.max_steps_cap, a.budget_base * a.budget_growth ** ((1.15 - 1.0) / 0.05)))
            grid.append(dict(base, arm="clamp", rho=1.15, tau=float(tau), max_steps=bud,
                             config_id=cid(f"up_rho1.15_tau{tau:.2f}")))
    atomic_json(os.path.join(run_dir, "grid_spec.json"), grid)

    ckpt_path = os.path.join(run_dir, "checkpoint.json")
    ckpt = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {"done": [], "wc": None, "base_frac": None, "summary": []}
    wc = ckpt.get("wc"); base_frac = ckpt.get("base_frac")
    log(run_dir, f"start: mode={a.mode} p={a.p} {len(grid)} cfgs dev={device} "
                 f"resume={'yes' if ckpt['done'] else 'no'} done={len(ckpt['done'])} wc={wc} base_frac={base_frac}")
    for cfg in grid:
        if cfg["config_id"] in ckpt["done"]: continue
        if cfg["arm"] != "none" and (wc is None or base_frac is None):
            log(run_dir, "ERROR: control cell must grok first to set wc and base_frac."); sys.exit(1)
        cfg["base_frac"] = base_frac if base_frac else [0.0, 0.0, 0.0]
        ng, tg, out = run_config(cfg, wc if wc else 0.0, base_frac if base_frac else [1/3**0.5]*3,
                                 paths, device, a.ckpt_every)
        if cfg["arm"] == "none" and wc is None:
            wg = out["wn_at_grok"][out["T_grok_per_seed"] > 0]
            if len(wg):
                wc = float(np.median(wg)); ckpt["wc"] = wc
                # base_frac from per-layer norms at the control grok (last logged values)
                nE, nW1, nW2 = out["wn_E"][-1], out["wn_W1"][-1], out["wn_W2"][-1]
                tot = math.sqrt(nE ** 2 + nW1 ** 2 + nW2 ** 2)
                base_frac = [float(nE / tot), float(nW1 / tot), float(nW2 / tot)]
                ckpt["base_frac"] = base_frac
                atomic_json(os.path.join(run_dir, "calibration.json"),
                            {"wc": wc, "base_frac": base_frac, "n_grok": int(ng)})
                log(run_dir, f"   calibrated wc={wc:.2f} base_frac(E,W1,W2)={[round(x,3) for x in base_frac]}")
        ckpt["done"].append(cfg["config_id"])
        ckpt["summary"].append(dict(config_id=cfg["config_id"], arm=cfg["arm"], rho=cfg["rho"],
                                    gamma=cfg["gamma"], tau=cfg["tau"], grok=ng, Tgrok_med=tg,
                                    budget=cfg["max_steps"]))
        atomic_json(ckpt_path, ckpt); atomic_json(os.path.join(run_dir, "results_summary.json"), ckpt["summary"])
        log(run_dir, f"[{len(ckpt['done'])}/{len(grid)}] {cfg['config_id']} grok={ng}/{a.seeds} Tg={tg} "
                     f"bud={cfg['max_steps']} {out['wall_s']:.0f}s")
    res = analyze(paths, run_dir, a.mode)
    log(run_dir, f"VERDICT [{a.mode} p={a.p}]: {res.get('verdict', 'insufficient grokked cells')}")
    if a.mode == "alloc" and "alloc_T_ratio_maxmin" in res:
        log(run_dir, f"   Arm A: T_grok max/min across gamma = {res['alloc_T_ratio_maxmin']:.2f} "
                     f"(falsifier_A>{2.0} => mediator). gammas={res['alloc_gamma']} T={[round(x) for x in res['alloc_Tgrok']]}")
    if a.mode == "tempmed" and "tempmed" in res:
        tm = res["tempmed"]
        log(run_dir, f"   Arm B: T0={tm['T0']:.0f} T1={tm['T1']:.0f} T2(comp,tau*={tm['tau_star']:.2f})={tm['T2']:.0f} "
                     f"(falsifier_B => T2~T0 while T1>2T0)")
    log(run_dir, "DONE")

if __name__ == "__main__":
    main()
