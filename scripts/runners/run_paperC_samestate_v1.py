r"""run_paperC_samestate_v1.py — the same-state free-vs-clamp test that closes the clamp-rescaling confound.

The confound: the clamp holds ||W|| fixed by rescaling the weights, and rescaling also moves the logits.
So a skeptic can ask whether the norm-induced delay reflects the held *value* of the norm or an artifact of
the rescaling *operation* applied every step. This experiment separates the two by forking arms from one
identical state.

Design (per modulus, 12 seeds):
  Phase 1 (shared): train free from init to a fork step t_fork chosen AFTER memorization and BEFORE
                    grokking. Snapshot the full optimizer state and record each seed's norm N0 there.
  Phase 2 (arms, each continues from the identical snapshot):
    free          : no clamp (norm evolves on its own)
    clamp_same    : clamp each seed to its own N0          (clamp operation ON, no rescaling jump)
    clamp_raised  : clamp each seed to rho * N0            (operation ON, value raised)
    clamp_lowered : clamp each seed to N0 / rho            (operation ON, value lowered)

Decisive contrasts (printed as the verdict):
  clamp_same  vs  clamp_raised : both arms apply the identical clamp operation from the identical state and
                                 differ only in the held value. A longer delay for clamp_raised means the
                                 delay tracks the norm VALUE, not the clamp operation -> confound closed.
  free        vs  clamp_same   : whether the clamp operation at unchanged value introduces any spurious
                                 delay by itself.

Engineering: timestamped run dir under --out_root; resumable (phase-1 checkpoint + per-arm checkpoint +
within-arm state); the fork snapshot and per-arm final weights are kept in Temp/ for later analysis.

Run (single CUDA GPU):
    python run_paperC_samestate_v1.py --p 59 --device cuda
    python run_paperC_samestate_v1.py --p 97 --device cuda
"""
import argparse, json, os, sys, math, time, datetime, glob
import numpy as np
import torch
import torch.nn.functional as F

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

# ----------------------------- data / model (MLP, matches the main runner) ----
def make_pairs(p, device):
    a = torch.arange(p, device=device).repeat_interleave(p)
    b = torch.arange(p, device=device).repeat(p)
    return a, b, (a + b) % p

def per_seed_train_mask(S, N, alpha, seed0, device):
    ntr = int(alpha * N); tm = torch.zeros(S, N, dtype=torch.bool, device=device)
    for s in range(S):
        g = torch.Generator().manual_seed(1000 + seed0 + s)
        tm[s, torch.randperm(N, generator=g)[:ntr].to(device)] = True
    return tm, ntr

def init_mlp(S, p, d, H, seed, device):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *sh, fan: (torch.randn(*sh, generator=g) / math.sqrt(fan)).to(device)
    P = dict(E=rn(S, p, d, fan=d), W1=rn(S, 2 * d, H, fan=2 * d), b1=torch.zeros(S, H, device=device),
             W2=rn(S, H, p, fan=H), b2=torch.zeros(S, p, device=device))
    for k in P: P[k].requires_grad_(True)
    return P

WMATS = ["E", "W1", "W2"]
def total_wnorm(P): return torch.sqrt(sum((P[k] ** 2).flatten(1).sum(1) for k in WMATS))  # (S,)
def egather(E, idx):
    S, M = idx.shape; return torch.gather(E, 1, idx.unsqueeze(-1).expand(S, M, E.shape[-1]))
def fwd(P, Ai, Bi):
    x = torch.cat([egather(P["E"], Ai), egather(P["E"], Bi)], -1)
    h = F.gelu(torch.einsum("smk,skh->smh", x, P["W1"]) + P["b1"][:, None])
    return torch.einsum("smh,shp->smp", h, P["W2"]) + P["b2"][:, None]

@torch.no_grad()
def clamp_perseed(P, target):                       # target: (S,) per-seed norm target
    sc = target / (total_wnorm(P) + 1e-12)
    for k in WMATS: P[k].mul_(sc.view(-1, *([1] * (P[k].dim() - 1))))

# ----------------------------- one training segment ---------------------------
def adam_step(P, m, v, t, lr, lam, b1=0.9, b2=0.999, eps=1e-8):
    with torch.no_grad():
        for k in P:
            g = P[k].grad
            m[k].mul_(b1).add_(g, alpha=1 - b1); v[k].mul_(b2).addcmul_(g, g, value=1 - b2)
            mh = m[k] / (1 - b1 ** t); vh = v[k] / (1 - b2 ** t)
            P[k].addcdiv_(mh, vh.sqrt().add_(eps), value=-lr)
            if k in WMATS: P[k].add_(P[k], alpha=-lr * lam)

def evaluate(P, A, B, Y, train_m, test_m, ntr):
    with torch.no_grad():
        pred = fwd(P, A[None].expand(train_m.shape[0], -1), B[None].expand(train_m.shape[0], -1)).argmax(-1)
        tr = ((pred == Y[None]) & train_m).sum(1).float() / ntr
        te = ((pred == Y[None]) & test_m).sum(1).float() / test_m.sum(1).float()
    return tr, te

# ----------------------------- main -------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=59); ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--H", type=int, default=256); ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.40); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--rho", type=float, default=1.15)
    ap.add_argument("--t_fork", type=int, default=0, help="0 = auto (just after all seeds memorize)")
    ap.add_argument("--t_fork_min", type=int, default=800)
    ap.add_argument("--max_steps", type=int, default=60000)
    ap.add_argument("--log_every", type=int, default=200); ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=int, default=5000); ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out_root", default="./runs")
    ap.add_argument("--resume_dir", default=None)
    a = ap.parse_args()
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    S, p = a.seeds, a.p

    run_dir = a.resume_dir or os.path.join(
        a.out_root, f"paperC_samestate_p{p}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    paths = {"metrics": os.path.join(run_dir, "metrics"), "Temp": os.path.join(run_dir, "Temp")}
    for q in (run_dir, *paths.values()): os.makedirs(q, exist_ok=True)
    atomic_json(os.path.join(run_dir, "preregistration.json"), dict(
        H1="from one identical post-memorization state, grok delay is a monotone function of the held norm "
           "VALUE; clamp_same vs clamp_raised differ only in value (operation identical) -> the difference "
           "isolates the value and closes the clamp-rescaling confound",
        arms=["free", "clamp_same", "clamp_raised", "clamp_lowered"], rho=a.rho))

    A, B, Y = make_pairs(p, device); N = A.shape[0]
    train_m, ntr = per_seed_train_mask(S, N, a.alpha, a.seed0, device)
    test_m = ~train_m
    fork_path = os.path.join(paths["Temp"], "fork_state.pt")

    # ---------------- Phase 1: free training to the fork point ----------------
    if not os.path.exists(fork_path):
        P = init_mlp(S, p, a.d_model, a.H, a.seed0, device)
        m = {k: torch.zeros_like(P[k]) for k in P}; v = {k: torch.zeros_like(P[k]) for k in P}
        T_mem = np.full(S, -1, np.int64); t_fork = None
        log(run_dir, f"phase 1: free training to fork (p={p}, auto t_fork)")
        for t in range(1, a.max_steps + 1):
            logits = fwd(P, A[None].expand(S, -1), B[None].expand(S, -1))
            le = -F.log_softmax(logits, -1).gather(-1, Y[None].expand(S, -1).unsqueeze(-1)).squeeze(-1)
            L = ((le * train_m).sum(1) / ntr).sum()
            for k in P: P[k].grad = None
            L.backward(); adam_step(P, m, v, t, a.lr, a.lam)
            if t % a.eval_every == 0 or t == 1:
                tr, te = evaluate(P, A, B, Y, train_m, test_m, ntr)
                T_mem[(T_mem < 0) & (tr >= 0.99).cpu().numpy()] = t
                memorized = (T_mem > 0).all()
                grokked_any = (te >= 0.90).any().item()
                want = a.t_fork if a.t_fork > 0 else max(a.t_fork_min, t)
                if (a.t_fork > 0 and t >= a.t_fork) or (a.t_fork == 0 and memorized and t >= a.t_fork_min):
                    if grokked_any:
                        log(run_dir, f"   WARNING: a seed grokked by t={t} before fork; "
                                     f"consider lowering --t_fork_min."); 
                    t_fork = t; break
        if t_fork is None: log(run_dir, "ERROR: never reached fork condition."); sys.exit(1)
        N0 = total_wnorm(P).detach().cpu().numpy()
        torch.save(dict(P={k: P[k].detach().cpu() for k in P}, m={k: m[k].cpu() for k in m},
                        v={k: v[k].cpu() for k in v}, t_fork=t_fork, N0=N0, T_mem=T_mem), fork_path)
        log(run_dir, f"   forked at t={t_fork}; N0 median={np.median(N0):.2f} "
                     f"range=[{N0.min():.1f},{N0.max():.1f}]; T_mem median={np.median(T_mem[T_mem>0]):.0f}")
    snap = torch.load(fork_path, map_location=device, weights_only=False)
    t_fork = int(snap["t_fork"]); N0 = torch.tensor(snap["N0"], device=device, dtype=torch.float32)

    # ---------------- Phase 2: arms from the identical snapshot ----------------
    arms = {"free": None, "clamp_same": N0.clone(), "clamp_raised": a.rho * N0, "clamp_lowered": N0 / a.rho}
    ckpt_path = os.path.join(run_dir, "checkpoint.json")
    ckpt = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {"done": [], "summary": []}

    for arm, target in arms.items():
        if arm in ckpt["done"]: continue
        state = os.path.join(paths["Temp"], f"{arm}_state.pt")
        P = {k: snap["P"][k].clone().to(device).requires_grad_(True) for k in snap["P"]}
        m = {k: snap["m"][k].clone().to(device) for k in snap["m"]}
        v = {k: snap["v"][k].clone().to(device) for k in snap["v"]}
        t0 = t_fork; T_grok = np.full(S, -1, np.int64)
        BUF = {k: [] for k in ["steps", "train_acc", "test_acc", "weight_norm", "logit_scale"]}
        if os.path.exists(state):
            st = torch.load(state, map_location=device, weights_only=False)
            for k in P: P[k].data = st["P"][k].to(device); m[k] = st["m"][k].to(device); v[k] = st["v"][k].to(device)
            t0, T_grok, BUF = st["t"], st["T_grok"], st["buf"]
        if target is not None and t0 == t_fork: clamp_perseed(P, target)   # engage clamp at fork (no jump for 'same')
        tw = time.time()
        for t in range(t0 + 1, a.max_steps + 1):
            logits = fwd(P, A[None].expand(S, -1), B[None].expand(S, -1))
            le = -F.log_softmax(logits, -1).gather(-1, Y[None].expand(S, -1).unsqueeze(-1)).squeeze(-1)
            L = ((le * train_m).sum(1) / ntr).sum()
            for k in P: P[k].grad = None
            L.backward(); adam_step(P, m, v, t, a.lr, a.lam)
            if target is not None: clamp_perseed(P, target)
            if t % a.eval_every == 0:
                tr, te = evaluate(P, A, B, Y, train_m, test_m, ntr)
                T_grok[(T_grok < 0) & (te >= 0.90).cpu().numpy()] = t
            if t % a.log_every == 0 or t == t0 + 1:
                tr, te = evaluate(P, A, B, Y, train_m, test_m, ntr)
                with torch.no_grad():
                    lg = fwd(P, A[None].expand(S, -1), B[None].expand(S, -1))
                BUF["steps"].append(t); BUF["train_acc"].append(tr.mean().item())
                BUF["test_acc"].append(te.mean().item()); BUF["weight_norm"].append(total_wnorm(P).mean().item())
                BUF["logit_scale"].append(lg.norm(dim=-1).mean().item())
            if t % a.ckpt_every == 0:
                torch.save(dict(P={k: P[k].detach().cpu() for k in P}, m={k: m[k].cpu() for k in m},
                                v={k: v[k].cpu() for k in v}, t=t, T_grok=T_grok, buf=BUF),
                           state + ".tmp"); os.replace(state + ".tmp", state)
            if (T_grok > 0).all(): break
        out = dict(arm=arm, p=p, rho=a.rho, t_fork=t_fork, n_seed=S, alpha=a.alpha, lr=a.lr, lam=a.lam,
                   T_grok_per_seed=T_grok, N0=snap["N0"], T_mem_per_seed=snap["T_mem"],
                   target=(target.detach().cpu().numpy() if target is not None else np.zeros(S)),
                   steps_run=t, wall_s=time.time() - tw)
        for k, val in BUF.items(): out[k] = np.array(val, dtype=np.float32)
        atomic_bytes(os.path.join(paths["metrics"], f"{arm}.npz"), lambda f: np.savez_compressed(f, **out))
        # keep final weights for later analysis
        atomic_bytes(os.path.join(paths["Temp"], f"{arm}_weights.npz"),
                     lambda f: np.savez_compressed(f, **{k: P[k].detach().cpu().numpy() for k in WMATS},
                                                   T_grok_per_seed=T_grok))
        if os.path.exists(state): os.remove(state)
        g = T_grok[T_grok > 0]; tgm = float(np.median(g)) if len(g) else float("nan")
        delay = (tgm - t_fork) if not math.isnan(tgm) else float("nan")
        ckpt["done"].append(arm)
        ckpt["summary"].append(dict(arm=arm, grok=int((T_grok > 0).sum()), Tgrok_med=tgm, delay_from_fork=delay))
        atomic_json(ckpt_path, ckpt)
        log(run_dir, f"   arm {arm:13s} grok={int((T_grok>0).sum())}/{S} T_grok={tgm:.0f} "
                     f"delay_from_fork={delay:.0f} {out['wall_s']:.0f}s")

    # ---------------- analysis: the decisive contrasts ----------------
    summ = {r["arm"]: r for r in ckpt["summary"]}
    def dl(arm): return summ[arm]["delay_from_fork"] if arm in summ else None
    res = dict(t_fork=t_fork, arms=ckpt["summary"])
    if all(x in summ for x in ["clamp_same", "clamp_raised"]):
        res["value_vs_operation"] = dict(
            clamp_same_delay=dl("clamp_same"), clamp_raised_delay=dl("clamp_raised"),
            raised_over_same=(dl("clamp_raised") / dl("clamp_same")) if dl("clamp_same") else None,
            reading="both arms apply the identical clamp operation from the identical state; a ratio>1 means "
                    "the extra delay is due to the held norm VALUE, not the clamp operation (confound closed)")
    if all(x in summ for x in ["free", "clamp_same"]):
        res["operation_check"] = dict(free_delay=dl("free"), clamp_same_delay=dl("clamp_same"),
            reading="clamp at unchanged value vs free; isolates the clamp operation itself")
    atomic_json(os.path.join(run_dir, "analysis.json"), res)
    if "value_vs_operation" in res:
        v = res["value_vs_operation"]
        log(run_dir, f"VERDICT p={p}: clamp_same delay={v['clamp_same_delay']:.0f}, "
                     f"clamp_raised delay={v['clamp_raised_delay']:.0f}, ratio={v['raised_over_same']:.2f} "
                     f"(>1 => delay tracks the norm VALUE, operation identical => confound closed)")
    log(run_dir, "DONE")

if __name__ == "__main__":
    main()
