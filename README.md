# What Does the Weight Norm Control in Grokking? reproducibility repository

Code, raw data, and analysis for *What Does the Weight Norm Control in Grokking? Logit-Scale Mediation
under Cross-Entropy*.

Claim: under cross-entropy the weight-norm dependence of the grokking delay runs through the effective
logit scale (and the softmax saturation it causes), not the scalar norm; under mean-squared error that
channel is absent. This repo regenerates every number, table, and figure from the raw training outputs.

## Quick start

    git clone https://github.com/ClevixLab/grokking-logit-scale
    cd grokking-logit-scale

    pip install -r requirements.txt
    python run_all.py

`run_all.py` does not retrain (raw outputs are bundled). It writes `reproduced_numbers.json` (every scalar
in the paper), `reproduced_samestate.json`, and figures.

## What reproduces what (no orphan numbers)

| paper float | produced by | from |
|---|---|---|
| Table 1 (exp-law alpha + AIC) | make_paper_numbers.py | metrics/dose/ |
| Table 2 (mediation, recovery fraction + CI) | make_paper_numbers.py | metrics/mediation/ |
| Tables 4/5 (per-cell CE / MSE) | make_paper_numbers.py | metrics/mediation/ |
| Table 3 (collapse regression) | make_paper_numbers.py | metrics/collapse/ |
| Tables 6/9 (layer allocation) | make_paper_numbers.py | metrics/alloc/ |
| transformer table + 4.7x | make_paper_numbers.py | metrics/transformer/ |
| Section 4.6 same-state (3.3x + CI) | make_paper_numbers.py / analyze_samestate.py | metrics/samestate/ |

`make_paper_numbers.py` is the single source of truth; it prints a coverage report.

## Coverage (honest)

Reproduces bit-for-bit from bundled data: Table 1 at p in {59,67,97,113} (alpha 0.140/0.126/0.094/0.087),
recovery fractions 0.83 [0.82,0.85] and 0.89 [0.88,0.91], collapse R2(logit)=0.968/0.973, the
layer-allocation swings, transformer 4.7x, and same-state ratios 3.36 [3.27,3.43] / 3.27 [3.21,3.38].

Two components need one GPU run each to reach 100%:
- Table 1 at p=43: SC-free range is n=4 here; published n=6 adds three low-rho cells (the exponent already
  matches: 0.174 vs 0.173). Run scripts/runners/run_paperC_dosecells_v1.py, drop output into metrics/dose/.
- float64 softmax-collapse audit (Section 5): float32 sc rate is in the dose cells; the float32-vs-float64
  comparison needs scripts/runners/run_paperC_float64audit_v1.py.

## Notes
- Precision float32, no mixed precision (the logit-scale and SC-audit measurements depend on it).
- Bootstrap CIs use a fixed RNG seed (deterministic).
- See env.txt for exact hardware/software.

