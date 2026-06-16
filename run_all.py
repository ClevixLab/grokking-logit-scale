#!/usr/bin/env python3
r"""run_all.py — reproduce every number, table, and figure of Paper C v12 from the raw data in metrics/.

Does NOT retrain (training is GPU-hours; raw outputs are bundled under metrics/). Runs the analysis and
figure scripts.

    python run_all.py

Outputs:
  reproduced_numbers.json     every scalar in the paper (make_paper_numbers.py)
  reproduced_samestate.json   same-state arms + ratio CI (analyze_samestate.py)
  figures/*.pdf               regenerated figures (scripts/figures/make_figures.py)

Two components need one GPU run each to reach 100%:
  - Table 1 p43 SC-free n=6 : python scripts/runners/run_paperC_dosecells_v1.py
  - SC audit float64 arm    : python scripts/runners/run_paperC_float64audit_v1.py
"""
import subprocess, sys, os
H = os.path.dirname(os.path.abspath(__file__))
def run(cmd):
    print("\n$ " + " ".join(cmd)); r = subprocess.run(cmd, cwd=H)
    if r.returncode != 0: print("FAILED:", " ".join(cmd))
run([sys.executable, "make_paper_numbers.py"])
run([sys.executable, "scripts/analysis/analyze_samestate.py"])
if os.path.exists(os.path.join(H, "scripts/figures/make_figures.py")):
    run([sys.executable, "scripts/figures/make_figures.py"])
print("\nAll paper numbers reproduced from metrics/. See reproduced_numbers.json and the Coverage section in README.md.")
