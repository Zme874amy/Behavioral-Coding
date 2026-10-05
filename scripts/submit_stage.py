"""Expand a stage config into re-run cells and submit them to MLeRP.

    python scripts/submit_stage.py conf/stages/stage1.yaml            # dry run
    python scripts/submit_stage.py conf/stages/stage1.yaml --submit   # sbatch

Each cell is one scripts/mlerp_run.slurm job (train + predict one fold). At
most LANES jobs run at once: job i gets the name rerun_lane<i % LANES> and
--dependency=singleton, and SLURM runs one job per name at a time, which keeps
us inside lion's 4-job cap without a polling loop. Cells whose prediction file
is already complete (.meta.json "complete": true) are skipped, so re-running
the submitter after a failure only resubmits what is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from baseline import folds, rerun  # noqa: E402

LANES = 4


def expand(cfg: dict):
    for cell in cfg["cells"]:
        for pname in cell["protocols"]:
            proto = cfg["protocols"][pname]
            for sname in cell["students"]:
                st = cfg["students"][sname]
                if not st.get("enabled", True):
                    continue
                for seed in cell.get("seeds", cfg["seeds"]):
                    for fold in range(folds.n_folds(proto["manifest"], proto["design"])):
                        yield {"MANIFEST": proto["manifest"], "DESIGN": proto["design"], "FOLD": fold,
                               "SEED": seed, "ARM": cell["arm"], "INF": cell["inf"],
                               # staged copy on MLeRP if given (compute nodes are offline)
                               "STUDENT": st.get("path", st["model"]),
                               "CTX": cfg["ctx"], "PROMPT": cell.get("prompt", cfg["prompt"]), "SOURCE": cell.get("source", ""),
                               "EXTRA": st.get("extra", "")}


def done(c: dict) -> bool:
    src = {"misc.hlqc.gold": c["SOURCE"]} if c["SOURCE"] else None
    p = rerun.result_path(c["MANIFEST"], c["DESIGN"], rerun.student_slug(c["STUDENT"]), c["ARM"], c["INF"],
                          c["CTX"], c["PROMPT"], c["SEED"], c["FOLD"], src).with_suffix(".meta.json")
    return p.exists() and json.loads(p.read_text()).get("complete", False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    cells = list(expand(cfg))
    todo = [c for c in cells if not done(c)]
    print(f"{len(cells)} cells, {len(cells) - len(todo)} complete, {len(todo)} to run")
    for i, c in enumerate(todo):
        env = {k: str(v) for k, v in c.items()}
        cmd = ["sbatch", f"--job-name=rerun_lane{i % LANES}", "--dependency=singleton",
               "--export=ALL," + ",".join(f"{k}={v}" for k, v in env.items() if v and k != "EXTRA"),
               "scripts/mlerp_run.slurm"]
        label = f"{c['ARM']}{'+' + c['SOURCE'] if c['SOURCE'] else ''} {c['MANIFEST']}/{c['DESIGN']} " \
                f"f{c['FOLD']} s{c['SEED']} {c['STUDENT'].split('/')[-1]} {c['PROMPT']}"
        if not a.submit:
            print("DRY", label)
            continue
        # EXTRA holds spaces, so it travels through the environment, not --export's comma list.
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                             env={**os.environ, "EXTRA": c["EXTRA"]})
        print(label, "->", (out.stdout or out.stderr).strip())


if __name__ == "__main__":
    main()
