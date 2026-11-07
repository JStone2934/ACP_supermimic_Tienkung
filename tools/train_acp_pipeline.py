"""Run the documented Stage 1 -> offline planner -> Stage 2 experiment.

Restart with the same --run-dir to reuse completion markers and resume last.pt.
The systemd service owns all child processes. Stop it before changing checkpoints.
"""

import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--prep-gpu", default="0")
    args = ap.parse_args()
    os.chdir(REPO)
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute_pipeline(args, root)


def execute_pipeline(args, root):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    refined = REPO / "data" / ("php_parkour_refined_" + root.name)
    base, ft, planner, stage2 = [root / n for n in ("stage1", "stage1_ft", "planner", "stage2")]
    for directory in (base, ft, planner, stage2):
        directory.mkdir(exist_ok=True)

    def status(stage, **extra):
        info = dict(stage=stage, updated_at=dt.datetime.now().astimezone().isoformat(), **extra)
        write_json(root / "status.json", info)
        print(json.dumps(info), flush=True)

    def run(name, module, options, *, resume_dir=None, resume=True, extra_env=None):
        marker = root / (name + ".done.json")
        if marker.exists():
            return
        command = [sys.executable, "-u", "-m", module, *map(str, options)]
        if resume and resume_dir and (resume_dir / "last.pt").exists():
            command += ["--resume", str(resume_dir / "last.pt")]
        write_json(root / (name + ".command.json"), command)
        status(name, command=command, state="running")
        with (root / (name + ".stdout.log")).open("a") as log:
            log.write("\nSTART " + dt.datetime.now().astimezone().isoformat() + "\n")
            log.flush()
            result = subprocess.run(command, env=env | (extra_env or {}), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status(name, state="failed", returncode=result.returncode)
            raise RuntimeError(f"{name} failed; inspect {name}.stdout.log before restarting")
        write_json(marker, dict(finished_at=dt.datetime.now().astimezone().isoformat(), command=command))

    # Prepare refined clips on the other GPU while the original-data tracker trains.
    prep = None
    report = refined / "refine_all.json"
    if report.exists() and len(json.loads(report.read_text())) == 296:
        write_json(root / "refine.done.json", {"output": str(refined)})
    if not (root / "refine.done.json").exists():
        command = [sys.executable, "-u", "-m", "supermimic.acp.refine_motion", "--out", str(refined)]
        write_json(root / "refine.command.json", command)
        with (root / "refine.stdout.log").open("a") as prep_log:
            prep = subprocess.Popen(command, env=env | {"CUDA_VISIBLE_DEVICES": args.prep_gpu},
                                    stdout=prep_log, stderr=subprocess.STDOUT)
        write_json(root / "refine.process.json", {"pid": prep.pid, "gpu": args.prep_gpu})

    shared = ["--num_envs", 4096, "--seed", 0, "--ckpt_every", 1000, "--log_every", 50,
              "--save_full_checkpoints"]
    run("stage1", "supermimic.acp.train_stage1",
        [*shared, "--iterations", 30000, "--log_dir", base], resume_dir=base)
    if prep:
        status("refine", state="waiting", pid=prep.pid)
        code = prep.wait()
        if code:
            status("refine", state="failed", returncode=code)
            raise RuntimeError("Motion refinement failed; see refine.stdout.log")
        write_json(root / "refine.done.json", {"output": str(refined)})
    run("refined_labels", "supermimic.acp.make_labels", ["--motion_dir", refined, "--workers", 16],
        extra_env={"OMP_NUM_THREADS": "1"})
    run("stage1_ft", "supermimic.acp.train_stage1",
        [*shared, "--iterations", 4000, "--init", base / "last.pt", "--motion_dir", refined,
         "--target_min_steps", 4, "--log_dir", ft], resume_dir=ft)

    selection = root / "selected_tracker.json"
    if not selection.exists():
        scores = []
        for ckpt in sorted(ft.glob("ckpt_*.pt")):
            name = "eval_" + ckpt.stem
            output = root / (name + ".json")
            # Match the README's Stage 1 protocol: every clip, no action noise.
            run(name, "supermimic.acp.evaluate", ["--ckpt", ckpt, "--out", output])
            result = json.loads(output.read_text())
            if not math.isfinite(result["norm_return"]):
                raise RuntimeError(f"Non-finite evaluation: {ckpt}")
            scores.append({"checkpoint": str(ckpt), "n_failed": result["n_failed"],
                           "norm_return": result["norm_return"], "evaluation": str(output)})
        if not scores:
            raise RuntimeError("No fine-tuned checkpoints available")
        chosen = min(scores, key=lambda s: (s["n_failed"], -s["norm_return"]))
        if chosen["n_failed"] == 296:
            raise RuntimeError("All evaluation clips failed; hold for six-hour review")
        write_json(selection, {"selected": chosen, "candidates": scores,
                               "criterion": "fewest failed clips, then highest normalized return"})
    tracker = Path(json.loads(selection.read_text())["selected"]["checkpoint"])
    run("planner", "supermimic.acp.train_planner_offline",
        ["--tracker", tracker, "--num_envs", 4096, "--steps", 20000, "--seed", 0, "--log_dir", planner])
    common2 = ["--init", tracker, "--planner_init", planner / "planner_020000.pt", "--num_envs", 4096,
               "--seed", 0, "--ckpt_every", 1000, "--log_dir", stage2]
    run("stage2_initial", "supermimic.acp.train_stage2", [*common2, "--iterations", 24000], resume_dir=stage2)
    run("stage2_window", "supermimic.acp.train_stage2",
        [*common2, "--iterations", 40000, "--term_window", 2, "--eta_rew_max", 0.58], resume_dir=stage2)
    run("stage2_evaluation", "supermimic.acp.evaluate",
        ["--ckpt", stage2 / "last.pt", "--planner", stage2 / "last.pt", "--train_noise", "action",
         "--out", root / "stage2_evaluation.json"])
    status("complete", state="complete")


if __name__ == "__main__":
    main()
