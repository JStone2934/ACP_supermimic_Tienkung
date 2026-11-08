"""Run Stage 1 -> offline planner -> Stage 2, optionally continuing to Stage 3.

Restart with the same --run-dir to reuse completion markers and resume last.pt.
The systemd service owns all child processes. Stop it before changing checkpoints.
"""

import argparse
import datetime as dt
import fcntl
import json
import hashlib
from supermimic.acp.robots import get_robot, sha256
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
    ap.add_argument("--robot", choices=["g1", "tienkung3_29dof"], default="g1")
    ap.add_argument("--motion-dir")
    ap.add_argument("--contacts-dir")
    devices = ap.add_mutually_exclusive_group()
    devices.add_argument("--gpu", default="1")
    devices.add_argument("--gpus", help="comma-separated devices for one synchronous training job, e.g. 0,1")
    ap.add_argument("--prep-gpu", default="0")
    ap.add_argument("--master-port", type=int, default=29547)
    ap.add_argument("--through-stage", type=int, choices=(2, 3), default=2)
    ap.add_argument("--num-envs", type=int, default=4096, help="global simulated environments")
    ap.add_argument("--wm-batch", type=int, default=2048)
    ap.add_argument("--pi-batch", type=int, default=4096)
    ap.add_argument("--planner-batch", type=int, default=4096)
    ap.add_argument("--planner-sub-batches", type=int, default=None)
    ap.add_argument("--planner-min-entries", type=int, default=None)
    ap.add_argument("--planner-capacity", type=int, default=None)
    ap.add_argument("--buffer-steps", type=int, default=None)
    ap.add_argument("--reward-chunk-size", type=int, default=0, help="checkpoint analytic reward rows to reduce activation memory")
    args = ap.parse_args()
    args.train_gpus = [g.strip() for g in (args.gpus if args.gpus is not None else args.gpu).split(',')]
    if any(not g.strip() for g in args.train_gpus) or len(set(args.train_gpus)) != len(args.train_gpus):
        ap.error('GPU identifiers must be nonempty and distinct')
    if not 1 <= args.master_port <= 65535:
        ap.error('--master-port must be in 1..65535')
    workers = len(args.train_gpus)
    for name in ('num_envs', 'wm_batch', 'pi_batch'):
        value = getattr(args, name)
        if value < workers or value % workers:
            ap.error(f'global {name} must be positive and divisible by the GPU count')
    sub_batches = 8 if args.planner_sub_batches is None else args.planner_sub_batches
    if sub_batches < 1 or args.planner_batch < 1 or args.planner_batch % (workers * sub_batches):
        ap.error('global planner batch must be divisible by planner sub-batches × GPU count')
    if args.through_stage == 3:
        for value in (args.num_envs, args.pi_batch):
            goal = round(value * .5)
            if min(goal, value-goal) < workers or goal % workers or (value-goal) % workers:
                ap.error('Stage 3 reference and goal groups must each be divisible by the GPU count')
    os.chdir(REPO)
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children = []
        try:
            execute_pipeline(args, root, children)
        finally:
            for process in children:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def execute_pipeline(args, root, children=None):
    robot = get_robot(args.robot)
    source = Path(args.motion_dir or robot.motion_dir).resolve()
    contacts = Path(args.contacts_dir or (str(source) + "_contacts"))
    inputs = sorted(source.glob("*/*_motion.npz"))
    if not inputs: raise ValueError(f"no input motions in {source}")
    input_files = inputs + [Path(str(p).replace("_motion.npz", "_terrain.npy")) for p in inputs]
    input_files += sorted(contacts.rglob("*.npz"))
    fingerprint = hashlib.sha256(json.dumps(dict(robot=robot.metadata(), files=[(str(p), sha256(p)) for p in input_files]), sort_keys=True).encode()).hexdigest()
    binding = root / "inputs.json"
    if binding.exists() and json.loads(binding.read_text())["fingerprint"] != fingerprint:
        raise ValueError("run inputs changed; use a new --run-dir (old markers/checkpoints are not reusable)")
    if not binding.exists() and list(root.glob("*.done.json")):
        raise ValueError("unbound legacy completion markers; use a new --run-dir")
    write_json(binding, dict(fingerprint=fingerprint, clips=len(inputs), robot_metadata=robot.metadata()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.train_gpus[0], OMP_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    refined = REPO / "data" / (robot.robot_id + "_refined_" + root.name)
    base, ft, planner, stage2 = [root / n for n in ("stage1", "stage1_ft", "planner", "stage2")]
    for directory in (base, ft, planner, stage2):
        directory.mkdir(exist_ok=True)

    def status(stage, **extra):
        info = dict(stage=stage, updated_at=dt.datetime.now().astimezone().isoformat(), **extra)
        write_json(root / "status.json", info)
        print(json.dumps(info), flush=True)

    def run(name, module, options, *, resume_dir=None, resume=True, extra_env=None):
        marker = root / (name + ".done.json")
        command = [sys.executable, "-u", "-m", module, "--robot", args.robot, *map(str, options)]
        training = module.rsplit('.', 1)[-1] in ('train_stage1', 'train_stage2', 'train_stage3', 'train_planner_offline')
        if training and len(args.train_gpus) > 1:
            command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--nnodes=1',
                       f'--nproc-per-node={len(args.train_gpus)}', '--master-addr=127.0.0.1',
                       f'--master-port={args.master_port}', '--max-restarts=0', '--module', module, *command[4:]]
        dependencies = {}
        for i, token in enumerate(command[:-1]):
            if token in ("--ckpt", "--tracker", "--init", "--planner_init"):
                path = Path(command[i + 1])
                if path.is_file(): dependencies[str(path)] = sha256(path)
            elif token in ("--motion_dir", "--contacts_dir"):
                directory = Path(command[i + 1])
                dependencies[str(directory)] = [(str(p.relative_to(directory)), sha256(p)) for p in sorted(directory.rglob("*.np*"))]
        command_hash = hashlib.sha256(json.dumps(dict(command=command, dependencies=dependencies), sort_keys=True).encode()).hexdigest()
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved.get("fingerprint") != fingerprint or saved.get("command_hash") != command_hash:
                raise ValueError(f"stale completion marker: {name}")
            return
        if resume and resume_dir and (resume_dir / "last.pt").exists():
            command += ["--resume", str(resume_dir / "last.pt")]
        write_json(root / (name + ".command.json"), command)
        status(name, command=command, state="running")
        with (root / (name + ".stdout.log")).open("a") as log:
            log.write("\nSTART " + dt.datetime.now().astimezone().isoformat() + "\n")
            log.flush()
            run_env = env | ({'CUDA_VISIBLE_DEVICES': ','.join(args.train_gpus)} if training else {}) | (extra_env or {})
            result = subprocess.run(command, env=run_env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status(name, state="failed", returncode=result.returncode)
            raise RuntimeError(f"{name} failed; inspect {name}.stdout.log before restarting")
        write_json(marker, dict(finished_at=dt.datetime.now().astimezone().isoformat(), command=command, fingerprint=fingerprint, command_hash=command_hash))

    # Prepare refined clips on the other GPU while the original-data tracker trains.
    prep = None
    report = refined / "refine_all.json"
    # File count alone cannot establish that refinement used these inputs.
    refine_marker = root / "refine.done.json"
    if refine_marker.exists() and json.loads(refine_marker.read_text()).get("fingerprint") != fingerprint:
        raise ValueError("stale refinement marker")
    if not (root / "refine.done.json").exists():
        command = [sys.executable, "-u", "-m", "supermimic.acp.refine_motion", "--robot", args.robot, "--motion_dir", str(source), "--contacts_dir", str(contacts), "--out", str(refined)]
        write_json(root / "refine.command.json", command)
        with (root / "refine.stdout.log").open("a") as prep_log:
            prep = subprocess.Popen(command, env=env | {"CUDA_VISIBLE_DEVICES": args.prep_gpu},
                                    stdout=prep_log, stderr=subprocess.STDOUT)
        if children is not None:
            children.append(prep)
        write_json(root / "refine.process.json", {"pid": prep.pid, "gpu": args.prep_gpu})

    def finish_prep():
        nonlocal prep
        if prep is None:
            return
        status("refine", state="waiting", pid=prep.pid)
        code = prep.wait()
        if code:
            status("refine", state="failed", returncode=code)
            raise RuntimeError("Motion refinement failed; see refine.stdout.log")
        write_json(root / "refine.done.json", {"output": str(refined), "fingerprint": fingerprint})
        prep = None

    if args.prep_gpu in args.train_gpus:
        finish_prep()
    sizes = ['--num_envs', args.num_envs, '--wm_batch', args.wm_batch, '--pi_batch', args.pi_batch]
    if args.reward_chunk_size:
        sizes += ['--reward_chunk_size', args.reward_chunk_size]
    buffer_options = ['--buffer_steps', args.buffer_steps] if args.buffer_steps is not None else []
    planner_options = ['--planner_batch', args.planner_batch]
    if args.planner_sub_batches is not None:
        planner_options += ['--planner_sub_batches', args.planner_sub_batches]
    if args.planner_min_entries is not None:
        planner_options += ['--planner_min_entries', args.planner_min_entries]
    shared = [*sizes, *buffer_options, "--seed", 0, "--ckpt_every", 1000, "--log_every", 50, "--save_full_checkpoints"]
    run("stage1", "supermimic.acp.train_stage1",
        [*shared, "--motion_dir", source, "--contacts_dir", contacts, "--iterations", 30000, "--log_dir", base], resume_dir=base)
    finish_prep()
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
        if chosen["n_failed"] == len(inputs):
            raise RuntimeError("All evaluation clips failed; hold for six-hour review")
        write_json(selection, {"selected": chosen, "candidates": scores,
                               "criterion": "fewest failed clips, then highest normalized return"})
    tracker = Path(json.loads(selection.read_text())["selected"]["checkpoint"])
    run("planner", "supermimic.acp.train_planner_offline",
        ["--tracker", tracker, "--num_envs", args.num_envs, *buffer_options, *planner_options,
         *(['--capacity', args.planner_capacity] if args.planner_capacity is not None else []),
         "--steps", 20000, "--seed", 0, "--log_dir", planner], resume_dir=planner)
    co_options = [*sizes, *buffer_options, *planner_options, *(['--planner_capacity', args.planner_capacity] if args.planner_capacity is not None else [])]
    common2 = ["--init", tracker, "--planner_init", planner / "planner_020000.pt", *co_options,
               "--seed", 0, "--ckpt_every", 1000, "--log_dir", stage2]
    run("stage2_initial", "supermimic.acp.train_stage2", [*common2, "--iterations", 24000], resume_dir=stage2)
    run("stage2_window", "supermimic.acp.train_stage2",
        [*common2, "--iterations", 40000, "--term_window", 2, "--eta_rew_max", 0.58], resume_dir=stage2)
    run("stage2_evaluation", "supermimic.acp.evaluate",
        ["--ckpt", stage2 / "last.pt", "--planner", stage2 / "last.pt", "--train_noise", "action",
         "--out", root / "stage2_evaluation.json"])
    if args.through_stage == 3:
        stage3 = root / 'stage3'
        stage3.mkdir(exist_ok=True)
        run('stage3', 'supermimic.acp.train_stage3',
            ['--init', stage2 / 'last.pt', *co_options, '--iterations', 30000, '--seed', 0,
             '--ckpt_every', 1000, '--log_dir', stage3], resume_dir=stage3)
        run('stage3_evaluation', 'supermimic.acp.evaluate_goal',
            ['--ckpt', stage3 / 'last.pt', '--out', root / 'stage3_evaluation.json'])
    status("complete", state="complete")


if __name__ == "__main__":
    main()
