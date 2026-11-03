# SuperMimic reproduction on Isaac Lab 3 (kit-less, MuJoCo-Warp)

**Unofficial** reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
Model* (arXiv 2610.28480) — no official code is released, so everything here is implemented from
the paper. Target experiment: the controlled LAFAN1 comparison (Fig. 4a): 40 G1-retargeted LAFAN1
clips (2.45 h), one GPU; the paper reaches evaluation normalized return 0.82 after 1.2 h on an H100.

Author: **Chong Zhang** (ETH Zurich). License: [MIT](LICENSE).

> **Disclaimer:** this repository was speed-vibecoded in 1 hour with an AI coding
> assistant. It
> deliberately does not live under any lab's organization, because the code is too much "slop"
> for that: fast and unreviewed. Use it at
> your own risk.

> **One G1 motion tracker for all 40 LAFAN1 clips, trained in only ~1.5 GPU-hours**
> (1.47 h on a single RTX PRO 6000): evaluation return **0.829** with **0/40 failed clips**.
> For scale, the paper's PPO-in-simulator baseline does not reach return 0.82 within 18 h on up to
> 8 GPUs.

![SuperMimic G1 tracking LAFAN1 motions](results/supermimic_lafan1_highlight.gif)

*Solid robot: the trained policy simulated in Isaac Lab with MuJoCo-Warp. Blue ghost: the reference
motion. Clips: dance, jumps, fall-and-get-up, sprint.*

Progress and timestamps: [MILESTONES.md](MILESTONES.md).

## Layout

| Path | What |
|---|---|
| `IsaacLab/` | Isaac Lab `main` @ 46d6cc3 (VERSION 3.0.0), unmodified — cloned by `tools/setup.sh`, not tracked |
| `docker/` | `Dockerfile`: `supermimic:latest` = `isaac-lab-kitless:main-46d6cc3` (built from `IsaacLab/docker/Dockerfile.kitless`) + tmux, ffmpeg, gcc (for `torch.compile`); `run_container.sh` |
| `data/` | not tracked; `tools/download_data.sh` fetches LAFAN1 G1 retargets (HF `lvhaidong/LAFAN1_Retargeting_Dataset`) and the G1 URDF/meshes, then builds `g1_29dof.xml` (MJCF) and the USD |
| `supermimic/` | implementation (see below) |
| `supermimic/acp/` | attention-based contact planner + contact tracker for the G1 on box terrain, on mjlab ([below](#contact-planner-on-box-terrain-g1-mjlab)) |
| `tools/` | setup, asset conversion, tests, benchmark, plotting, rendering, failure analysis |
| `logs/<run>/` | not tracked: `train_log.csv`, `eval.csv`, checkpoints, stdout |
| `results/` | success curve and highlight GIF (full videos are rendered locally with `tools/render.py`) |

## Setup

Neither Isaac Lab nor any data is included in this repository; the steps below fetch them.

**Requirements:** Linux, an NVIDIA GPU (tested: RTX PRO 6000 Blackwell, driver 580 / CUDA 13),
Docker with the NVIDIA Container Toolkit (`docker run --gpus` must work), ~20 GB of disk for the
images and ~0.2 GB for the data.

Physics: Newton with the MuJoCo-Warp solver (`NewtonCfg(MJWarpSolverCfg(...))`), no Kit, no PhysX;
50 Hz control, 4 substeps of 5 ms, implicitfast integrator, MuJoCo collision pipeline.

### 1. Isaac Lab (kit-less Docker image)

The code runs against Isaac Lab 3 `main` at commit
[`46d6cc3`](https://github.com/isaac-sim/IsaacLab/commit/46d6cc3d7b115906fad12efd372fbf69f4e3d3ed)
(VERSION 3.0.0), unmodified, inside Isaac Lab's **kit-less** image (no Isaac Sim / Kit, no PhysX).
Newer `main` commits may work but are untested.

```bash
tools/setup.sh
```

which is equivalent to:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout 46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
(cd IsaacLab && docker build -f docker/Dockerfile.kitless -t isaac-lab-kitless:main-46d6cc3 .)
docker build -t supermimic:latest docker/     # + tmux, ffmpeg, gcc, NVIDIA EGL vendor file
```

Then start the long-lived container (repository mounted at `/workspace/supermimic`):

```bash
docker/run_container.sh 0 128g     # GPU index 0, 128 GB memory cap
```

### 2. Data: LAFAN1 motions and the G1 model

| What | Source | License | Goes to |
|---|---|---|---|
| 40 LAFAN1 clips retargeted to the 29-DoF G1 (30 fps CSV: root pos, root quat xyzw, 29 joint angles) | HF [`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), folder `g1/` | CC BY-NC-ND 4.0 (Ubisoft La Forge) | `data/lafan1_g1/*.csv` |
| G1 description `g1_29dof_rev_1_0.urdf` + meshes (the model the clips were retargeted to) | same dataset, folder `robot_description/g1/` | BSD-3-Clause (Unitree) | `data/g1_description/` |

With the container from step 1 running:

```bash
tools/download_data.sh
```

This also caches Isaac Lab's official ground plane and textures in `data/ground/`.
For existing datasets, run `bash tools/download_ground.sh` once. This avoids a startup
stall in OmniClient's S3 download. No Docker rebuild is needed.

It downloads both, then converts the robot inside the container:
`tools/urdf_to_mjcf.py` compiles the URDF into `data/g1_description/g1_29dof.xml` (free-floating
pelvis), and `tools/mjcf_to_usd.py` converts that MJCF to `data/g1_usd/g1_29dof/g1_29dof.usda`,
which Isaac Lab spawns. The conversion is offline because the kit-less importers cannot run inside
an Isaac Lab process (see Deviations). Expected result:

```
data/
├── lafan1_g1/                  40 CSVs (dance, fallAndGetUp, fight, jumps, run, sprint, walk)
├── g1_description/
│   ├── g1_29dof_rev_1_0.urdf
│   ├── g1_29dof.xml            generated MJCF
│   └── meshes/*.STL            38 meshes
└── g1_usd/g1_29dof/            generated USD (+ payloads/)
```

Check: `ls data/lafan1_g1 | wc -l` prints `40`, and the clips total 264,705 frames at 30 fps
(2.45 h). `tools/test_env.py` checks the converted robot against the reference motions (FK vs.
simulator should agree to ~1e-6):

```bash
docker exec supermimic python tools/test_env.py
```

## Usage

```bash
CPUS=0-23 ./run_train.sh lafan1_main          # training in tmux session 'lafan1_main' (optional CPU pinning)
docker exec -it supermimic tmux attach -t lafan1_main
docker exec supermimic tmux new -d -s eval_main "cd /workspace/supermimic && python supermimic/evaluate.py --run_dir logs/lafan1_main --watch"
docker exec supermimic python tools/plot.py logs/lafan1_main --out results/success_curve.png
docker exec supermimic python supermimic/evaluate.py --run_dir logs/lafan1_main --ckpt logs/lafan1_main/ckpt_014000.pt --record traj.npz
docker exec supermimic python tools/render.py traj.npz out.mp4 --clips dance2_subject1 --seconds 10
```

The first iterations are slow while `torch.compile` builds its kernels (cached in `.cache/`).
Checkpoints are written every 2000 iterations to `logs/<run>/`; the eval watcher scores each on all
40 clips and appends to `logs/<run>/eval.csv`.

### Community-trained checkpoint (Not Official)

[G1 policy weights on Hugging Face](https://huggingface.co/pantelismanouselis/g1-supermimic-lafan1),
published by Pantelis Manouselis, include evaluation results and motion videos.
This separate training run reached normalized return **0.8696**, with **0/40 failed training clips**.
The inference-only release uses Safetensors. Follow the
[compatible fork's loading instructions](https://github.com/Manouselis/supermimic_vibed_fork/blob/8bdf60a74c700720817669539e160f4f846b0cf4/docs/RELEASE.md).
The local `.pt` workflow above is unchanged.

## Result (LAFAN1, 40 clips, one RTX PRO 6000 Blackwell)

Evaluation return 0.824 at 1.24 h of training (paper: 0.82 at 1.2 h on an H100) and 0.829 with
**0/40 failed clips** at 1.47 h (iteration 14000).

![success curve](results/success_curve.png)

## Implementation (paper section -> file)

* G1 29-DoF, BeyondMimic actuators (10 Hz natural frequency, damping ratio 2, motor armatures,
  action scale 0.25·τmax/Kp) — `supermimic/robot.py`.
* Reference library: 30 -> 50 Hz resampling, finite-difference velocities, FK of 14 tracked bodies,
  1 s bins — `supermimic/motion.py`, differentiable torch FK `supermimic/kinematics.py`
  (matches the simulator's link poses/velocities to 1e-6).
* MDP (App. A): observation (q−q0, q̇, IMU ω, projected gravity, a_{t−1}; 10-frame history; Table A.1
  noise), K=10 future reference frames as the goal, BeyondMimic tracking reward without the
  self-collision term and joint-limit weight −2, termination |e_z|>0.25 m or e_tilt>0.8, sim context
  ξ = (friction U[0.3,1.2], joint offsets U[±0.01], torso CoM offset) — `supermimic/mdp.py`.
* Simulator wrapper with return-based reset sampling (50% uniform / 40% softmax(−25 G_b/G_max) /
  10% past-softmax) — `supermimic/env.py`.
* World model f_ω: MLP(2048,2048,2048) on heading-invariant state features + action + ξ, predicting
  frame-local increments; fit on k=8-step autoregressive windows (Eq. 1); analytic reward,
  observation and termination heads — `supermimic/networks.py`, `supermimic/train.py`.
* Policy: motion encoder MLP(2048,1024,512,256)->z∈R^32 (tanh), shared decoder
  MLP(2048,1024,512,256); critic MLP(256,128). BPTT over H=32 imagined steps from replay-buffer
  states with a soft survival weight and critic bootstrap (Eq. 2); critic on GAE(λ=0.95) targets
  (Eq. 3), 5 epochs × 4 minibatches.
* Hyperparameters as Table A.5: 4096 envs, buffer 1000 steps/env, action noise N(0, 0.1²),
  UTD 0.4 (2.5 vectorized sim steps per iteration), 10 WM batches of 2048 windows per iteration,
  4096 imagined rollouts, Adam lr 2e-4 (policy β=(0.9,0.95)), cosine decay to 0.03× over 250k
  iterations, grad clip 1.0 (+ element-wise 0.1 for the policy), γ=0.99.
* Evaluation (App. F.1): each clip once from its first frame, friction 1.0, no DR/noise,
  deterministic policy; failure = training termination — `supermimic/evaluate.py`.

## Deviations / choices not specified by the paper

* **Self-collisions disabled** in the simulator (speed); robot–ground contacts use the URDF's
  convex mesh hulls and foot spheres.
* **bf16 autocast for MLP matmuls** (paper: TF32). On this RTX PRO 6000 it is 24 % faster; all
  state integration, FK and rewards stay fp32.
* Reward, observation and termination heads are **analytic** (allowed by §3.2); termination in
  imagination is a sigmoid-softened version of the hard thresholds.
* The policy observation additionally contains the reference root orientation relative to the
  robot root (6D), i.e. BeyondMimic's `motion_anchor_ori_b` — IMU-observable; no global position.
* 500 world-model-only warm-up iterations before policy updates start.
* Torso-CoM and friction are randomized once per environment at start-up (BeyondMimic's
  start-up events); joint offsets are resampled every episode. Episodes time out after 10 s or at
  the clip end (bootstrapped, not terminal).
* Do not run under a virtual-memory `ulimit -v`: Isaac Lab + Warp reserve ~430 GB of *virtual*
  address space (CUDA VA reservations; RSS ~3 GB) and Warp's CUDA-graph capture needs its mempool,
  which cannot initialize under such a limit. Cap memory with the container's cgroup
  (`--memory`) instead. Rootless docker ignores `--cpuset-cpus`; pin CPUs with `taskset`
  (`CPUS=...` in `run_train.sh`).
* Asset import: the kit-less URDF/MJCF importers fail inside a running Isaac Lab process
  (`PhysxJointAPI` schema not registered), so the USD is converted once, offline, by
  `tools/mjcf_to_usd.py` (registers the importer's schema plugin first) and spawned with
  `UsdFileCfg(variants={"Physics": "mujoco"})`.

## Contact planner on box terrain (G1, mjlab)

`supermimic/acp/` implements *Humanoid Mobility via Attention-based Contacts* (He, Wen, Zhang et al.,
arXiv:2610.31608) with SuperMimic training (no PPO), on the PHP parkour G1 bundles. It is a self-contained
package on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo-Warp) with the same G1 model, actuators and
sim context as the Isaac Lab code above. It runs outside the container (`pip install mjlab`, torch with CUDA).

* **Contact planner** (50 Hz): a 17 × 11 height scan around the pelvis (0.1 m cells, x ∈ [−0.8, 0.8],
  y ∈ [−0.5, 0.5]; exact terrain height from the clip's boxes, Gaussian noise 3 cm), 5 steps of proprioception
  and a goal → for each of 9 bodies (feet, knees, hands, elbows, pelvis) the terrain point of its next contact
  within 1.5 s, or "no contact". This is the paper's Table S3 attention network, trained on hindsight labels.
* **Contact tracker** (50 Hz): 10 steps of proprioception, the 9 targets, the goal command and the paper's
  9 × 7 local height scan (0.8 × 0.6 m) → joint position targets (`--blind` drops the scan).
* The goal command is the paper's: a bounded direction d_xy / max(‖d_xy‖, r) and a relative heading, with
  r = 0.3 m toward the reference 0.5–1.5 s ahead and r = 0.5 m toward a terrain goal.
* In every stage the world model and the critic also see an exact height scan (31 × 15 points ahead of the robot).

Every quantity is in the pelvis heading frame, so the pipeline needs no odometry. Each clip has 10 terrain
variants (boxes moved or resized slightly). Every episode draws one variant that the reference motion fits.

### Data

The five PHP bundles (`motions-{low,high}-step`, `motions-{low,high}-climb-76`, `motions-locomotion`: 296
G1 clips at 50 fps, 1.0 h; each `<clip>_motion.npz` paired with `<clip>_terrain.npy`) go under
`data/php_parkour/`, either copied or symlinked to their folder:

```bash
ln -s <php_parkour>/data data/php_parkour
python -m supermimic.acp.make_labels          # contact labels -> data/php_parkour_contacts (~1 min, 16 CPU workers)
```

### Training pipeline

| Step | Command | Output |
|---|---|---|
| 1. Tracker | `python -m supermimic.acp.train_stage1 --log_dir logs/acp_stage1` | `logs/acp_stage1/ckpt_*.pt`, `last.pt` |
| 2. Planner, offline | `python -m supermimic.acp.train_planner_offline --tracker logs/acp_stage1/ckpt_060000.pt --log_dir logs/acp_planner` | `logs/acp_planner/planner_020000.pt` |
| 3. Co-training | `python -m supermimic.acp.train_stage2 --init logs/acp_stage1/last.pt --planner_init logs/acp_planner/planner_020000.pt --log_dir logs/acp_stage2` | `logs/acp_stage2/ckpt_*.pt` (tracker `pi` + `planner`) |
| 4. Stage 3 | `python -m supermimic.acp.train_stage3 --init logs/acp_stage2/last.pt --log_dir logs/acp_stage3` | `logs/acp_stage3/ckpt_*.pt` |
| 5. Evaluate | `python -m supermimic.acp.evaluate --ckpt <ckpt> --planner <same ckpt> [--record traj.npz]`; `python -m supermimic.acp.evaluate_goal --ckpt <stage 3 ckpt>` | json next to the checkpoint |

* **Stage 1** follows the paper: the Stage 1 reward (reference tracking, an asymmetric contact-label term,
  contact positions, regularization), the backward curriculum, and targets on the touched terrain points of
  contacts that last at least 10 steps.
* **Offline planner:** the frozen tracker runs from uniform start phases. Successful episodes (started in the
  first 10 % of their clip, reached 90 %) are labeled in hindsight. Every 10th clip of each folder is held out for
  validation (`planner_val.csv`).
* **Stage 2:** η_obs hands the tracker's targets over to the planner (from iteration 2k, over 20k iterations).
  η_rew moves the reward to the planned contacts (from 12k). The planner keeps relabeling from the tracker's
  new behavior.
* **Stage 3** (the paper's Fig. 3) trains on reference-following and goal-reaching environments in one run
  (`--goal_frac 0.5`).
  * Reference environments start every clip from its first frame, with reward η_rew = 1.
  * Goal environments use procedural box terrains with 9 types and 10 levels each (`supermimic/acp/goal.py`).
    The types are stairs up and down, walled stairs, platforms of random orientation, climbs, gaps, stepping
    stones, balance beams and pallets. Each terrain has a reachable goal, and episodes last at most 15 s.
  * The goal reward is move2goal 40, standatgoal 20 and the planned-contact term 1.5, plus the style and
    regularization terms. Its indicators are made smooth, as sigmoids.
  * There are two critics, V_m and V_fs. V_fs warms up for 500 iterations and trains at a fixed lr of 1e-3.
    The return of each group is scaled separately.
  * The goal curriculum promotes a level on a success with success EMA > 0.5, and demotes it when the closest
    approach to the goal stays beyond 1 m.
  * The planner also learns from goal episodes that come within 0.5 m / 0.1 m of the goal. These samples are
    kept in a buffer balanced by terrain type.
  * Not built: ladders, get-up terrains, the stumbling term (it needs contact forces). Falls end an episode
    when the pelvis is less than 0.25 m above the terrain or the tilt exceeds 0.8. These thresholds sit just
    beyond the references' 0.345 m and 0.695; the paper gives none.
* Settings travel in each checkpoint (`ctc_cfg`, `reward_cfg`, `planner_cfg`), and later scripts rebuild
  them from there.
* A replay buffer of 1350 rows × 4096 environments needs about 7.5 GB of GPU memory, since it must hold the
  longest episode (1326 frames).
* Open-loop, the G1 does not stand with these soft BeyondMimic gains: holding the reference pose, it tips over
  within about a second. Short episodes under an untrained policy are expected.

## Attribution and licenses

* Method: Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath,
  *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*,
  CoRL 2026, arXiv:2610.28480. This code is an independent re-implementation, not affiliated with
  the authors.
* Isaac Lab (BSD-3-Clause), Newton and MuJoCo-Warp (Apache-2.0) are used unmodified.
* LAFAN1 motions (Ubisoft La Forge): CC BY-NC-ND 4.0; G1 retargets and description by Unitree
  (BSD-3-Clause). The motion data is not redistributed here; the demo GIF shows the
  reference motions for illustration only and is not covered by this repository's MIT license.

## License

The code in this repository is released under the [MIT License](LICENSE).

Copyright (c) 2026 ETH Zurich, Chong Zhang.
