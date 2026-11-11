# Stage1 → Stage2 衔接手册（自动看护用）

2026-11-10 21:30 建立。当前 Stage1 正常训练，**Stage2 尚未启动**。本文件是后台看护任务在 Stage1 结束后执行
Stage1 全量评估 → 离线 planner → Stage2 联合训练的固定流程；任何一步失败都停下来报告，不重复启动同名作业。

## 0. 门禁

1. `systemctl --user show acp-stage1-php-all-20261110.service -p ActiveState` 为 `inactive`，
   且 `logs/tienkung3_29dof/stage1_php_all_spheres224_20261110/status.json` 的 `stage` 为
   `training_finished_evaluation_pending`（或 `training`/`completed` 且迭代数达到 60000）。
2. `systemctl --user list-units 'acp*' --state=active` 中没有其他训练单元；同一时间只允许一个所有者。
3. 开始前记录 `free -g`、服务 `MemoryCurrent`、`systemd-run --user --wait --pipe nvidia-smi` 的两卡显存/温度。

路径约定：

```bash
ROOT=/home/js/robot/locomotion/ACP_supermimic_Tienkung
S1=$ROOT/logs/tienkung3_29dof/stage1_php_all_spheres224_20261110
S2=$ROOT/logs/tienkung3_29dof/stage2_php_all_spheres224_20261110
PY=/home/js/miniconda3/envs/acp_supermimic/bin/python
```

## 1. Stage1 全量评估（单卡 cuda:0）

```bash
systemd-run --user --unit=acp-stage1-final-eval-20261111 \
  --slice=php-compute.slice \
  --property=WorkingDirectory=$ROOT \
  --property=MemoryHigh=48G --property=MemoryMax=64G \
  --property=Restart=no --property=KillMode=control-group --property=TimeoutStopSec=120 \
  --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=OMP_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MUJOCO_GL=egl --setenv=PYTHONUNBUFFERED=1 \
  /bin/bash -lc "exec $PY -m supermimic.acp.evaluate \
    --ckpt $S1/train/last.pt --out $S1/stage1_final_eval.json \
    > $S1/stage1_final_eval.stdout.log 2>&1"
```

`evaluate` 一次把所有 2960 条片段作为并行环境在 cuda:0 上跑完；按 60 条诊断评估外推约 30–60 分钟。
若显存不足（journal 里 CUDA OOM）：改为按 5 个动作类别分批，用符号链接建子目录后逐类评估，
再把 `n_failed`/`norm_return` 汇总；不要删片段或改终止窗口。

评估结果 `stage1_final_eval.json` 关键字段：`norm_return`、`n_failed`、`failed`、`per_folder`、`contacts`。

**放行规则**：`norm_return >= 0.1` 且失败片段不超过 80% → 继续；否则暂停并向用户报告，不启动 Stage2。
若 `last.pt` 明显差于近期编号检查点，可加评 `ckpt_059000.pt` 等，按“失败少、回报高”选定，并记录选择依据。
**选定哪个断点，就要把下文离线 planner 和 Stage2 命令里的 `$S1/train/last.pt` 全部替换为该断点。**

## 2. 离线 planner（双卡，20000 步）

```bash
mkdir -p $S2/offline
systemd-run --user --unit=acp-stage2-offline-20261111 \
  --slice=php-compute.slice \
  --property=WorkingDirectory=$ROOT \
  --property=MemoryHigh=64G --property=MemoryMax=80G \
  --property=Restart=no --property=KillMode=control-group --property=TimeoutStopSec=120 \
  --setenv=CUDA_VISIBLE_DEVICES=0,1 --setenv=OMP_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MUJOCO_GL=egl --setenv=PYTHONUNBUFFERED=1 \
  /bin/bash -lc "exec $PY -m torch.distributed.run --nnodes=1 --nproc-per-node=2 \
    --master-addr=127.0.0.1 --master-port=29571 --max-restarts=0 \
    --module supermimic.acp.train_planner_offline \
    --robot tienkung3_29dof --tracker $S1/train/last.pt \
    --num_envs 512 --buffer_steps 1350 --planner_batch 512 --planner_sub_batches 8 \
    --capacity 300000 --steps 20000 --seed 0 --ckpt_every 500 --verify_sync \
    --log_dir $S2/offline > $S2/offline.stdout.log 2>&1"
```

完成后确认存在 `$S2/offline/planner_020000.pt`，并检查 `planner_val.csv` 的 F1 / 落点误差。

## 3. Stage2 联合训练（双卡，先 24000 轮，再续到 40000 轮）

第一段（严格参考终止，η_obs 2k–22k，η_rew 12k–32k）：

```bash
mkdir -p $S2/joint
systemd-run --user --unit=acp-stage2-joint-20261111 \
  --slice=php-compute.slice \
  --property=WorkingDirectory=$ROOT \
  --property=MemoryHigh=64G --property=MemoryMax=80G \
  --property=Restart=no --property=KillMode=control-group --property=TimeoutStopSec=120 \
  --setenv=CUDA_VISIBLE_DEVICES=0,1 --setenv=OMP_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MUJOCO_GL=egl --setenv=PYTHONUNBUFFERED=1 \
  /bin/bash -lc "exec $PY -m torch.distributed.run --nnodes=1 --nproc-per-node=2 \
    --master-addr=127.0.0.1 --master-port=29573 --max-restarts=0 \
    --module supermimic.acp.train_stage2 \
    --robot tienkung3_29dof --init $S1/train/last.pt \
    --planner_init $S2/offline/planner_020000.pt \
    --num_envs 512 --wm_batch 256 --pi_batch 256 --reward_chunk_size 0 \
    --buffer_steps 1350 --horizon 32 --wm_window 8 --wm_batches 10 --utd 0.4 \
    --lr 2e-4 --amp bf16 --critic_epochs 5 --critic_minibatches 4 \
    --planner_batch 512 --planner_sub_batches 8 --planner_capacity 300000 \
    --term_window 0 --iterations 24000 --ckpt_every 500 --log_every 25 \
    --save_full_checkpoints --verify_sync --benchmark_warmup 25 --benchmark_steps 50 \
    --seed 0 --log_dir $S2/joint > $S2/joint.stdout.log 2>&1"
```

第一段结束后（24k，检查 `train_log.csv` 中 η_obs≈1、admitted 非零、planner 指标有限），
用**新的 unit 名** `acp-stage2-joint-window-20261111` 重启（旧 unit 已 inactive 但名字仍占用），
其余参数不变，追加 `--resume $S2/joint/last.pt --iterations 40000 --term_window 2 --eta_rew_max 0.58`。
这一段是上游 README 的 Stage2 窗口，
放宽 ±2 s 终止以避免把“规划器按自己节奏走”误判为失败。

Stage2 训练完成后再跑：`supermimic.acp.evaluate --ckpt $S2/joint/last.pt --planner $S2/joint/last.pt --train_noise action`。

## 4. 内存与 SSH 规则

- 所有重任务都用 `systemd-run --user` 放进 `php-compute.slice`，带 `MemoryHigh=64G`、`MemoryMax=80G`；
  不用 `nohup`/`tmux`/前台 SSH 会话承载训练。SSH 或 VS Code 断开不影响 systemd 单元。
- 启动前确认没有同名 unit、没有第二份 `train_stage1/train_stage2/bench_*`；异常时先 `systemctl --user status`
  和 `journalctl --user -u <unit>`，不要直接 kill 进程。
- 每 30 分钟看护会检查：服务状态、`MemoryCurrent`/`MemoryPeak`、`free -g`、两卡显存/温度、日志是否停滞。
  主机可用内存低于 8 GiB、单卡显存超过 30 GiB 或温度超过 82 °C 时停止新作业并报告。
- 停止用 `systemctl --user stop <unit>`；恢复前确认旧进程已退出、输入/哈希未变，再用 `--resume`。
