# 双 RTX 5090：ACP 同步训练

四个 ACP 入口支持单机多进程数据并行：每张 GPU 独立仿真和采样，在每次优化器更新前同步梯度。使用现有 `acp_supermimic` Conda 环境；不需要 Isaac Lab。通信使用 [PyTorch 2.9 的 NCCL 后端](https://docs.pytorch.org/docs/2.9/distributed.html)，不在编译图中插入 collective。

本机环境、显存、单/双卡性能对照及恢复验收记录见 [双 RTX 5090 测试档案](../dual_rtx5090/README.md)。

## 参数与启动

`--num_envs`、`--wm_batch`、`--pi_batch`、`--planner_batch` 和 replay 容量都是**全局总量**。例如双卡 `--num_envs 512` 为每卡 256 个环境。`--buffer_steps` 是每个环境的时间长度，不除以卡数；Stage 3 的 `--planner_capacity` 仍分别表示 reference、goal 两组的全局容量。更新次数、horizon、学习率和学习率调度不随卡数缩放。

环境数和训练 batch 必须能被卡数整除，planner batch 还须能被子批次数整除；Stage 3 的 reference/goal 环境和 imagination 两组也分别检查。非法配置会在建立仿真前报错。模型和优化器在每卡保留副本，双卡显存不能视为一块 64 GB 显存。

```bash
conda activate acp_supermimic
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29547 --max-restarts=0 \
  --module supermimic.acp.train_stage1 \
  --robot tienkung3_29dof \
  --motion_dir /path/to/motions --contacts_dir /path/to/contacts \
  --num_envs 512 --wm_batch 256 --pi_batch 256 --reward_chunk_size 64 \
  --log_dir logs/tienkung_dual/stage1
```

动作目录和接触标签需对应同一批已准备的数据。以上参数是本机配置起点，实际显存还受动作、horizon 和地形规模影响。默认 bootstrap 站立数据只用于数值检查，不能代替正式训练数据。

`--reward_chunk_size 64` 将几何奖励按 64 个状态分块，并在反向传播时重算中间量，以计算换显存。它不改变全局 batch、horizon、loss 权重或优化器更新次数。默认 `0` 保留原执行方式。

重新填充 replay 时，多个 episode 可能同时结束。hindsight 接触几何始终按 64 个状态分块计算，保留所有有效标签，防止这一步的临时张量随 episode 数量增长而耗尽显存。

本机数字主机名 `5090` 存在解析问题，使用显式 loopback 地址；同时运行的作业需使用不同端口。当前两卡没有 CUDA P2P，NCCL 使用可用传输路径，不强制修改 NCCL 环境变量。

其他三个入口使用同样的 `torchrun` 前缀：

| 入口 | 输入 | 附加规模参数示例 |
|---|---|---|
| `supermimic.acp.train_planner_offline` | `--tracker logs/tienkung_dual/stage1/last.pt` | `--num_envs 512 --planner_batch 512 --capacity 300000` |
| `supermimic.acp.train_stage2` | `--init <stage1/last.pt> --planner_init <planner/last.pt>` | `--num_envs 512 --wm_batch 256 --pi_batch 256 --planner_batch 512 --reward_chunk_size 64` |
| `supermimic.acp.train_stage3` | `--init <stage2/last.pt>` | `--num_envs 512 --wm_batch 256 --pi_batch 256 --planner_batch 512 --reward_chunk_size 64` |

三个入口都需要各自的 `--log_dir`。planner 的 `--planner_sub_batches` 和 `--planner_min_entries` 默认继承检查点或原配置；最少样本数依据全局缓冲区判断。某张卡暂无标签也能参与训练。

单卡调用保留原形式，也可直接选卡：

```bash
python -m supermimic.acp.train_stage1 --device cuda:1 ...
```

`--device` 只用于单进程；`torchrun` 模式根据 `LOCAL_RANK` 绑定设备。

## 流水线与恢复

```bash
python -m tools.train_acp_pipeline \
  --robot tienkung3_29dof --run-dir logs/tienkung_dual_pipeline \
  --motion-dir /path/to/motions --contacts-dir /path/to/contacts \
  --gpus 0,1 --master-port 29547 --through-stage 3 \
  --num-envs 512 --wm-batch 256 --pi-batch 256 \
  --planner-batch 512 --planner-capacity 300000 --reward-chunk-size 64
```

默认 `--through-stage 2` 保持原阶段顺序，选择 `3` 才追加 Stage 3 和目标任务评估。预处理与训练使用重叠 GPU 时串行执行；评估使用列表中的第一张卡。单卡流水线仍可使用 `--gpu 1 --prep-gpu 0`。任一进程失败会终止当前训练作业，流水线不写该阶段完成标记。

四个训练入口均支持 `--resume <last.pt>`。恢复时使用原训练参数，仅修改设备、输出目录以及目标迭代数；迭代数表示累计目标。离线 planner 的 `--steps` 同理，恢复时可省略 `--tracker`，程序会检查原 tracker 文件哈希。

例如，继续上面的 Stage 1 实验：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 \
  --master-port=29547 --max-restarts=0 \
  --module supermimic.acp.train_stage1 \
  --robot tienkung3_29dof --resume logs/tienkung_dual/stage1/last.pt \
  --motion_dir /path/to/motions --contacts_dir /path/to/contacts \
  --num_envs 512 --wm_batch 256 --pi_batch 256 --reward_chunk_size 64 \
  --iterations 80000 --log_dir logs/tienkung_dual/stage1
```

恢复到单卡时将 `torchrun ... --module` 换为 `python -m`，添加 `--device cuda:1`，保留全局规模参数。其余阶段同样使用各自的 `last.pt`，离线 planner 以 `--steps` 指定累计目标。

`last.pt` 保存完整模型、优化器、归一化统计、调度状态和全局采样计数；编号检查点保持原来的轻量/完整规则。只有 rank 0 原子写入公共文件，模型键名没有 `module.` 前缀，评估和渲染继续直接加载。

相同全局配置支持单卡与双卡互相恢复；Stage 3 的地形课程按全局环境编号保存、重新分片，改变环境数或 goal 分组数量会报错。启动新 Stage 3 实验应使用 Stage 2 检查点作为 `--init`。replay 和仿真状态不保存，恢复后重新采样填充，不提供逐位复现保证。

`updates.json` 记录优化器更新、非有限梯度跳过次数和各卡显存；`--verify_sync` 额外核对模型、优化器及共享课程/调度状态的哈希，适用于验收和排查，会增加检查点开销。

## 可复现验收

```bash
python -m unittest tools.test_acp_distributed tools.test_acp_pipeline tools.test_robot_contracts
python -m tools.check_acp_distributed --stages stage1,planner,stage2
python -m tools.check_acp_distributed --stages stage3 --label memory --reward-chunk-size 64
python -m tools.check_acp_distributed --resume-only --stages stage1,planner,stage2
python -m tools.check_acp_distributed --resume-only --stages stage3 --label memory
python -m tools.check_acp_distributed --profile local512 --mode dual \
  --iterations 3 --benchmark-steps 0 --reward-chunk-size 64
```

输出位于 `.cache/dual-5090/acceptance/`，包括每次调用命令、日志、完整检查点、`acceptance.json`、`benchmark.json` 和恢复报告。已完成的验收阶段会跳过；需要重新测量时使用新的 `--root` 或新的 `--label`。`local512` 是资源与更新检查，不生成性能窗口；它保持 horizon 32、world-model 窗口 8、每轮 10 次 world-model 更新和 5×4 次 critic 更新。

短训使用独立生成的 10 段 41 帧常值姿态及重新计算的接触标签，包含训练/验证切分。为在短时间内覆盖 planner 更新，测试将 hindsight 阈值设为 `start_max=1, reach_min=0, min_entries=16`；在线网络学习率为 `1e-6`，立即执行 actor 更新。测试不修改正式默认值，不能据此宣称已学会站立、运动或目标到达。

测试全局环境数 64，world-model/policy/planner batch 均为 64，horizon 8，4 个 CPU 线程/进程。world-model、policy、critic 的 MLP 使用 BF16，planner 训练保留 eager FP32；在线 planner 目标推理使用 FP16。单卡与双卡对照使用相同全局参数，后续阶段使用相同初始检查点。初始编译及前 10 次有效训练循环不计入性能窗口，测量后续 50 次；无有效 planner 更新不算通过。

`device_peak_used` 是窗口内各循环边界采样的设备总占用，包含 Warp、CUDA 上下文及桌面等其他进程；它不能捕捉循环内部瞬时峰值。PyTorch 峰值由 allocator 统计，`warp_pool_peak` 为进程内 Warp CUDA 内存池高水位，包含预热。两项峰值可能不同时发生，不直接相加。小批量结果用于检查通信与调度开销，不代表大规模训练加速比。
