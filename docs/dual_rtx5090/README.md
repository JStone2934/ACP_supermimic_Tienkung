# 双 RTX 5090 测试档案

测试日期：2026-11-08。分支：`feat/dual-5090-scheduling`。本目录保留环境、训练规模、显存口径和测试证据，供后续训练配置参考。启动与恢复接口见 [ACP 双卡训练指南](../tienkung3_29dof/dual_gpu.md)。

| 文件 | 内容 |
|---|---|
| [performance.md](performance.md) | 四阶段单/双卡吞吐、50 轮耗时和逐卡显存对照 |
| [environment.json](environment.json) | 硬件、驱动、包版本、通信及线程设置 |
| [results.json](results.json) | 原始性能窗口、512 环境资源检查、优化器计数、状态哈希及恢复结果 |
| [commands.json](commands.json) | 各次实际运行的参数数组及 GPU 可见性设置 |
| [source_manifest.json](source_manifest.json) | 最终工作区代码和测试输入的 SHA-256，便于区分后续版本 |

## 环境

| 项目 | 实测配置 |
|---|---|
| GPU | 2 × NVIDIA GeForce RTX 5090，每卡 32607 MiB |
| 驱动 / PyTorch CUDA 构建 | 580.178.04 / CUDA 12.8 |
| CPU / 内存 | Xeon w9-3475X，72 逻辑线程 / 约 125.4 GiB |
| 系统 | Ubuntu 22.04.5，Linux 6.8.0-138-generic |
| Conda / Python | `acp_supermimic` / 3.11.18 |
| PyTorch / Triton / NCCL | 2.9.0+cu128 / 3.5.0 / 2.27.5 |
| mjlab / MuJoCo / mujoco-warp / Warp | 1.6.0 / 3.11.0 / 3.11.0 / 1.16.0 |
| GPU 拓扑 | NODE，同 NUMA；无 NVLink，本机 CUDA P2P 不可用 |
| 通信 | NCCL，一卡一进程，`127.0.0.1:29547` |
| CPU 线程 | 每进程 OMP 4、OpenBLAS 1、Inductor 编译线程 8 |

详细记录见 [environment.json](environment.json)。本项目实际运行栈为 mjlab/MuJoCo/Warp，这批测试未使用 Isaac Lab。GPU 0 驱动桌面，空闲占用约 790 MiB；单卡性能对照使用 GPU 1。所有性能作业依次运行。

## 显存口径

所有表格以 **GiB = 2³⁰ 字节**计量，原始 JSON 保留字节值。

- **PyTorch allocated peak**：PyTorch allocator 记录的张量分配峰值。性能测试在预热结束后重置；512 环境短训记录进程内峰值。
- **PyTorch reserved peak**：PyTorch 内存池保留峰值，包含缓存空间。
- **Warp pool peak**：Warp CUDA 内存池的进程内高水位，包含编译、预热和 replay 填充。
- **设备占用**：CUDA 查询到的整张卡占用，包含 PyTorch、Warp、CUDA 上下文及桌面等其他进程。性能测试为训练循环边界的最大采样值；512 环境短训为保存检查点时的值，均不是瞬时峰值。

这些峰值可能出现在不同时刻，不能将它们直接相加。两卡各自保存模型和优化器，显存余量需逐卡判断。

## 512 环境配置检查

四个阶段均完成实际仿真、反向传播和优化器更新，两卡状态校验一致，无非有限梯度跳过。全局规模为环境 512、world-model/policy batch 各 256、planner batch 512、planner 子批次数 8、最少样本 1000；每卡环境 256。奖励分块大小 64。

在线阶段运行 3 轮，离线 planner 运行 20 步。保持 horizon 32、world-model 窗口 8、每轮 10 次 world-model 更新及 5×4 次 critic 更新；Stage 3 第 3 轮已包含 goal actor 更新。

| 阶段 | GPU | PyTorch allocated peak | PyTorch reserved peak | Warp pool peak | 保存时设备占用 |
|---|---:|---:|---:|---:|---:|
| Stage 1 | 0 | 6.69 | 8.83 | 0.33 | 11.38 |
| Stage 1 | 1 | 6.69 | 8.83 | 0.33 | 10.64 |
| 离线 planner | 0 | 3.27 | 4.21 | 0.33 | 6.77 |
| 离线 planner | 1 | 3.27 | 4.89 | 0.33 | 6.70 |
| Stage 2 | 0 | 10.64 | 13.52 | 0.33 | 16.09 |
| Stage 2 | 1 | 10.64 | 13.52 | 0.33 | 15.35 |
| Stage 3 | 0 | 15.30 | 16.02 | 0.33 | 18.59 |
| Stage 3 | 1 | 15.30 | 16.30 | 0.33 | 18.13 |

| 阶段 | 完成的优化器更新次数 |
|---|---|
| Stage 1 | world-model 30、policy 3、critic 60 |
| 离线 planner | planner 9 |
| Stage 2 | world-model 30、policy 3、critic 60、planner 3 |
| Stage 3 | world-model 30、policy 3、reference critic 60、goal critic 60、planner 3 |

replay 时间长度：Stage 1/离线 planner/Stage 2 为 1000，Stage 3 为 1350；planner 全局容量 300000，Stage 3 的 reference、goal 两组各为 300000，地形库每类每等级 16 个实例。这是短训的显存实测，正式数据的片段长度、接触几何和地形会改变占用。

这批短训使用独立生成的 10 段 41 帧常值姿态及真实计算的接触标签。为了尽快覆盖代码路径，使用测试学习率 `1e-6`、立即启用 actor、放宽 planner admission 阈值。正式训练默认值未更改，结果不证明策略已学会站立、运动或目标到达。

## 正确性与恢复

| 检查 | 实测结果 |
|---|---|
| 回归测试 | 24 项通过；保留机器人及流水线回归 |
| 分布式梯度 | 两进程梯度与 Adam 更新匹配拼接 batch 基准；覆盖缺失梯度、非有限梯度共同跳过 |
| 归一化与课程 | 全局均值/方差、空标签 rank、延迟统计同步、课程增量与分桶回报通过 |
| planner | 不均衡有效接触数、单侧空标签、地形分组采样、合并验证统计通过；真实 planner 参数确有更新 |
| 四阶段短训 | 真实 MuJoCo/Warp 仿真、反向及优化器更新通过；Stage 1/2/3 覆盖编译反向，离线 planner 沿用 eager FP32；模型、优化器、共享调度状态哈希一致 |
| 恢复 | 四阶段各完成 2→2、2→1、1→2，共 12 项；恢复后计数继续增长且优化器更新继续发生 |
| 旧检查点 | 实际旧单卡 Stage 1 检查点恢复，以及双卡 `--init` 加载通过 |
| 评估 | Stage 1、Stage 2、Stage 3 检查点评估加载通过，未以策略成功率作为本次数值验收门槛 |
| 课程映射 | Stage 3 改变全局环境数时明确拒绝恢复，未写入新检查点 |
| 故障退出 | 注入真实 torchrun 子进程异常后作业失败；流水线回归确认不写完成标记、不进入下一阶段 |

恢复及评估共 18 项检查。恢复时重新填充 replay；不要求复现中断前的逐步采样轨迹。测试中的非有限梯度注入是受控负例，正式短训通过标准为跳过次数 0。

测试期间修复了两处显存问题：Stage 3 单卡大批几何奖励通过显式 `--reward_chunk_size 64` 分块重算；replay 重新填充时集中结束 episode 的 hindsight 几何改为固定 64 状态分块。前者有编译后梯度等价测试，后者有真实机器人接触标签等价测试，恢复与四阶段短训已覆盖实际路径。分块不减少训练 batch 或丢弃标签。

## 复现

在项目根目录运行：

```bash
conda activate acp_supermimic
python -m unittest tools.test_acp_distributed tools.test_acp_pipeline tools.test_robot_contracts

# 性能对照：每个阶段单卡、双卡各测 50 轮，之前预热 10 轮。
python -m tools.check_acp_distributed --stages stage1,planner,stage2 --label measured
python -m tools.check_acp_distributed --stages stage3 --label measured --reward-chunk-size 64

# 本机建议规模的资源检查，不启动长期训练。
python -m tools.check_acp_distributed --profile local512 --mode dual \
  --iterations 3 --benchmark-steps 0 --reward-chunk-size 64

# 已有检查点的 2→2、2→1、1→2 恢复及评估加载检查。
python -m tools.check_acp_distributed --resume-only --stages stage1,planner,stage2
python -m tools.check_acp_distributed --resume-only --stages stage3 --label memory
```

后两条恢复命令对应本机已完成的原始验收批次。若从新的目录复现上面的 `measured` 批次，恢复命令统一使用 `--label measured`。所有命令可添加 `--root <新目录>` 保留独立结果；已存在 `acceptance.json` 的阶段会跳过。

大型检查点和完整日志留在 `.cache/dual-5090/acceptance/`，不纳入版本控制。本目录保留小型 JSON 结果、精确命令和代码哈希，使清理缓存后仍可查阅测试依据。
