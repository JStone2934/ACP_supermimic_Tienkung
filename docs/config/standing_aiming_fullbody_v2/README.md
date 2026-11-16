# 全身站立瞄准 v2：运行记录

## 2026-11-15 用户要求停止并诊断

H16 服务已正常停止并保存第 47103 轮检查点，未恢复训练。最后接纳策略为第 17010 轮，
随后连续 786 批被拒绝；仍在 A 阶段，约 1.36 秒倒地。有限物理复核、完整训练过程和
原因分析见 [H16 站立失败报告](../../report/20261115_h16_standing_failure/README.md)。
TensorBoard 服务保留用于查看历史曲线，训练日志写入器已停止。下面保留此前启动记录。

## 2026-11-14 H16、4096 critic 环境与新奖励续训

当前配置为 [h16_physical_dual_aggressive.json](h16_physical_dual_aggressive.json)：
策略想象 H=16，WM 训练窗口仍为 8，历史仍为 10 帧；GPU 0 为 4096 环境，
GPU 1 为 4096 个完整回报环境，其中 3072 训练、1024 验证，critic batch 仍为 4096。
标准 FP32，AMP/TF32 关闭；训练轮数、actor 更新预算和运行时长无限制。
训练服务不设单独的内存或交换空间上限，主机统一保护保持原样。

旧 H8 服务在第 4795 轮正常保存并退出。新运行
`logs/tienkung3_29dof/standing_aiming_fullbody_v2/dual_h16_critic4096_reward_v2_20261114`
保留 actor、WM 权重及 WM 优化器；奖励变化后重建 critic，清空 actor 的旧奖励动量，
保持切换时 actor 学习率 `1e-7`。累计训练计数保留，旧回报基准清空，先重新校准 critic
并补做完整评估。迁移源、新初始检查点、差异清单及冻结源码均保存在新运行目录中。

服务：`aiming-fullbody-h16-physical.service`，异常后从本运行最新 `train/last.pt` 恢复。
TensorBoard 仍为 `http://localhost:6007`，新曲线 `H16_critic4096_reward_v2`，旧曲线保留。
脚位奖励尺度 10 cm 不改变正式验收的脚漂移 5 cm 上限；完整 30 秒验收保持原样。
43 项 physical、fullbody 和 TensorBoard 检查通过，包含奖励迁移与旧状态隔离检查。
启动后已完成真实 4096 环境采集与 critic 校准，H16 WM 诊断通过；截至第 4820 轮，
新配置的前 5 批策略更新均被物理检查接纳，服务无异常重启，TensorBoard 已更新。
启动时的完整站立验收仍未通过，以上只确认新配置已正常训练。核验结果保存在运行目录
`verification.json`，权重迁移一致性记录在 `migration/verification.json`。

## 2026-11-14 奖励参数调整

项目中的四份 v2 配置已更新：双脚位置权重 `feet=2.0`，位置奖励尺度
`feet_position_scale_m=0.1`（10 cm），瞄准权重 `aim=4.0`。
A 阶段和回采的 A 样本仍使用零瞄准权重，B/C 样本使用配置中的瞄准权重。
开启瞄准后的正奖励权重总和为 12.2，每步上界为 0.244；A 阶段为 0.164。

脚位尺度现在随任务元数据保存，并参与 checkpoint 兼容性检查。旧配置缺少该字段时，
按历史实现的 5 cm 读取；旧 checkpoint 中的显式奖励权重继续保留。
因此修改项目配置不会改变已冻结运行或直接 `--resume` 的奖励；应用新奖励需要显式迁移训练。
该次奖励调整时尚未切换运行中的训练，37 项 fullbody/physical 契约检查通过；
随后已按上节迁移并启动 H16 训练。

2026-11-14：用户已停止旧 H=32 训练，并授权修复超限请求梯度、阶段探索后，以 **H=8** 重新训练。新实验使用 [h8_request_stage.json](h8_request_stage.json)，实现、验证和运行位置见 [H=8 说明](h8_restart.md)。下面保留 11 月 13 日 H=32 实验的历史参数与操作记录，不代表当前运行状态。

## 2026-11-13 H=32 历史记录

更新：2026-11-13（Asia/Shanghai）。[实施规格](../../plan/standing_aiming_fullbody_v2/plan.md) · [任务记录](../../plan/standing_aiming_fullbody_v2/taskmemory.md) · [默认参数](default.json)。

已按用户要求启动 H=32 正式训练，服务为 `aiming-fullbody-v2.service`。旧 `aiming-training.service` 和检查定时器已停止，旧模型与日志保留。当前训练从 A 阶段开始；启动及短闭环通过不代表物理验收通过。

## 生效参数

| 项目 | 正式配置 |
|---|---|
| 并行环境 / WM batch / actor batch | 512 / 256 / 256 |
| 想象长度 / WM 监督窗口 / 本体历史 | 32 / 8 / 10 |
| 初始真实采集 / 每轮真实采集 | 256 / 3 个向量控制步 |
| replay 长度 | 1000 个向量控制步 |
| 总预算 | 无上限（iterations=0），持续推进课程，最终独立验收通过后结束 |
| 第 1–1000 轮 | 仅 WM，每轮 10 次更新 |
| 第 1001–1100 轮 | WM 每轮 10 次，critic 每轮 5 epochs × 4 minibatches；actor 不更新 |
| 第 1101 轮起 | 每轮 WM 10 次、actor 1 次、critic 20 次 |
| actor / WM / critic 学习率 | 1e-5 / 1e-4 / 1e-4 |
| actor Adam / 梯度裁剪 | betas=(0.9,0.95)；先全局范数 1，再逐元素 ±0.1 |
| gamma / lambda-return | 0.99 / 0.95 |
| 探索噪声 | 右臂 0.02 rad，其余 0.005 rad；按 actor 更新次数线性衰减，10000 次尺度，最低保留初值 10% |
| WM 独立诊断 | 每 100 轮，16 环境 × 512 步，检查 1/8/16/32 步 |
| 物理评估 / 保存 | 每 500 轮 |
| 控制台进度 | 每 100 轮及阶段切换；详细数值写入文件 |

总轮数及累计 optimizer 更新次数不设上限。10000 仅作为探索噪声衰减尺度，达到后仍保留 10% 噪声并继续训练。恢复运行会先重建 replay 并做 100 轮 WM 预热，因此实际累计更新数以日志和 checkpoint 为准。

WM 检查只作诊断，不再因未达到工程阈值无限暂停 actor，也不要求先通过 H=8 才使用 H=32。A 阶段诊断根/双脚，B/C 额外诊断指向误差。根 P95 5 cm、脚 P95 2.5 cm、目标距离折算方向 P95 `min(10 cm,rho/2)` 是诊断阈值，不能替代物理验收。非有限状态或梯度仍立即报错。

## 当前运行与恢复

运行目录：`logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113/`。

- `config.json`：启动参数副本；`train/config.json`：训练器生效参数。
- `launch.json`：服务、固定初始化来源及启动身份。
- `stdout.log`：简短进度和错误；`train/status.json`：最近一次进度；`train/metrics.jsonl`：详细指标。
- `train/wm_validation/`：独立轨迹预测诊断；`train/evaluation/`：物理评估。
- 每 500 轮生成 `train/last.pt`、`ckpt_*.pt`，并按评估结果维护 `best.pt`。
- `train/code_identity.json`：包含未提交代码的文件哈希，不能只用 Git HEAD 代表运行版本。

当前服务已从第 500 轮正式 checkpoint 恢复，启动参数显式包含 `--iterations 0`。服务独立于前端运行，使用现有 Python 环境及主机资源限制，不增加监控/重启控制器。

```bash
systemctl --user status aiming-fullbody-v2.service
cat logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113/train/status.json
```

固定初始化为 `preflight_20261113/train/init.pt`，SHA-256 为 `a9743084c5b6ed1ae1781198b31d04d7c626bd5281d6123da6e80cc2af281c99`；actor/WM 来自同一源快照。正式运行不使用短测试训练出的权重。全新实验命令如下，输出目录必须为空；当前服务运行时不要重复启动：

```bash
PY=/home/js/miniconda3/envs/acp_supermimic/bin/python
$PY -u -m supermimic.acp.train_aiming_fullbody \
  --config docs/config/standing_aiming_fullbody_v2/default.json \
  --init logs/tienkung3_29dof/standing_aiming_fullbody_v2/preflight_20261113/train/init.pt \
  --log-dir logs/tienkung3_29dof/standing_aiming_fullbody_v2/NEW_RUN/train
```

恢复时先停止现有服务，再用同一运行目录的 `last.pt`。默认以 checkpoint 内的配置为准；`--iterations 0` 可覆盖旧 checkpoint 的总轮数上限；兼容早期 v2 checkpoint 的缺省字段及旧优化器状态，支持 `--log_dir` 别名。旧任务 checkpoint 必须先迁移，不能直接 resume；smoke checkpoint 不能用于正式运行。目录互斥锁防止重复写入。

```bash
systemctl --user stop aiming-fullbody-v2.service
PY=/home/js/miniconda3/envs/acp_supermimic/bin/python
RUN=logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113
$PY -u -m supermimic.acp.train_aiming_fullbody --resume "$RUN/train/last.pt" --iterations 0 --log-dir "$RUN/train"
```

## 实现与验收

单 actor 输出全部 29 关节，右臂七维均可学习，统一按 RobotSpec 的缩放、偏置和软限位执行；replay、WM、历史及动作变化奖励使用实际执行动作。观测为 `10×99`，指令为 107 维；源统计冻结，新增维度显式缩放。平地、静态世界目标、零速度指令，无 motion/contacts/IK 读取、固定肩肘或教师约束。实际和想象使用相同奖励与任务上下文。

课程为 A→B1→B2→C1a→C1b→C2a→C2b→C3a→C3b，每阶段固定测试连续两次全部通过才升级，保留 20% 上一阶段样本。每次评估完整记录 30 秒、1501 帧，不重置、不删除初始调整段。根漂移 10 cm、每脚漂移 5 cm、倾角 15° 及原捕获/保持要求不变。最终 27 点通过、课程完成并独立进程重载复验后才记录 `passed=true`。

两种初始化源均未通过 v2 的 A 阶段站立验收，历史结果见 [任务记录第 10 节](../../plan/standing_aiming_fullbody_v2/taskmemory.md)。长想象能否提高物理表现仍需本次训练验证。

## 已完成检查

本轮 10 项 CPU 契约测试、真实物理 reset/timeout/动作奖励配对、H=32 critic-only 与 actor 梯度检查通过。critic 预热不修改 actor/WM；actor 更新梯度有限并正确裁剪。

`h32_20261113/checks/capacity/` 用 512 环境、batch 256/256、H=32 跑完 5 轮短闭环：2 轮 WM、1 轮 critic、2 轮联合更新。该测试每轮仅 1 次 WM、1 次 critic 更新，不是正式训练吞吐基准。PyTorch allocated 峰值约 1.53 GiB，不包含 Warp 分配。`checks/resume/` 从第 5 轮恢复到第 10 轮，累计 actor 更新 5 次，未重复已完成的 critic 预热。

```bash
PY=/home/js/miniconda3/envs/acp_supermimic/bin/python
INIT=logs/tienkung3_29dof/standing_aiming_fullbody_v2/preflight_20261113/train/init.pt
$PY -m tools.test_aiming_fullbody --physics --checkpoint "$INIT"
$PY -m supermimic.acp.train_aiming_fullbody --init "$INIT" --log-dir /tmp/NEW_SMOKE --smoke
# 正式环境数和 batch 的 5 轮短检查，使用新的输出目录：
$PY -m supermimic.acp.train_aiming_fullbody --init "$INIT" --log-dir /tmp/NEW_CAPACITY --smoke --capacity-check
$PY -m tools.validate_aiming_fullbody_wm --help
$PY -m tools.evaluate_aiming_fullbody --help
$PY -m tools.export_aiming_fullbody --help
```
