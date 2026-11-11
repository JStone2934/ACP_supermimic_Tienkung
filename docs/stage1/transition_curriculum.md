# 攀爬失败过渡课程（仅第一步）

2026-11-11：按用户要求先提交诊断，再实现失败区域课程。诊断提交 `014f344`。训练仍保持停止；本次不改奖励、网络、动作含义、世界模型训练或正式评估终止条件，也没有启动续训。

## 采样与局部推进

显式传入 `--transition_curriculum configs/stage1_climb_transition_20261111.json` 才启用。默认入口仍使用原倒序课程，Stage2/3 和正式评估不启用该功能。

固定划分每张卡的并行环境，使短 episode 不会在 replay 中被长 episode 稀释：

| 环境/采集 transition 占比 | 任务 |
|---|---|
| 50% | 攀爬失败及接触切换局部窗口 |
| 25% | 攀爬从第一帧执行至片段末尾 |
| 12.5% | 攀爬已会的后段，从原课程起点或 60% 位置中较晚者开始 |
| 12.5% | 非攀爬动作，沿用检查点已有课程起点 |

配额按每卡环境数取整；正式 512 环境、2 GPU 时，每卡为 128/64/32/32。局部窗口先均匀选攀爬片段，再均匀选其事件，避免事件多的片段被过采样。局部 episode 的数量占比会高于 50%，但采集的 transition 配额保持 50%。replay 仍按原实现均匀采样，不改损失和更新次数。

事件计划来自 100k 的全量严格评估、原始同身份动作和接触标签。包含全部 800 条攀爬、4800 个事件窗口：失败位置、首次手接触，以及附近 1.5 秒内的脚/膝/手接触切换。原 2960 条输入没有删减；其余动作进入保留环境。

初始窗口通常从事件前 0.3 秒开始，到事件后 0.6 秒结束（末尾边界裁剪）。一个事件累计至少 20 次当前难度的完成尝试、其中至少 80% 达标后，起点前移 0.15 秒，按 50 Hz 四舍五入实际为 8 帧/0.16 秒；终点不移动。失败批次不把起点往后退。上一难度仍在运行的 episode 不计入下一难度的推进统计。

局部提前结束是 timeout/truncation，写入 replay 的 `done`，防止世界模型窗口跨 reset；原来的高度/倾斜失败 `term` 不变，也没有新增奖励。完整攀爬/保留样本不受默认 500 步上限截断。

## 局部达标门槛

到达窗口终点且没有原失败终止，并在最后 0.2 秒检查的帧中至少 80% 同时满足：

- 根部三维位置误差 ≤0.20 m；关节位置 RMSE ≤0.50 rad。
- 参考要求的接触覆盖率 ≥80%，手部与其他部位分开计算，避免双脚掩盖手部缺失。
- 接触距离代理 ≤0.02 m；连续参考接触部位的最近地形接触点位移速度代理 ≤0.20 m/s。

这是可配置的课程推进门槛，不是新 reward，不是正式任务验收，也不是实际接触力判据。最近接触点切换可能影响滑移代理，需要在实际实验中审查门槛拒绝情况。完整攀爬验收仍运行原 `supermimic.acp.evaluate`；局部达标不能替代从头完成。

`transition_curriculum.json` 记录三个环境类别的 started/ended/passed、配额、窗口数量和推进次数；其中 full/retention 的 passed 仅表示未触发原终止。`transition_plan.json` 保存本次计划。完整 checkpoint 保存窗口起点、累计尝试/成功、推进次数与统计；双卡汇总这些数据，保存时纳入同步哈希。临时仿真 episode 不保存，与原训练的仿真/replay 恢复方式一致。

## 初始化与恢复

初次切换课程必须使用 `--init` 和独立日志目录，从 `ckpt_110500.pt` 初始化；加载网络、优化器与旧后段课程，学习率沿用原进度，新实验迭代编号从零开始。正式双卡 worker、512 环境、256/256 batch 等原执行配置应保留。当前只准备课程，没有执行该训练。

在既有 Stage1 命令中，把旧 `--resume` 替换为：

```text
--init logs/tienkung3_29dof/stage1_continue_80k_20261111/train/ckpt_110500.pt
--transition_curriculum configs/stage1_climb_transition_20261111.json
--log_dir logs/tienkung3_29dof/stage1_climb_transition_experiment
```

显式设置有界 `--iterations`，不要沿用旧命令中的 `-1` 连续执行。后续恢复新实验时使用 `--resume` 并指定相同计划；计划身份不一致、启用/禁用课程变化或缺少数据覆盖都会拒绝继续。更新事件计划应成为新的显式 `--init` 实验。

重新生成计划的命令（输出已存在时拒绝覆盖）：

```bash
python -m tools.build_transition_curriculum \
  --evaluation logs/tienkung3_29dof/stage1_continue_80k_20261111/eval_100000/strict_eval.json \
  --motion-dir data/tienkung3_29dof/stage1/php_all_variants_20261110/motions \
  --contacts-dir data/tienkung3_29dof/stage1/php_all_variants_20261110/contacts \
  --out configs/stage1_climb_transition_20261111.json
```

## 验证

CPU 回归入口为 `python -m unittest tools.test_transition_curriculum`，覆盖配额与片段均匀性、失败不回退、稳定门槛、提前失败、旧难度 episode、状态恢复、事件选择和双进程空结束批次同步。

`tools.check_transition_curriculum` 使用真实 checkpoint 与全量 MotionLib，执行双 GPU 有界仿真，只推理、无优化器更新。逐步核验原 reward 和失败 term 不变、本地窗口 timeout 正确、完整 episode 上限、双卡状态一致及保存状态恢复。执行结果见 [验证回执](evidence/transition_curriculum_check_20261111.json)。

验证通过仅代表代码和仿真接线正确，尚无该课程的训练收益或攀爬成功率结论。

本次验证结果：29 项 CPU 回归通过；双 GPU、共 32 环境、128 个控制步通过接线检查，发生 58 次 episode 结束，其中 20 次到达局部窗口终点且未触发原失败判定。局部质量门槛达标为 0；232 个末尾检查帧中 root/joint/hand/other/slip/all 分别通过 60/190/98/170/13/4。这个小样本说明“存活到窗口末尾”与“稳定跟踪达标”不同，不能据此承诺推进速度；尤其滑移代理的严格程度需要在有界课程实验中观察。当前只实现并验证第一步，没有改动门槛来制造通过结果。
