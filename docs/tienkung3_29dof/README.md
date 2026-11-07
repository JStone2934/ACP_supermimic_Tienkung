# 天工 3.0 29 DoF 接入：复现入口

本次范围为 **M0–M3 + 合成站立链路检查**。完整迁移计划保留在 [migration_plan.md](migration_plan.md)，当前状态见 [progress.md](progress.md)。设计与持续任务上下文分别见 [design.md](design.md)、[taskmemory.md](taskmemory.md)。

## 环境与资产

```bash
conda activate acp_supermimic
python -m tools.prepare_tienkung3
python -m tools.generate_standing_motion --robot tienkung3_29dof
python -m supermimic.acp.make_labels \
  --robot tienkung3_29dof \
  --out data/tienkung3_29dof/contacts/bootstrap --workers 1
```

环境锁定见仓库根目录 `environment.yml`。本机解释器为 `/home/js/miniforge3/envs/acp_supermimic/bin/python`。转换器下载固定上游提交，保留原 URDF/网格，在 generated 生成无损 OBJ 与固定头部的派生 URDF/MJCF；不将模型网格或训练文件加入 Git。上游归档未包含 LICENSE/COPYING，来源记录保留在 manifest，未替上游添加许可。

模型规格：29 个受控 hinge + 1 个 freejoint，裸模型 `nq=36, nv=35, nu=0`，mjlab 组装后 `nu=29`。根为 pelvis，躯干为 waist_pitch_link。策略 50 Hz、物理 200 Hz；动作目标为 `q_default + joint_offset + 0.25 * action`。速度限值没有物理硬约束。armature=0 是未辨识的仿真假设。

## 验证命令

```bash
python -m tools.check_robot_numerics \
  --out docs/tienkung3_29dof/evidence/numerics.json
python -m unittest tools.test_robot_contracts tools.test_acp_pipeline

for n in 1 16 64; do
  python -m tools.check_acp_setup --robot tienkung3_29dof \
    --contacts-dir data/tienkung3_29dof/contacts/bootstrap \
    --num-envs "$n" --steps 500 \
    --out "docs/tienkung3_29dof/evidence/smoke_${n}.json"
done

python -m tools.check_acp_setup --robot g1 --num-envs 16 --steps 16 \
  --out docs/tienkung3_29dof/evidence/g1_regression.json
python -m tools.preview_robot \
  --out docs/tienkung3_29dof/evidence/default_pose.png
```

`check_robot_numerics` 默认检查 1,000 个限位内随机姿态的位置、旋转、线/角速度，另用带非零 anchor、joint ref 和多级固定链的小模型检查；九类接触各 100 个样本覆盖平面、箱体顶面、侧面、边角。接触和 FK 均执行有限差分梯度检查。误差门槛见 design.md，失败会以非零退出码返回。

`check_acp_setup` 检查真实 reset/step、观测、状态、command、两种奖励、执行器、replay，以及 policy → world model → reward/critic 的反向传播；world model step 经过 torch.compile。另用 CPU MuJoCo 的执行器输出独立检查 PD 与饱和公式。

## CLI 与身份约定

- `--robot` 已接入训练、标注、动作清理、评估、渲染、检查和流水线；旧调用默认 G1，评估/渲染由文件元信息解析身份。
- 关节按名字双向重排，不能把两个机器人的 29 维数组视为兼容。
- 新文件/checkpoint 带 robot_id、model/config hash、canonical joint order、control_dt；checkpoint 记录数据哈希，标签绑定 motion/terrain 文件哈希。旧无身份文件只允许走 G1 路径。
- `MotionLib` 支持 `(T,36)` motion 和 `(0,1,10)` 空平地 terrain；合成站立为 601 帧、50 Hz。末帧不能作为有后继动作的 reset 起点。
- 运行目录绑定模型、配置、输入文件；更换输入不能复用旧流水线完成标记。正式流水线仍为 Stage 1 → offline planner → Stage 2；本次未新增或运行完整 Stage 3 流水线。

## 交付边界

合成站立文件是常值参考姿态，**不是已验证可执行的稳定站立控制器**。零动作检查中的跌倒和 reset 会如实报告；通过表示接口和数值链路正常，不表示站立/步行/攀爬能力通过。真实动作质量、任务尺度阈值、重定向、Stage 1–3 正式训练和实机部署均留待后续。

接触近似通过目标模型的样本验收，也不等于任意深度穿入、任意机器人和地形上的精确凸体求距。仿真维持无自碰撞策略；后续动作质检和实机工作仍需独立检查非相邻 link 自碰撞与可承载接触区域。

## Git 管理

工作分支 `feat/tienkung3-29dof-acp`。`origin` 为用户指定的 `JStone2934/ACP_supermimic_Tienkung`（HTTPS fetch，SSH push）；原仓库保留为 `upstream`。既有工作区改动已单独保存为 `98f38d2`，原始文件哈希见 evidence/baseline.json。没有强推或直接修改远程 main。

## 默认姿态

![天工默认姿态](evidence/default_pose.png)

## 已验证结果与短训练复现

[进度与验收表](progress.md) 和 [完整运行清单](evidence/run_manifest.json) 记录实际结果。M0–M3 数值验收、15 项回归测试、1/16/64 环境检查通过。合成参考不依赖真实动作库，因此可以先完成软件接入。

运行清单中保存了小规模训练、恢复、评估和录像的完整命令。训练只做 3 次更新后恢复至第 4 次，不是正式训练；评估在约 1.40 s 失败，未达到稳定站立能力。使用 `--resume` 时数据哈希必须一致；显式 `--init` 可在同模型的新数据上微调。

视频复现须先激活 conda 环境，确保 ffmpeg 位于 PATH。实际评估轨迹和视频保留于本地 `results/tienkung3_tiny_policy.npz` / `results/tienkung3_tiny_policy.mp4`，证据目录记录哈希与编码检查结果。模型/动作/标签/checkpoint/视频不随代码推送；新机器应按本页命令重建资产和合成数据。
