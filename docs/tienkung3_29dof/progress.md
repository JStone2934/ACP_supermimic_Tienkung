# 实施进度

日期：2026-11-07。**本次 M0–M3 接入与数值验证、合成站立链路检查已通过。** 这不代表 M4–M7 训练或稳定站立能力通过。

| 阶段 | 状态 | 验收证据 |
|---|---|---|
| M0 基线与 Git | 通过 | baseline.json、g1_baseline.json；原改动独立保存，296 clips / 181236 frames、16 GPU env、compile/backward 通过 |
| M1 模型资产 | 通过 | assets.json、robot.json；裸模型 nq/nv/nu=36/35/0，组装后 nu=29；质量 65.817756 kg，与原模型差约 0.000047 kg；足底对齐高度 0.991816 m |
| M2 通用接口 | 通过 | 实例级 RobotSpec、名字映射、动态维度、数据/权重身份与 CLI；15 项回归测试通过；G1 旧权重回放通过 |
| M3 FK 与接触 | 通过 | 天工/G1 各 1000 姿态；非零 anchor、joint ref、多级固定链；九类接触 900 样本；有限差分梯度通过 |
| 合成站立链路 | 通过 | 601 帧、50 Hz 合成参考和标签；1/16/64 env 各 500 步；三网络/reward backward、编译、PD 检查；3 次小更新后保存、恢复到第 4 次、评估及录像均跑通 |

实现提交为 `ca74a80`。完整命令、源码/报告哈希、版本、种子和交付边界见 [evidence/run_manifest.json](evidence/run_manifest.json)。报告均来自本机实际运行；初始失败报告单独保留，不作为最终通过证据。

## 核心数值结果

| 检查 | 实测 | 门槛 |
|---|---|---|
| 天工 FK 位置最大误差 | 9.63e-7 m | 1e-4 m |
| 天工 FK 旋转最大误差 | 4.21e-7 rad | 1e-4 rad |
| 线/角速度最大误差 | 2.07e-6 m/s / 2.97e-6 rad/s | 各 1e-3 |
| 九类接触距离总体 P95 / 最大误差 | 0.0409 mm / 1.324 mm | 各类 P95 ≤5 mm、最大 ≤15 mm |
| 接触梯度有限差分 | 72 个方向，最大绝对误差 9.56e-8 | rtol=1e-2、atol=1e-3 |
| PD 与 MuJoCo 执行器独立对照 | 最大误差 2.84e-14 | 1e-8 |
| G1 旧 checkpoint 回放（3 clips × 20 步） | action/state/reward 最大差 4.05e-6 / 2.07e-5 / 2.24e-8 | 各 1e-3 |

每类脚、手等细分结果及最差样本见 [numerics.json](evidence/numerics.json)；G1 结果见 [g1_numerics.json](evidence/g1_numerics.json)、[g1_checkpoint_regression.json](evidence/g1_checkpoint_regression.json) 和 [g1_regression.json](evidence/g1_regression.json)。接触数值参照禁用了表面采样回退，使用原生 MuJoCo 求距及退化情况的有界重查，避免以同一近似验证自身。

## 合成检查与能力边界

| 并行环境数 | 每环境控制步 | 跌倒总数 | reset 总数 | 数值/编译/反向传播 |
|---|---|---|---|---|
| 1 | 500 | 6 | 7 | 通过 |
| 16 | 500 | 90 | 100 | 通过 |
| 64 | 500 | 371 | 409 | 通过 |

这些是零动作、带环境随机化的接口检查，不是站立控制验收。合成参考为常值姿态，只有双脚接触标签；没有真实动作泛化依据。

小规模 Stage 1 实际完成 3 次更新并保存 checkpoint，重新启动后恢复至第 4 次，参数与指标均有限，身份和数据哈希一致，见 [training_smoke.json](evidence/training_smoke.json)、[training_log.csv](evidence/training_log.csv)。随后真实评估 **1/1 片段失败，约 1.40 s 触发终止**，归一化回报 0.097，见 [synthetic_evaluation.json](evidence/synthetic_evaluation.json)。这证明评估与失败统计链路正常，不能作为站立能力证据。70 帧录像验证见 [render_smoke.json](evidence/render_smoke.json)，本地视频为 `results/tienkung3_tiny_policy.mp4`，未加入 Git。

## 实施中发现并修正

- 原调用链依赖 G1 和固定 296 个片段，已参数化；两个 29 维机器人的数据仍按身份和关节名校验，不能混用。
- 源 STL 超过 MuJoCo 200,000 面限制，转换为保留所有三角面的 OBJ；原始资产保留。
- 冻结上游归档没有 LICENSE/COPYING，manifest 如实记录；网格未提交仓库。
- 601 帧片段暴露末尾单帧 reset bin 越界；bin 只覆盖有后继帧的 transition，已结束评估环境的 frame 也作钳制。
- 初版表面 SDF 在箱体边角穿入时不达标，保留 numerics_initial_surface_only.json。v3 加入凸包面、箱体轴和采样棱边叉积方向，未放宽阈值。
- MuJoCo 原生求距存在零值与 witness 分离的退化情况；离线查询使用有界微位移重查并恢复状态，不能一致时报错。独立数值参照不使用表面采样回退，细节见 design.md。
- `--resume` 要求同数据身份；`--init` 是显式微调入口，可接受同机器人/同模型的新数据，以支持原流水线的 refined dataset。
- origin 使用用户指定仓库；HTTPS fetch、现有 SSH 凭据 push，保留 upstream，未强推或覆盖 main。

## 后续入口（未纳入本次交付）

真实天工动作到位后进入 M4：重定向/转换、数据质检、动作划分、同模型标注。随后才能开展正式 Stage 1–3 训练和各自能力验收。当前 Stage 2/3 与目标任务入口已传递机器人身份，但没有运行其正式训练；实机通信、硬件接线和部署也未实施。
