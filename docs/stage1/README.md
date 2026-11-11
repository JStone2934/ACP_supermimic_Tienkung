# 天工 Stage1 训练

**当前状态（2026-11-10 20:46）：四组超参对照完成，原配置已从第 3000 轮恢复正式后台训练。** 更快的两个候选出现根部跟踪退化；加倍 batch 提高样本吞吐，但没有显示足够的迭代和短训质量收益，本次保留原超参。详见 [实验结果](hyperparameter_tuning.md) 和 [实时进度](progress.md)。

恢复后正式实测 **0.431 秒/轮**；20:49 时进度 3225/60000，预计剩余约 **6.8 小时**，约北京时间 **11月11日 03:36** 完成迭代预算，评估另计。后续以实时进度为准。

Stage1 使用天工 `tienkung3_29dof`，训练输入覆盖 296 条原动作、2960 个原始地形组合、1812360 个展开帧。当前正式配置为 224 球奖励、整批奖励计算、双卡同步训练；累计预算 60000 轮，结束后仍需完整动作仿真评估。

正式服务：`acp-stage1-php-all-20261110.service`。运行目录：`logs/tienkung3_29dof/stage1_php_all_spheres224_20261110/`。后台作业独立于终端连接，测试与正式训练顺序运行，资源保护保持 64/80 GiB。

| 文档 | 内容 |
|---|---|
| [plan.md](plan.md) | 数据、配置、检查点和评估计划 |
| [taskmemory.md](taskmemory.md) | 授权、决定、交接与故障记录 |
| [asset_list.md](asset_list.md) | 数据、代码、日志、检查点及证据路径 |
| [usage.md](usage.md) | 启动、状态、停止与恢复 |
| [development.md](development.md) | 数据装配、后台控制器和验证依据 |
| [progress.md](progress.md) | 当前进度、速度和预计时间 |
| [hyperparameter_tuning.md](hyperparameter_tuning.md) | 环境数、batch、更新次数与实际跟踪质量对照 |

历史优化记录：[224 球采用](spheres224_migration.md)、[奖励执行优化](execution_tuning.md)。旧密集几何约 43.53 秒/轮；224 球加分块约 1.30 秒/轮；整批计算正式窗口约 0.421 秒/轮。它们是不同阶段的测量，最新 ETA 以实时进度为准，均不是收敛承诺。

接触采样诊断见 [contact_sampling.md](contact_sampling.md)；box、胶囊、凸包、SDF 检索及 G1/天工实模图见 [contact_geometry_options.md](contact_geometry_options.md)。

转换验收说明参考动作满足运动学和几何标准，策略的实际跟踪能力仍需独立检查。当前全量用于训练，诊断集合来自训练数据，不声称未见动作泛化。短训完成和 60000 轮预算完成均不自动代表策略通过验收。

Stage2 代码衔接审查与修正见 [stage2_alignment.md](stage2_alignment.md)：CPU 检查通过，Stage2 尚未启动，当前 Stage1 继续运行。
