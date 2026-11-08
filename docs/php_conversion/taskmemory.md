# PHP 转换任务记忆

首次整理及更新：2026-11-08（Asia/Shanghai）。返回[文档入口](README.md)。本文件用于交接与追加记录；历史原始报告仍以其实际日期和代码版本为准。

## 当前方向与停止状态

当前方向：先补齐通用转换工具链代码，以小样本验证求解、质检和任务管理，再考虑全量转换。该方向的待开发事项见 [development.md](development.md)，不表示相关代码已经完成。

2026-11-07 用户要求停止 PHP 试验并收尾。记录中 `automatic_resume=false`；本次 2026-11-08 授权仅落实文档建设，没有恢复求解、训练或批量任务。将来开始新工作时，以届时用户要求确定范围，在新运行目录保存结果，不覆盖本轮证据。

恢复阅读顺序：[本页当前状态](#当前交接状态) → [资产清单](asset_list.md) → [使用手册](usage.md) → [开发路线](development.md)。工作分支为 `feat/tienkung3-29dof-acp`；本轮文档编辑前 HEAD 为 `72f098f`，它不是随后文档变更的提交号。

## 历史开发与转换记录

| 日期 | 阶段/代码版本 | 已执行工作及结果 | 证据 |
|---|---|---|---|
| 2026-11-07 | M0–M3；原工作区基线 `98f38d2`，实现 `ca74a80` | 冻结天工资产、固定头部为 29 DoF，统一机器人接口；FK/接触数值验收、G1 回归和合成站立链路通过 | [接入进度](../tienkung3_29dof/progress.md)、[运行清单](../tienkung3_29dof/evidence/run_manifest.json) |
| 2026-11-07 | 合成站立小训练 | 更新 3 轮并保存，恢复到 4 轮；评估约 1.40 s 失败，仅证明训练/恢复/评估入口可用 | [训练检查](../tienkung3_29dof/evidence/training_smoke.json)、[评估](../tienkung3_29dof/evidence/synthetic_evaluation.json) |
| 2026-11-07 | PHP pilot 起点 `36219b6` | 选取站立、直行、左右转向、低台阶共 7 片段，保持原地形第 0 变体及来源哈希 | [试验设计](../tienkung3_29dof/php_pilot/design.md)、[来源清单](../tienkung3_29dof/php_pilot/evidence/selection_and_retargeting.json) |
| 2026-11-07 | pilot v1 → v2，代码 `c79c31e` | 首轮出现摆臂碰髋、脚部穿地和速度尖峰；改用平地中立手臂、台阶外展手臂，强化脚部穿透和速度惩罚 | [v1 报告](../tienkung3_29dof/php_pilot/evidence/quality_v1.json)、[剩余片段初版](../tienkung3_29dof/php_pilot/evidence/quality_v1_remaining.json)、[最终来源与配置](../tienkung3_29dof/php_pilot/evidence/selection_and_retargeting.json) |
| 2026-11-07 | 最终数据 `php-pilot-taskspace-v2` | 7 片段、1220 帧、50 Hz；初筛 7/7、严格几何 0/7、PD 跟踪 0/7；生成标签与参考录像 | [质量报告](../tienkung3_29dof/php_pilot/evidence/quality_final.json)、[标签统计](../tienkung3_29dof/php_pilot/evidence/final_labels.json) |
| 2026-11-07 | 站立/平地探索性训练 | 站立日志至 250 轮，checkpoint 为 200 轮；平地日志至 100 轮后 OOM、无 checkpoint；均无本轮训练后评估 | [站立日志](../tienkung3_29dof/php_pilot/evidence/standing_train_log.csv)、[平地错误日志](../tienkung3_29dof/php_pilot/evidence/flat_training.log) |
| 2026-11-07 | 用户停止；收尾提交 `72f098f` | 停止实验、保留失败及部分训练证据；历史 20 项回归通过，未继续补训或补做策略录像 | [收尾说明](../tienkung3_29dof/php_pilot/README.md)、[运行清单](../tienkung3_29dof/php_pilot/evidence/run_manifest.json)、[tests.txt](../tienkung3_29dof/php_pilot/evidence/tests.txt) |
| 2026-11-08 | 本轮文档建设 | 新建 PHP 使用/开发/交接/资产文档，链接现有证据；核对源码参数、本地文件与历史哈希 | [文档入口](README.md)、[资产清单](asset_list.md)；本轮未执行转换或训练 |

## 当前交接状态

### 数据与质量

- 最终动作根目录为 `data/tienkung3_29dof/motions/php_pilot_final/`，标签位于同级 `php_pilot_final_contacts/`。7 个片段详情见[资产表](asset_list.md#七个最终片段)。
- 默认碰撞模型存在两对小腿—脚部约 34 mm 重叠，轨迹中的原始最大值约 36 mm。初筛例外只用于诊断与明确标记的探索性训练，没有通过 M4 严格验收。
- 最终数据最大地形穿透约 1.85 mm，各片段支撑距离 P95 不超过 1.32 mm，支撑脚水平速度 P95 不超过 0.014 m/s；没有关节硬限位或参考速度上限超出。指标来自最终质量报告，不是新一轮测量。
- 当前仅使用第 0 地形变体，手臂固定；不能推广为所有变体或手支撑攀爬可行。全部样本为开发集，来源家族不足以建立独立 80/10/10 泛化评估。
- 原始 G1 在相同自由根部 PD 协议下也失败，见[源 PD 报告](../tienkung3_29dof/php_pilot/evidence/source_g1_pd.json)。PD 与学习控制器结论必须分开。

### 训练与录像

| 项目 | 持久化状态 | 恢复前需知道 |
|---|---|---|
| 单站立 `php_pilot_stand_v2` | `logs/tienkung3_29dof/php_pilot_stand_v2/ckpt_000200.pt` 与 `last.pt` 均为 200 轮 | 日志到 250 不等于保存到 250；原计划 800 轮未完成，200 轮 checkpoint 未评估 |
| 五片段平地 `php_pilot_flat_v2` | 日志和 args 存在，没有 checkpoint | 原计划 1000 轮；日志到 100 后在 Warp CUDA graph launch OOM，未诊断或重试 |
| `standing_prior_tiny_policy.json` | 先前 4 轮合成训练权重在新站立片段上的失败检查 | 不是本轮 200 轮 checkpoint 的评估 |
| `results/php_pilot_final/reference.mp4` | 7 片段运动学参考录像 | 不含策略控制成功含义；播放时长包含渲染尾帧，不能用视频时长代替 24.26 s 数据有效时长 |
| `results/php_pilot_final/<类别>/<片段>_pd.npz` | 7 份 PD 轨迹 | 本轮没有生成对应 PD MP4，也没有训练后策略录像 |

训练参数与门槛证据分别为[站立参数](../tienkung3_29dof/php_pilot/evidence/standing_args.json)、[平地参数](../tienkung3_29dof/php_pilot/evidence/flat_args.json)、[站立训练准入记录](../tienkung3_29dof/php_pilot/evidence/standing_training_gate.json)、[平地训练准入记录](../tienkung3_29dof/php_pilot/evidence/flat_training_gate.json)。

### 尚未完成

转换工具链：通用任务输入、碰撞模型处理、全身接触优化、长轨迹/重试、多变体验收、正式划分、可恢复多卡调度。控制能力：训练显存问题定位、学习策略跟踪评估及策略录像。复杂类别和全部 296 条转换尚未开始，M4/M5 能力验收未通过。

本机双卡 RTX PRO 6000 的存在仅是资源信息；双卡 5090 是讨论中的可选节点，没有已部署的跨机器调度。不要从规划文字推断有后台任务。

## 本轮文档核对

2026-11-08 核对了新文档的本地链接与锚点、Bash 示例语法，并通过源码中的 argparse 声明静态核对 Python 模块参数。实际文件检查确认 296 条源动作及 296 份源标签在本地，最终 7 对 motion/目标标签身份一致，源 motion/源标签及最终 motion/terrain/intent 哈希与清单一致。

历史运行清单记录的源码、报告、参考 MP4 和 `last.pt` 哈希均一致。两个站立 checkpoint 通过 CPU 只读加载核对了轮次和数据身份，均为 200 轮；编号文件有独立文件哈希，详见资产清单。此次没有执行数值回归、转换、训练或策略评估，历史 20 项测试结果仍归属于 2026-11-07。

## 交接与更新规则

1. 先核对当前分支、工作区和本页最后记录，再查看对应 manifest、模型身份和输出是否仍在本地。历史状态不能代替实时检查。
2. 按用户当次要求确定开发或实验范围。文档示例是操作参考，不是定时任务或自动恢复安排。
3. 每次转换使用新运行目录；记录代码提交及未提交差异、源数据/模型哈希、完整命令和随机种子。继续训练时另外核对 checkpoint 数据身份。
4. 将“运行完成”“几何初筛通过”“严格几何通过”“PD 达标”“策略跟踪通过”分别填写，未运行写未运行。
5. 小型命令、报告和 manifest 可按项目规则进入 Git；大数据、模型、权重和录像保留本地并记录重建/同步方式。历史 evidence 原样保留。

## 追加记录模板

复制以下模板，为新任务填写一条记录；不要把空模板当成已执行事件。

```text
日期/时区：YYYY-MM-DD，Asia/Shanghai
任务标识/run_id：
任务范围与用户要求：文档 / 开发 / 小样本转换 / 全量转换 / 训练评估
代码版本：commit；是否有未提交差异及其保存位置
目标与验收条件：
输入：源文件/帧区间/地形变体；来源 manifest/哈希
模型：robot_id、model_hash、config_hash、joint order、几何规则版本
环境与资源：环境文件/实际版本、设备、并行数
命令与配置：完整命令、配置文件、种子、迭代/时间伸缩参数
输出：motion/terrain/intent/labels/report/trajectory/checkpoint 路径
结果：发现/求解/导出数量、帧数、时长；各验收状态分别列出
失败原因：失败任务、约束/异常、日志、重试次数与是否隔离
证据：manifest、质量报告、哈希、录像；注明录像类型
停止/恢复状态：已完成 / 中断 / 失败；可恢复位置；是否仍有进程
下一步：具体开发项或验证项，前置依赖
资产清单更新：新增、替代或缺失项；保留的历史版本
```

## 2026-11-08 全量转换启动（当前任务）

用户已明确授权完成代码接线、全量数据转换、双卡效率优化及文档/Git 管理；历史停止状态只属于 11 月 7 日实验。本次不恢复历史训练。

- 工作区：`/home/js/robot/locomotion/ACP_supermimic_Tienkung`；分支 `feat/dual-5090-scheduling`。导入文档基线提交 `c0a51b8`。
- 实际设备：2 × RTX 5090（各 32607 MiB）；解释器 `/home/js/miniconda3/envs/acp_supermimic/bin/python`。此前文档中的 PRO 6000/miniforge 路径属于另一台机器。
- 输入：296 条源动作、181236 帧；2960 个地形槽位。平地变体几何相同，可以计算一次并显式保存别名；其余变体必须分别质检。全量源路径与 SHA-256 写入 `evidence/source_inventory.json`。
- 最小实现方向：保留 pilot；复用 RobotSpec/FK/地形与独立 MuJoCo 几何，新增任意源动作/变体输入、九类接触意图、分窗优化、每任务原子状态与恢复、独立 CPU 验收/标签、合格子集及来源家族划分。
- 调度：GPU 轨迹求解与 CPU 验收/标签分离，先测代表动作耗时/显存，再确定每卡 worker 与帧窗口。离线转换不创建训练仿真环境，不用训练 `num_envs` 充当转换并行参数。
- 完成标准：全部输入与变体都有可追溯终态、候选输出及成功/拒绝/失败原因；合格子集与标签对应；不通过放宽旧质量阈值制造通过，不将运动学结果当成策略能力。
- 当前状态：资产盘点已开始；代码与全量运行尚未完成。历史模型与输出保持原样。
