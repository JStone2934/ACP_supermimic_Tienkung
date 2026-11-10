# PHP 转换任务记忆

首次整理：2026-11-08；最新更新：2026-11-09（Asia/Shanghai）。返回[文档入口](README.md)。本文件用于交接与追加记录；历史原始报告仍以其实际日期和代码版本为准。

## 当前任务：2026-11-09 全量交付

用户最新授权：先解决 1 条程序异常，统计全部拒绝原因，针对主要问题改进算法、小批验证后重跑。v2 全量已结束（111 accepted、184 rejected、1 failed）；原失败候选已修复验收异常并在独立目录完成复验，属于质量拒绝。v3 两轮小批和长轨迹验证通过；攀爬试验未改善整体准入，不推广。15:48 启动、约 16:30 结束 `php_repair_v3_20261109`：146 条行走/台阶中 133 accepted、13 rejected、0 failed，合格变体 210→1093，新增 904、退化 21。全库优先保留旧通过项的索引已核验，共 205 条动作、1814 个变体；原有 910 个全部保留。运行使用 2 × 5090，每卡 2 worker，6 CPU worker。源码快照在新批次 `source_snapshot/`，命令/PID 在 `launch.json`，身份在 `plan.json`，实时状态在 `manifest.json`，日志在 `.cache/php_conversion/repair_v3/rerun.log`。旧批次不可覆盖，训练未恢复。

## 历史方向与停止状态（2026-11-08 文档整理时）

当前方向：先补齐通用转换工具链代码，以小样本验证求解、质检和任务管理，再考虑全量转换。该方向的待开发事项见 [development.md](development.md)，不表示相关代码已经完成。

2026-11-07 用户要求停止 PHP 试验并收尾。记录中 `automatic_resume=false`；本次 2026-11-08 授权仅落实文档建设，没有恢复求解、训练或批量任务。将来开始新工作时，以届时用户要求确定范围，在新运行目录保存结果，不覆盖本轮证据。

恢复阅读顺序：[历史 pilot 状态](#历史交接状态2026-11-07-pilot) → [资产清单](asset_list.md) → [使用手册](usage.md) → [开发路线](development.md)。工作分支为 `feat/tienkung3-29dof-acp`；本轮文档编辑前 HEAD 为 `72f098f`，它不是随后文档变更的提交号。

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

## 历史交接状态（2026-11-07 pilot）

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

## 2026-11-09 v3 修复收尾

- 数值修复：近接触零距离参加有界原生重试的一致性检查；原失败候选不改变轨迹完成 10 个变体验收、标签及哈希验证，状态为质量拒绝。
- 统计：全 296 条、2960 个变体的拒绝原因在 `evidence/full_v2_rejection_summary.json`，原因计数有重叠。
- 验证：43 项回归通过；12 条小批 30→96 合格变体；6 条长轨迹 25→52；两轮独立审计均通过，原合格变体无损失。攀爬试验有退化，未推广到攀爬。
- 重跑：146 条行走/台阶全部完成；自碰撞拒绝 990→10，地形穿透拒绝 651→355；原批次与试验批次均保留。详见 `evidence/repair_v3_rerun_comparison.json`。
- 组合：`tools.index_php_conversion_variants` 按原始 source/variant 优先保留 v2，新增 v3 合格变体；本地 `accepted_variant_index.json` 经核验含 910 个原版和 904 个新增。该索引不直接适配 MotionLib，读取者需使用条目中的父批次路径及导出槽位。
- 未完成能力：攀爬的多点接触、剩余地形变体穿透、21 个新版退化变体的算法修复、正式独立划分及学习策略跟踪。此次没有训练或降低验收门槛。

最终独立审计通过：146 条、91668 帧、1460 个变体、1262 个输出哈希，见 `evidence/repair_v3_rerun_audit.json`。转换进程已退出，无后台转换任务；日志、候选、拒绝报告和源码快照全部保留。下一步需另定攀爬/地形优化范围，不能把本次完成解释为全库全通过。

## 2026-11-09 同步预览的手臂与体型诊断

用户指出 G1/天工体型差异下动作还原不理想、天工手臂未灵活跟踪。已核验四条预览数据及准备目标：目标双臂 14 关节全程恒定，源双臂均有运动；平地选择 `neutral`、台阶选择 `clearance`，求解器每步将臂关节重置为常量。默认根部到足底标定比为 1.31133，当前身体目标未按各肢段单独适配。独立检查器也没有自由摆臂保真门槛。

证据为 `evidence/preview_arm_diagnosis_v3.json`，原视频清单及输入哈希保持一致。已在 README、development 和资产说明中明确几何合格不代表动作完整还原。本轮仅诊断和记录，没有修改求解器、覆盖数据或启动新批次。后续应优先恢复自由摆臂、改进肢段映射并补齐保真指标，在四条完整样本上生成新对照后再推广。

## 2026-11-09 v4 全地形目标推进（未完成）

用户明确要求自由摆臂、按肢段重建目标、手/肘轨迹/摆幅/时序验收，四条完整样本验证后推广，所有地形变体通过。活动目标保持 296 条/2960 原始槽位，不以四条样本或任一子集替代完成。

- 已实现 `php-segment-v4`，新增 `retarget_fidelity.py`；独立几何检查外增加原生 FK 的自由摆臂检查。恢复摆臂并按各肢段长度构造目标，保留原地形与帧时序。52 项回归通过。
- 四条完整样本 `php_segment_v4_preview_a_20261109` 已完成，40/40 同时通过两类检查，独立审计核验 36 个输出哈希。新版视频已生成并检查完整帧数，见 `results/php_segment_v4_previews_20261109/comparison/all_comparisons.mp4`。
- 全库首轮 17:13 启动，控制器 PID 2227018，exec session 10078，目录 `php_segment_v4_full_20261109`；2 GPU × 2 workers，6 CPU 验收。实际进程状态必须重新核实，不能仅据本记录认为仍活跃。
- 首个长动作 cfg0023 通过 8/10；变体 1、6 分别有约 20.5/33.6 mm 脚部穿透。已启动两条完整 1243 帧、各自指定 solve_variant 的修正探针，控制器 PID 2242828，exec session 39581，目录 `php_segment_v4_variant_probe_a_20261109`。日志 `variant_probe_a.log`。
- 两个运行均从 `.cache/php_conversion/segment_v4/frozen_runner` 执行。44 份求解/检查源码与小批 plan 哈希逐一一致；工作区后续工具改动不会污染正在运行的进程。新算法版本需要另建冻结执行目录。
- 已补充全覆盖核验和逐变体重试计划工具；资产索引按原始 source 身份映射不同输出名，并拒绝混入旧版没有保真检查的数据。
- 后续：核实两个进程并收集逐任务结果；先验证探针能否补齐对应原始地形；按全库主要拒绝原因改进与再次小批验证；对缺失地形逐一重定向。最后以 `audit_php_variant_coverage --require-all` 及实际质量报告证明全 2960 槽位，不得因进程结束而宣布目标完成。尚未训练策略。

### 17:39 更新（Asia/Shanghai）

- v4 变体探针已完成：两条完整派生轨迹各 9/10，均通过指定变体；独立审计通过，原始槽位索引合计 1 条源动作的 10/10，不能重复计成 18 个。`segment_v4_variant_probe_coverage.json` 明确其余全库仍缺 2950 槽位（该核验只输入探针）。
- v4 全库控制器 PID 2227018/session 10078 仍活跃；17:37 实际任务状态 36 accepted、9 rejected、4 solving、3 checking、244 pending；450 个已检查槽位中 326 个通过，20 条已全变体通过。进度需继续从磁盘/进程重读，不要固定引用这个时间点。
- 源膝擦碰诊断已完成，`segment_v4_source_contact_diagnosis.json` 包含全 296 条第 0 变体的材料点速度和持续时间。cfg0062 的右膝短接触实际滑动约0.55m/s，中间有8.4mm源穿透。不要把这种近接触直接当静止支撑。
- 工作区现在为 `php-segment-v5`：新增 `contact_intent.py`，仅识别不足0.3s且材料切向速度中位数>0.35m/s的膝/肘擦碰，保留原始标签、手脚支撑和全身碰撞门槛；记录 incidental_segments/policy。27 项相关回归通过，完整批次效果尚待验证。
- v5 14 条小批于17:38启动：PID 2297786，exec session 48847；目录 `php_segment_v5_contact_pilot_20261109`，冻结执行目录 `.cache/php_conversion/segment_v5/frozen_runner`，45份源码已对 plan 哈希核验。2GPU各1worker/3CPU，日志 `.cache/php_conversion/segment_v5/pilot.log`。与 v4 全库并行，均不应重启；先核实句柄和任务文件。
- 下一步：收齐 v5 对照结果，确认四条预览不退化、长动作假支撑得到修复，并单独评估攀爬；手部长期滑动仍未解决，不能用删除手支撑掩盖。全库运行结束后生成覆盖/缺失 selections；保留完整帧，逐地形改进和复验。活动目标未完成，仍要求全2960原始槽位通过。

17:43 实时核验：两个控制器 PID 2227018/2297786 均存活；v4 全库已检查56条/560槽位，其中326槽位通过，36条至少一个变体通过、20条全变体通过、20条全拒绝，尚余236条未开始、4条求解中。v5小批前两条已经通过，其中cfg0062从v4的0/10变为10/10；其余尚未完成，不能据此推广。v5完整54项回归通过，日志 `evidence/segment_v5_all_tests.txt`。实时状态脚本 `.cache/php_conversion/segment_v4/status.py` 会读取三个运行的任务文件及质量报告，并给出基于已完成样本的GPU时间估算；每次需配合进程状态确认。

### 18:23 更新：接触试验结果与用户新增前倾问题

- v5 小批结束：6 accepted / 8 rejected、58/140；四条预览 40/40 保留，cfg0062 新增 10、cfg0029 新增 8。独立审计 `segment_v5_contact_pilot_report.json` 已通过。
- v6 解剖接触点映射小批结束：6 accepted / 3 rejected、60/90，四条预览保留且 cfg0082/cfg0076 各新增 10；独立审计通过 69 个输出哈希。攀爬接触/穿透仍失败，部分穿透退化，不能整体推广。新增策略区分靠近足踝且已有同面足支撑的小腿远端擦碰，手接触使用解剖基底映射；证据 `segment_v6_source_patch_diagnosis.json`、`segment_v6_patch_pilot_report.json`。
- v7 当前/近接触碰撞面刷新试验失败：7 rejected、0/70，包括四条原通过预览都因自碰撞退化。独立审计 `segment_v7_collision_pilot_audit.json` 通过，只证明报告/资产一致，不证明算法成功。不得推广 v7。
- 工作区求解器当前仍是实验 `php-segment-v8`（最差帧穿透、超速和滑动损失），57 项回归已通过，日志 `segment_v8_tests.txt`；尚未启动 v8 转换，且继承 v7 风险。不要直接用它开展全库重跑。
- v4 全库控制器 PID 2227018 于18:18仍运行：47 accepted、79 rejected、1 failed、4 solving、3 checking、162 pending；1260 已验收槽位中431通过。1条失败为 `motions-high-climb-76/run_right_90_1.0s_step0.8s` 标签距离重试不一致，task `16d9fc4181e8d2b949afe9c0`，候选已生成，待隔离诊断。不要改写活动运行目录。
- 用户新增指出天工比 G1 前倾。新增 `tools.diagnose_php_posture` 使用原生 FK、各自中立姿态校准、源根部水平朝向测量躯干/骨盆前倾，正号向前。原文件哈希与视频清单核验、3项方向/校准测试通过。旧四条首秒 G1约2.95°，v3天工13.56°–15.16°，v4天工10.61°–10.88°；骨盆反而比源少前倾0.7°–1°，主要偏差在腰部/上身。现有躯干局部摆臂检查对整个上身转动不敏感，缺少直接躯干姿态目标。
- 已独立验证修正 `php-segment-v9-posture-probe`：基于冻结 v6，仅增加校准后的源躯干朝向目标及20倍矩阵误差、5倍骨盆朝向误差，保留原模型/地形/帧率/所有既有门槛。四条完整动作1397帧、40/40几何与摆臂通过，独立审计36个输出哈希通过。全段躯干前倾RMSE 0.084°–0.276°，P95 0.158°–0.408°，首秒约2.97°；骨盆动态P95仍3.64°–7.07°、最大13.02°，未声明整体姿态已解决。
- v9 运行目录 `data/tienkung3_29dof/motions/php_posture_v9_pilot_20261109`；启动控制器2447714/session15673，已完成。冻结代码 `.cache/php_conversion/posture_v9/frozen_runner`，45文件哈希与plan逐一核验并复制到运行目录 `source_snapshot`。补丁 `evidence/posture_v9_probe.patch`。本修正**尚未并入工作区 v8 求解器**，不能误认为全量首轮已有修正。
- v9 证据 `posture_v9_pilot_comparison.json`、`posture_v9_pilot_audit.json`；对照视频正在session33271生成，输出 `results/php_posture_v9_previews_20261109/comparison`，需核对结束与帧数后交付。v6→v9单因素对照诊断session72927，输出 `posture_v9_ablation.json`。
- 后续优先：收齐新录像并核验；将躯干/骨盆姿态加入独立验收及索引策略，避免旧“几何+摆臂”通过项冒充整体姿态通过；基于验证成功的 v6+姿态约束继续改进骨盆、攀爬和逐地形适配，不采用 v7 退化路线。活动目标仍是全部2960原始槽位通过，不得标记完成。全库首轮继续作为诊断基线，验收计数未包含新增姿态标准。

18:25 补充：v9 渲染 session33271 已正常结束，四条录像及合辑生成；逐片 ffprobe 帧数/50fps/1920×640、SHA-256 与视频清单校验，合辑1397帧。`posture_v9_preview_manifest.json`、`posture_v9_summary.json`保存交付身份。v6→v9单因素对照完成：前者四条躯干RMSE 4.46°–6.09°，后者0.08°–0.28°；除显式姿态项、准备目标与版本标识外求解器不变，补丁已保留。当前有效修正源码位于v9的冻结目录和source_snapshot，主工作区仍是未经转换验证的v8，下一轮应先整理合并稳定修正再重跑。

### 18:40 更新：正式姿态验收与扩大试验

- 上轮属于实质进展。本轮全库控制器2227018仍活跃，18:37首轮1970个已检查槽位中1007个通过旧几何/摆臂门槛，尚有70 pending/4 solving/24 checking/1 failed；不计入新增躯干门槛下的正式覆盖。
- 已将旧工作区v8源码保存至 `.cache/php_conversion/segment_v8/source_snapshot`，工作区求解器现为 **php-segment-v9**：基于验证稳定的v6+姿态约束，撤回v7碰撞面刷新及未经验证v8峰值项。随后为躯干加入 `.1*max(frame_rotation_error)`，修复probe直行的11°瞬时三维误差。四条probe的新门槛复核见 `posture_v9_probe_gate_recheck.json`；只有直行因最大角度失败，前倾结论不变。
- 新独立姿态策略 `torso-attitude-v1` 已接入checker/export/audit/index/coverage：躯干全三维角误差RMSE≤3°、P95≤5°、max≤10°，骨盆单独报告；旧结果不能与正式新姿态策略混合计数。63项回归通过，`segment_v9_tests.txt`。首次回归发现阈值字典键拼接错误，已修复并完整重跑通过，原失败日志保留在cache，不引用为通过证据。
- 正式v9扩大14条完整动作：`php_segment_v9_pilot_20261109`，PID2513727/session25936，冻结 `.cache/php_conversion/segment_v9/frozen_runner`，45源码与plan核验并复制source_snapshot；日志 `segment_v9/pilot.log`。18:40已3 accepted/1 rejected/2 solving/8 pending；cfg0076、cfg0082各10/10，cfg0029为9/10，cfg0062因自碰撞0/10。剩余控制与攀爬尚未完成，禁止推广。
- 隔离v10峰值损失试验7条（四控制+三攀爬）：`php_segment_v10_peak_pilot_20261109`，PID2539619/session96176；冻结 `.cache/php_conversion/segment_v10/frozen_runner`。新增sample/self/native collision峰值、速度峰值、滑动均值/峰值项，**继续累积碰撞面，不采用v7刷新**。两卡各1worker/3CPU，与v9/首轮并行；18:40为2 solving/5 pending，无完成结果。工作区仍v9，不要混淆。
- 攀爬逐帧原生诊断 `segment_v6_climb_collision_frames.json`：低速正向攀爬仅11帧>3mm，最坏frame370左右腿/脚相交43.8mm、地形41.6mm，支撑手短时离锚点22cm。峰值损失试验由此驱动，未放宽验收。
- v4首轮异常已复现到variant9、左踝pitch几何与box0，原生返回distance=0却witness gap≈0.213m。1e-7及1e-6扰动各只有一条有效距离，分别0.0861167358/0.0861168244m；旧代码每尺度清空候选，虽两条相互一致却失败。工作区 `make_labels.geom_distance` 已改为跨尺度保留有效候选，仍使用原最大1e-5m扰动/原一致性界限。fixture `distance_cross_scale.json`、证据 `segment_v4_distance_retry_diagnosis.json`。两项距离测试及原candidate复验session76942正在运行，日志 `segment_v9/distance_tests.log`/`distance_fixed_recheck.log`。两个活跃小批使用修改前冻结源码，不受影响。
- `.cache/php_conversion/live_coverage.py` 现在分别统计当前三项门槛和旧几何/摆臂基线。18:40正式已通过29/2960槽位、缺2931；旧口径1137不能相加。最终仍须全原始2960槽位的独立覆盖审计。

### 18:53 更新：两批收齐、异常恢复、宽度适配试验

- 正式v9已结束：9 accepted/5 rejected、88/140、10476帧；四条控制40/40全部通过包括躯干姿态。独立审计106哈希通过，`segment_v9_pilot_report.json`。v10已结束：3 accepted/4 rejected、30/70；直行因右脚支撑距离失败，三攀爬仍全拒绝，姿态/接触均明显退化，**不推广峰值损失路线**。审计47哈希通过，`segment_v10_peak_pilot_report.json`。
- `audit_php_variant_coverage v9 v10 --require-all` 按预期返回1：88/2960通过、缺2872，7条全变体通过。证据 `segment_v9_v10_coverage.json`及`.index.json`，绝非目标完成。v10通过项均被v9覆盖，没有新增槽位。
- 四条正式v9同步视频已重新生成，`results/php_segment_v9_previews_20261109/comparison/all_comparisons.mp4`，显示Geometry/Arm fidelity/Torso三个结论；1397帧、50fps、1920×640，逐片和合辑帧数/哈希核验。清单与指标 `segment_v9_preview_manifest.json`、`segment_v9_preview_summary.json`。不要与之前没有姿态gate的posture-v9-probe混同。
- 工作区当前仍正式v9，增加距离跨尺度重试和中断状态记录后，完整65项回归通过 `segment_v9_final_tests.txt`。`tools.convert_php` 现在把捕获到的worker退出记成manifest interrupted，避免把已经退出的控制器继续报告为running；已用模拟worker -9回归验证，已完成任务不伪造终态。导出manifest也声明posture_policy。
- 距离异常修复完成：原v4失败候选不改动运动数据，全部10变体/666帧重验及标签/哈希核验通过，结果是质量拒绝（0/10），不再是程序异常。证据 `segment_v4_distance_recheck.json`，隔离目录 `php_segment_v4_full_20261109_distance_recheck`保存原始状态/provenance/45文件源码快照。原v4目录保留failed历史，未改写。session94202已完成。
- 重要运行事件：原v4 PID2227018已退出，日志显示GPUworker2227114 exit -9，manifest仍残留running，不能信它判断进程。内核日志无权限，`/proc/vmstat oom_kill=31`不能确认具体时刻；当时多批并行的主机内存压力很高，不能断言一定GPU OOM。两个小批结束后内存降至16GiB；已降低并发恢复原v4：PID2580132/session55445，**每卡1worker/3CPU**，同冻结源码/同输入/`--resume`，日志 `.cache/php_conversion/segment_v4/full_resume.log`。不要再启动第二个同目录控制器。plan.execution仍历史2/card，实际恢复命令是1/card；估计ETA必须用当前进程参数。18:52为43 pending/2 solving/35 checking/124 accepted/91 rejected/1 failed。
- 新隔离v11宽度适配8条：`php_segment_v11_width_pilot_20261109`，PID2600063/session41201，冻结 `.cache/php_conversion/segment_v11/frozen_runner`，45文件与plan核验并复制source_snapshot；两卡各1worker/2CPU。基于正式v9+距离/状态修复，不包含v10峰值惩罚。按模型中立位髋/肩安装宽度差，把脚/膝每侧移约2.86cm、手/肘每侧约7.26cm；以源root/torso水平朝向计算，支撑阶段取常量并投影回原接触面，飞行脚目标也同样调整。源数据/地形未改，旧门槛未放宽，segments保存morphology_shift_world/anchor_mapping。准备阶段检查通过，`segment_v11_width_preparation.json`；18:52为2 solving/6 pending，未有质量结论。工作区仍v9，**v11不在主工作区**。
- 资源约束：最多同时运行当前v4恢复+一个小批；每卡合计2个GPUworker、共5CPU验收，不再叠加第三批。启动新批前看主机可用内存，不只看显存。18:51主机使用73GiB/125GiB、可用50GiB；所有活跃进程须下一轮重新核实。
- 下一步：收齐v11判定宽度映射是否改善cfg0062自碰撞与攀爬，验证四控制不退化；继续诊断多点支撑与源/目标解剖约束。收齐v4恢复结果只作失败分布基线。用带三项验收的新算法对全部源/地形重跑及专门变体重定向，最终全2960槽位必须有真实通过证据。目标仍active，无外部阻塞。

### 19:05 更新：v11验证完成并启动全库

- v11小批已完成5 accepted/3 rejected、50/80、4871帧；四控制40/40保留，cfg0062从v9的0/10（自碰撞24.14mm）变为10/10（1.24mm）。三攀爬仍失败且部分穿透/姿态退化，不能说攀爬已解决。`segment_v11_width_pilot_report.json`独立审计验证60哈希。v11冻结环境首次测试缺少3份docs JSON夹具，复制真实夹具后完整重跑65项通过（146.9秒），`segment_v11_tests.txt`；失败测试日志原样保留，未放宽任何门槛。
- 主工作区现为 **php-segment-v11**，不是v9。已复制验证过的宽度求解器，全部45生产源码哈希与v11小批plan逐一一致。姿态和距离/中断修复均保留；没有采用v10峰值损失。冻结目录仍 `.cache/php_conversion/segment_v11/frozen_runner`。
- 四条最新版同步对照 `results/php_segment_v11_previews_20261109/comparison/all_comparisons.mp4`，1397帧、50fps、1920×640，单片/合辑帧数、视频和quality哈希核验；`segment_v11_preview_manifest.json`、`segment_v11_preview_summary.json`。躯干三维角RMSE约0.137°–0.373°、最大约2.01°；每条所有10变体通过三项检查。源动作、地形、帧率未改。
- v11在前、v9在后的独立全覆盖审计 `segment_v11_coverage.json`及`.index.json`：98/2960通过、缺2862，8条源动作全变体通过。`--require-all`按预期返回1，目标未完成。
- 新全库 `data/tienkung3_29dof/motions/php_segment_v11_full_20261109` 于19:03启动：**PID2659104/session22105**，两卡各1worker/2CPU，日志 `.cache/php_conversion/segment_v11/full.log`。selections为 `segment_v11_full_selections.json`，恰好296原始源、181236帧、2960槽位。先scan建计划，再在独立审计后复用8条小批终态；逐task字典与ID完全相同、所有产物复制前后hash核验、source_snapshot完整，来源 `reused_pilot.json`。无重复求解这8条，剩288条。19:05状态5 accepted/3 rejected/2 solving/286 pending；不要再建第二个同目录控制器。
- 全库初估按小批561.95求解秒/4871帧、剩176365帧/2worker约169.6分钟，另有验收尾部；只能解释为本轮求解约3小时，不能承诺全部地形通过时间。后续用实测当前任务更新。
- 旧v4恢复PID2580132/session55445于19:05仍活跃，但GPU求解已全部结束，仅35条CPU checking；169 accepted/91 rejected/1 failed（原异常已在独立目录复验为质量拒绝）。日志 `segment_v4/full_resume.log`。旧口径结果仍不能计入当前姿态覆盖。
- `.cache/php_conversion/live_coverage.py` 已增加真实 `/proc` 控制器pid/live字段，不能再仅凭manifest running判断活跃。19:03启动新批前主机使用35GiB、可用88GiB；当前最多两个批次，旧v4仅CPU+新v11两GPU。后续启动额外攀爬试验前应等待旧v4退出并检查内存，避免重现多个大模型进程叠加导致资源压力。
- 下一步优先：核实两个控制器；收齐旧v4原始失败分布；观察新v11全库并以三项门槛统计。并行开发攀爬的多点接触/初始化/锚点可达性，保持小批对照和旧通过资产；不能将50/80或98/2960当作完成。需要进一步验证源骨段方向的初始IK是否能避免当前全默认q初始化引起的局部错误分支，以及膝/手接触点在转面与滑动阶段是否被错误静态锚定；这些仅是待检假设，未实现。最终要求仍是全2960原始槽位全部真实通过，goal保持active。

### 19:24 更新：前倾复核、v12不推广、旧全库结束

- 主工作区保持验证过的v11；全库 PID2659104/session22105 仍在运行。19:23：26 accepted/3 rejected/2 solving/2 checking/263 pending，本批234槽位通过；跨正式姿态版本去重暂计275/2960，缺2685（provisional，未替代最终hash/labels审计）。新增 `.cache/php_conversion/status_current.py` 根据/proc实际worker参数及实测耗时估计：求解约113分钟、CPU验收工作量约76分钟，两阶段重叠。不要把运行结束估计当全2960通过承诺。
- v12仅在 `.cache/php_conversion/segment_v12/frozen_runner`：自由脚从G1世界位置目标改为按天工骨段重建、相对骨盆的脚底目标；接触脚仍追踪原世界锚点，接触掩码/点/面/法线/foot_target全部不变。准备验证表明两种悬空脚目标平均相差12–13cm，最大24cm；证据 `segment_v12_swing_preparation.json`。65项回归97.140秒通过，`segment_v12_tests.txt`。
- v12小批8条4871帧全部完成：4 accepted/4 rejected、40/80；独立审计56输出哈希通过，`segment_v12_swing_pilot_report.json`。45生产源码逐一与plan校验并复制到run/source_snapshot。没有程序错误。四控制40/40，三攀爬仍0/30，cfg0062从10/10退化到0/10，自碰撞1.24→6.55mm。低攀爬自碰撞55.2→21.5mm、43.2→7.36mm，躯干RMSE12.30→10.94°、14.18→12.40°但仍超标。没有新增通过槽位，**不推广v12，不替换当前v11**；同源对照 `segment_v12_swing_comparison.json`。pilot PID2702526已退出，session66132历史，不报告为运行中。
- 新生成同源同帧v3/v11前倾复核：`posture_v3_v11_comparison.json` 和 `results/php_segment_v11_previews_20261109/comparison/torso_lean_v3_v11.png`。所有motion哈希与视频manifest一致。首秒G1约2.95°、旧天工13.56–15.16°、v11约2.97°；整段有符号前倾RMSE分别旧14.06/13.33/11.37/16.20°，新0.182/0.058/0.089/0.107°。图像已查看，源与新版曲线紧贴；动态阶段旧版也可能少前倾或后仰，不能概括为整段恒定前倾偏置。全三维姿态指标另见v11_preview_summary，骨盆误差仍单列、不作已解决承诺。
- 旧v4恢复已结束，203 accepted/92 rejected/1 historical failed，共2950槽位检查、1950通过旧几何/摆臂门槛。原1异常在distance_recheck中10槽位全部完成并质量拒绝，合并历史基线2960槽位/181236帧、1950通过/1010拒绝；80条攀爬全部失败。当前独立全量核验 `.cache/php_conversion/segment_v12/audit_v4_baseline.py` 仍在运行，PID2700267/session18255，约100%CPU、760MB，日志baseline_audit.log，目标evidence/segment_v4_final_baseline_audit.json尚未生成。不要把任务终态计数说成该独立审计已结束，下一轮核实。
- 后续重点依然是攀爬多点接触可达性/源手膝滑动与换面意图/初始化。v12降低部分冲突但不够；不能靠松门槛或删除真实接触来增加通过数。完整目标仍active，必须全2960原始槽位通过。

### 19:52 更新：初始化/攀爬姿态推进、距离异常修复、地形专项重跑

- 上轮有实质进展；本轮新增两次完整8条小批、独立审计、两类数值异常修复和地形专项重跑，目标仍active，全2960原始槽位要求不变。
- 旧v4全量独立审计已完成：296源/181236帧/2960槽位、1950旧口径通过/1010拒绝，2292输出哈希检查，原1异常的隔离复验也纳入；`segment_v4_final_baseline_audit.json`。80条攀爬全部失败，998槽位含terrain_penetration、719含左膝滑动、650含右脚滑动、632含左膝距离、570/553含左右手滑动、550含自碰撞；原因非互斥。旧结果没有姿态策略，不能填当前覆盖。
- **v13隔离初始化** `.cache/php_conversion/segment_v13/frozen_runner`：v11原目标/原损失不变，先固定根部位姿，用天工FK拟合八段肢体向量和源躯干朝向500步，再进行原接触优化；后续窗口新帧也使用该初始化，不再全部复制上窗末帧q。不复制G1关节角、不变源帧或接触掩码。CUDA图首次预检失败因list索引在capture中触发分配，改为预先生成GPU索引tensor，原失败日志保留。直行269帧/低攀爬前512帧骨段向量RMSE14.25/18.79cm→2.40/2.99mm，`segment_v13_initialization_precheck.json`。38项主回归+8项delivery/pilot通过，不是此前65项全套；日志分为part1/part2。
- v13完整8条4871帧：50/80通过（四控制+cfg0062全部保留），三攀爬仍拒绝。正向低攀爬label生成1个程序异常，隔离复验后0/10，原候选不变；综合审计60哈希通过，`segment_v13_init_pilot_audit.json`、`segment_v13_init_comparison.json`、`segment_v13_distance_recheck.json`。原run保留5 accepted/2 rejected/1 historical failed，不能直接用通用report对原run声称全终态审计；综合结果通过cache脚本audit_completed.py核验。三攀爬躯干RMSE从v11的12.30/14.18/10.94°降至3.02/3.15/2.27°；两个转向样本的支撑项全过，仍有碰撞/姿态峰值问题。生产未升级v13。
- `segment_v13_pose_error_frames.json`：两条低攀爬最差躯干都在348帧、仅左右手支撑，超过5°主要289–353附近；并非主要发生在512帧窗口接缝。高攀爬最差206帧，多点支撑。`segment_v13_contact_reachability.json`为膝/脚同时固定锚点的保守距离界限诊断：两低攀爬仅11/7帧有同时约束，没有超过2cm的界限违背（正向最大19.95mm），**不证明整体可达，也不证明不可达**。
- **v14隔离小批已结束** `php_segment_v14_pose_polish_pilot_20261109`：v13基础上躯干权重20→80、碰撞修正8→12轮。8条4871帧50/80，四控制40+cfg0062 10均保留，无程序错误；独立审计60哈希通过，`segment_v14_pose_polish_pilot_report.json`及comparison。三攀爬躯干均通过，RMSE0.904/1.009/0.933°、max3.192/3.631/3.333°。正向低攀爬仍自碰撞18.05mm、左膝距离、手滑动；左转低攀爬只剩自碰撞6.08mm；高攀爬只剩自碰撞6.80mm/地形6.74mm。**仍不推广**，没有新增槽位。PID2767940/session3205已退出。45源码均核验并存source_snapshot。
- **主工作区只更新距离计算，求解器仍v11**。v13异常：重试有一个distance=0且两个witness全零的无效查询，与约43.5135mm有效结果混在一起使共识失败；新增零距离最近点必须位于对应几何包围球内（plane例外），初始查询也检查该条件。独立凸多面体QP给出43.5135166mm；fixture `distance_zero_witness.json`，原生重试/几何恢复回归通过。v14冻结于这一修复的第一步（只检查重试点），不含随后初始查询检查，勿改其冻结目录。
- 全库v11又出现同类异常：288081a3acfae5a8485c11d5（right_90_walk0.9s_step1.2s）在独立`_distance_recheck_288081a3`完成10槽位，0通过；bf725426c27609c4634fd8ba（left_45_walk0.6s_step1.2s）首次复验仍失败，保存原尝试。新诊断发现一个1e-7偏移查询返回负2.06497mm，而witness间距2.06605mm，其余17轴向查询都在正2.065mm附近；独立凸几何QP确认为正2.0650714mm。原候选过滤允许10微米自身距离/见证差异，现收紧为0.1微米，剔除这个不一致值；最大扰动仍10微米、不同查询共识门槛不变，未放宽任何质量阈值。fixture `distance_signed_witness.json`。第二次隔离`_distance_recheck_bf725426_signed`全部10槽位复验成功、0通过；`segment_v11_distance_recheck_bf725426_signed.json`。新第三异常d28554734995072be11d9bd9正在复验，session22023，日志 `segment_v13/full_d2855473_recheck.log`，下一轮必须核实结果。
- 最终代码48项相关回归130.090秒全部通过，`segment_v13_distance_final_tests.txt`；其中4个真实数值退化fixture单独通过 `distance_signed_witness_tests.txt`。新初始零见证和签名见证修复在main、terrain_retry冻结目录和v15准备目录；**不在正在运行的v11全库冻结目录中**，该批后续同类历史failed须隔离复验，不能改运行源码或假定异常自动消失。
- **地形专项重跑已启动** `php_segment_v11_terrain_retry_pilot_20261109`，PID2800668/session73760，2GPU各1worker/2CPU，日志 `.cache/php_conversion/segment_v11_terrain_retry/pilot.log`。从当时16个已完成非攀爬来源抽取38个仅terrain_penetration拒绝的槽位；逐task输出hash和三项策略核验，`segment_v11_committed_terrain_retry_plan.json`/selections。选8个不同来源，每个针对实际失败地形求解，完整9089帧、每条仍检查全部10变体；selections `segment_v11_terrain_retry_pilot_selections.json`。未更改地形。初始scan在最终距离修复前生成，已完整连同45源码快照搬到cache的initial_scan，未执行过；实际新plan与最终冻结源码45hash核验/source_snapshot齐全。19:52为2 solving/6 pending；原始槽位去重后计算增益，不能把重试80检查数直接相加。
- **后续v15仅准备、未运行** `.cache/php_conversion/segment_v15/frozen_runner` + `segment_v15_collision_peak_pilot_selections.json`。基于v14，只为native碰撞修正加`.1*max(frame_violation)`，保留mean和累计平面，拟解决残余孤立深穿透被全帧均值稀释；不包含v10其他峰值项/不包含v7丢弃平面/不改变独立门槛。带最终距离修复。尚无转换或回归结论，不能推广；下一轮等terrain retry释放资源后再考虑启动，勿同时开第三GPU批次。用12 collision rounds与v14保持其余配置一致。
- **全库v11仍在运行** PID2659104/session22105，19:52 47 accepted/17 rejected/3 historical failed/2 solving/1 checking/226 pending，本批429槽位；跨当前正式姿态版本去重暂计432/2960，尚缺2528，29来源全变体通过、47来源部分或全部通过。当前求解剩余约93.5分钟、CPU验收工作量约64分钟，两阶段并行；不能等同全部通过时间。活跃GPU批次只有全库v11+terrain retry；v13/v14已结束。下一轮先核实PID/第三异常复验/terrain retry结果，再推进攀爬；最终要求仍全部2960通过独立审计。

19:56 补充：第三异常d28554734995072be11d9bd9（low-climb/right_45_walk1.5s_step0.7s）在最终轴向重试修复下仍失败，session22023已结束，**不是仍运行，也不是已解决**。原状态/失败复验目录保留。新逐变体定位在variant7、geom16/box1，原生距离0但witness gap28.52mm；现场位于 `.cache/php_conversion/segment_v13/full_d2855473/distance_failure.npz/json`。逐方向扰动与独立凸QP诊断正在session40049，日志同目录directions.log，目标evidence/segment_v11_d2855473_distance_diagnosis.json；下一轮先读结果再决定修复，不能放宽一致性或质量门槛。

19:56 地形专项已有3条终态（25/30变体通过，非新增25），跨版本原始槽位去重现在435/2960，缺2525，新增3槽位；cfg0021实际目标6已通过（原9/10→合并10/10，重试本身丢0但原结果保留），cfg0023目标1已通过。真实进程：全库PID2659104，47 accepted/20 rejected/3 historical failed/2 solving/1 checking/223 pending、本批429槽位；专项PID2800668，3 accepted/2 solving/3 pending。专项剩余求解估计6.1分钟、CPU约2.7分钟（重叠），全库约94.7分钟求解。v14独立审计已完成60hash，v13综合审计已完成60hash。最终48项回归130.090秒通过并已归档。v15仍仅准备未运行；下一轮不能把旧v14/v13进程当活跃。


### 20:03 收齐地形专项、第三异常恢复、启动v15

- 第三异常d2855473已定位为另一例错误负距离：独立凸QP的真实分离约1.527991mm，而部分原生负距离/见证范数几乎一致，单收紧范数差异不够。主工作区 `make_labels.geom_distance` 在**负距离重试候选**上新增分离轴证明：沿见证方向用完整mesh顶点/box解析投影区间（其他有界几何用保守包围球；plane不使用此判据），若区间分离>1e-7m，则不可能穿透，剔除该候选。不是用多数票忽略冲突，也未放宽门槛或把碰撞忽略。原生有界重试仍保持原幅度和共识要求。
- 新fixture `distance_false_penetration.json`、诊断 `segment_v11_d2855473_distance_diagnosis.json`。**49项完整相关回归124.111秒通过**，另加真实box/box穿透的重试回归1项通过，确认不会过滤真实负距离；证据 `segment_v13_distance_separation_tests.txt`、`distance_real_penetration_test.txt`。真实穿透测试是在49项运行加载后新增，因此是49+1，不声称一次运行50项。`git diff --check`通过。
- 第三条最终隔离复验 `php_segment_v11_full_20261109_distance_recheck_d2855473_separated`已完成全部10槽位822帧、0通过（质量拒绝），motion未改；`segment_v11_distance_recheck_d2855473_separated.json`。session13887已退出。此前未成功的signed复验及现场保留。全库当前原3个failed历史标记都已在独立目录完成复验，不能删除原标记，也不能把manifest failed3继续称为3个未处理异常；后续新增异常仍须逐条核实。
- **地形专项8条已结束**，PID2800668已退出/session73760历史。9089帧、80检查中70通过；独立审计72输出hash通过 `segment_v11_terrain_retry_pilot_report.json`。8个指定失败地形编号全部修复，较原v11补8槽位；其中1槽位早已被另一个当前版本覆盖，**全局实际新增7槽位**，`segment_v11_terrain_retry_comparison.json`。有些新轨迹在其他原通过变体退化，因此保留原资产并按原始来源/变体择取，不覆盖旧文件。不能把70/80计成新增70，也不能把目标8/8解释为这8条所有变体都完成；原+新并集仍有部分其他变体待重试。
- **v15已启动，覆盖之前“仅准备”的状态**：run `php_segment_v15_collision_peak_pilot_20261109`，PID2841706/session12910，冻结 `.cache/php_conversion/segment_v15/frozen_runner`，日志同父目录pilot.log。基于v14初始化+80躯干权重+12碰撞轮，仅native碰撞损失加`.1*max(frame_violation)`，其他均值/平面累积/独立阈值不变；包含最后的分离轴距离修复。45源码已与plan核验并复制source_snapshot。8条同样完整4871帧/80槽位；预计小批10–15分钟，尚无质量结论，不推广。主工作区求解器仍v11（距离函数与测试已更新）。
- **20:02真实全库状态**：PID2659104/session22105仍活跃，48 accepted/26 rejected/3 historical failed/2 solving/1 checking/216 pending，本批439槽位。正式版本去重暂计**449/2960**、缺2511，33来源全变体通过；实时统计不是全库最终独立审计。全库剩余求解约93分钟、CPU工作量约62分钟，两阶段并行。只有全库v11+v15两个GPU批次，不再启动第三批。
- 下一步：核实v15与全库；收齐v15，若控制不退化且攀爬新增通过再推广相应改进并生成新版对照。继续对所有未覆盖地形做对应变体求解，先从已验证有效的38个专项计划中减去已有通过编号（含其他版本已覆盖），不要盲重跑已通过槽位。第三异常无需再重复复验；新故障另定位。主目标始终全部2960原始槽位真实通过，当前449远未完成。


### 20:29 覆盖快照、地形扩大重跑与 v16 约束试验

- v15 已完成，PID2841706退出。8条4871帧、50/80通过，四控制和cfg0062保持；三攀爬仍拒绝，无新增槽位。60输出哈希独立审计：`segment_v15_collision_peak_pilot_report.json`；对照 `segment_v15_collision_peak_comparison.json`。最差帧native碰撞权重未带来可靠收益：低攀爬直行躯干最大40.42°、左45最大15.74°，不推广。
- 地形8条小批有效后，从全部当前已提交通过结果中扣除已有覆盖，生成 **31个原始缺失槽位、15条来源、33854完整帧** 的专项计划。不能把每次检查10变体计为新增310槽位。计划和清单：`segment_v11_terrain_retry_expand_plan.json`、`segment_v11_terrain_retry_expand_selections.json`。运行目录 `php_segment_v11_terrain_retry_expand_20261109`，PID2867285/session70304，冻结 `.cache/php_conversion/terrain_retry_expand/frozen_runner`，日志同父目录run.log，GPU0单worker/CPU1。45源码已核验并复制snapshot。
- 并发安排更新：全库v11使用2GPUworker/2CPU，专项与算法小批分别1GPUworker/1CPU，总量仍4GPUworker和4CPU。此前“只能两个控制器”是自定调度安排，现改为三个控制器共享同一worker总预算。不要同时多开另一完整GPU小批；等待当前算法小批退出后接替。
- 隔离 v16：保留肢段初始化与碰撞优化，取消自由根朝向变量，参数化root旋转，使原生FK躯干精确匹配中立校准后的源朝向；另用Lipschitz包络对关节增量投影到0.95倍目标速度上限。固定前缀在投影前后恢复，已提交帧字节不变。不是修改验收结果。新增3个不变量测试和全部53项测试通过，`segment_v16_constraints_tests.txt`、`segment_v16_tests.txt`，CUDA图及固定前缀smoke通过 `segment_v16_graph_smoke.json`。
- v16完整小批运行 `php_segment_v16_constrained_pilot_20261109`，PID2889817/session19783，GPU1/CPU1，12碰撞轮；同8条4871帧。20:28求解已结束，仍有3条CPU验收；三攀爬姿态、自由臂和速度都通过，但碰撞全部拒绝，低直行还拒绝左膝支撑距离。低直行37mm级手腕/腿碰撞集中在右手支撑结束帧350，原生contact和mesh投影核验记录 `segment_v16_collision_diagnosis.json`；不能忽略碰撞以凑通过。主工作区求解器仍v11。
- 全库冻结v11继续产生旧距离异常：新增c4bcaa27、e8dd4a97、c89ee388均已使用当前距离修复，在独立目录完成全部10变体验收/标签核验，motion未改，全部质量拒绝。证据 `segment_v11_distance_recheck_<前8位>_separated.json`。加上前三条，当前6个历史failed都已恢复成明确质量结论；原failed标记保留。新出现的其他ID要另行核实。
- `index_php_conversion_variants` / `audit_php_variant_coverage` 新增显式 `--allow-running`。默认仍拒绝未终态父批次；可选择核验已提交accepted，跳过pending/checking，不把其他状态计入通过。追加源/任务/报告/候选绑定、完整帧数与50Hz、原地形完整性、导出运动和地形编号一致性。manifest内容及哈希来自同一次read_bytes，避免并发状态刷新导致错配。4项防护检查 `committed_index_guards.txt` 通过，包括错误source、candidate hash、variant编号必须拒绝。
- 首次运行中快照522/2960；加强帧/地形导出绑定后复核为 **533/2960**、缺2427，42来源全变体通过，`segment_v11_committed_coverage_20261109_2028.json/.index.json`。这是核验期间已提交结果的覆盖下限，不是全库完成证明；`--require-all`正确返回1。实时统计会继续增长，不可混同快照。
- 20:28实测全库PID2659104仍运行：58accepted、50rejected、6historical failed、179pending、2solving、1checking，本批537槽位；预计求解剩73.5分钟。专项6accepted/1solving/24pending，50个检查槽位通过（非新增50），预计剩53分钟；各批CPU并行。
- 已准备v17在独立 `.cache/php_conversion/segment_v17/frozen_runner`：从v14软姿态80权重和native均值碰撞出发，仅加已测试速度投影；保留自由根朝向，试验24碰撞轮以检查收敛。此时只在跑回归session78240，未启动完整求解；需回归与CUDA smoke通过、v16控制器退出后再启动。v16精确姿态试验不推广。


### 20:32 v16收齐、v17接替启动

- v16 PID2889817已退出。完整8条4871帧、5accepted/3rejected/0failed、50/80通过；独立审计60输出哈希通过 `segment_v16_constrained_pilot_report.json`，与v14比0新增0丢失 `segment_v16_constrained_comparison.json`。没有攀爬通过，不推广。
- v17已经真正启动，覆盖20:29“仅准备”：PID2944122/session43641，`php_segment_v17_velocity_pilot_20261109`，冻结 `.cache/php_conversion/segment_v17/frozen_runner`，日志同父目录pilot.log，GPU1单worker/CPU1，24碰撞轮。45源码哈希核验并已复制source_snapshot，8条完整4871帧80槽位。回归为三次调用 **44+3+5=52项** 全通过（不要写成一次52）；记录 `segment_v17_tests.txt`、`segment_v17_delivery_tests.txt`、`segment_v17_pilot_tests.txt`。CUDA图及固定前缀检查 `segment_v17_graph_smoke.json` 通过。预计10–20分钟，尚无质量结果。
- 第7个全库历史异常cb043e0aee7e99227645ae8b（high-climb/run_left_90_0.6s_step0.8s）也已独立复验完成640帧10变体、0通过；`segment_v11_distance_recheck_cb043e0a_separated.json`，不是仍未解决的程序失败。原标记保留。当前7个已出现异常均已恢复明确质量结论。
- 20:32全库PID2659104：58accepted/54rejected/7historical failed/2checking/2solving/173pending，本批537槽位；求解约70.6分钟。专项PID2867285：8accepted/1solving/22pending，66检查槽位通过，约47.2分钟；66不是新增数。全局实时去重555/2960、缺2405、45来源全变体通过；最近独立533覆盖快照为更早核验下限。
- 下一轮先核实v17进程及结果、全库新增failed ID（前三个288081a3/bf725426/d2855473及后四个c4bcaa27/e8dd4a97/c89ee388/cb043e0a都已恢复，不重复）。如v17仍失败，按碰撞/支撑的具体帧与阶段研究接触落点适配；不可放宽门槛、删除原始变体或修改源时间。v17未比现有基线新增通过前，不推广、不要重新生成没有改进的四段“新成功”录像。


### 20:51 源接触标签假零距离定位与v18试验

本轮属于有实质进展：完成v17独立审计，发现并证明源接触标签中的数值假零，新增保留真实接触/输入不变/复核记录防篡改测试，启动v18完整小批。目标仍全部2960原始槽位，不标完成。

- v17已结束，PID2944122/session43641为历史：8条4871帧，50/80，5accepted/3rejected/0failed，60输出哈希审计 `segment_v17_velocity_pilot_report.json`。对照 `segment_v17_velocity_comparison.json` 为0新增0丢失；三攀爬姿态/速度通过但仍碰撞拒绝，低左45自碰撞峰值60.317mm，低直行37.141mm。速度投影和24轮并未优于v14，不推广，也不作为v18求解器基线。
- 初步检查脚部姿态/接触片段 `climb_foot_patch_diagnosis.json` 发现关键异常：低攀爬直行第488、491帧源left_knee标签为0；中间489、490真实标签约32mm，但binary_closing将两零点与中间帧合并为488:492四帧支撑。原生G1重算和严格见证修复分别得到31.515009mm、32.745098mm，`source_zero_contact_diagnosis.json`。
- 另作**不依赖原生距离算法**的分离平面证明：将完整G1小腿mesh所有顶点与原box精确区间投影到世界Z轴，间隔下界恰为上述31.515/32.745mm，证明这些姿态不可能接触。`source_zero_contact_certificate.json`。这不是以目标机器人失败反推删除源接触；是源几何自身证伪了存储的0距离。源motion、terrain、标签文件均保持原哈希。实际应称“源动作的接触标签数值错误”，不能归咎于原始动作录像。
- 隔离v18新模块 `.cache/php_conversion/segment_v18/frozen_runner/supermimic/acp/source_contacts.py`：`validated-source-zeros-v1`，只对存储abs(distance)<=1e-10的源标签，使用原G1、当前原始地形、对应身体的全部collision geoms与全部地形面重新做strict native queries（保留已验证的有界见证重试）；不使用采样fallback，不修改源文件/帧/地形。更新距离、见证点、法向与面编号后，再运行原接触分类和片段合并。每个零点保存original/validated receipt。真实零接触仍可保留，不是一律删除零值；非零输入逐字节保留。
- `prepare`使用修复副本；转换器将每个unique variant的receipt与策略写入intent，plan身份包含策略，源码fingerprint新增模块为46文件。CPU evaluate_task 从原始motion/labels/terrain重新计算全部receipt，缺失/错配必须失败，才做原几何/自由臂/姿态验收；冻结版独立audit也再做源receipt验证。**运行该v18最终审计时使用其冻结runner的tools.audit_php_conversion**，主工作区审计器尚未包含这条新策略核验。
- 测试fixture `source_zero_contacts_fixture.json` 来源原488/491及邻帧；新增3个测试（真实假零恢复且输入不变、真实ground touch保留、独立重算拒绝错配和缺失receipt）。53项完整相关测试126.927秒通过 `segment_v18_tests.txt`。v0预检 `segment_v18_source_contact_precheck.json`：三攀爬15个零标签发生修正，13个实际>5mm，2个仍接触；488:492假膝支撑消失，356:392真实膝支撑保留。
- 全地形预检 `segment_v18_all_variants_precheck.json`：8条对应53个不同地形（别名仍占80原始槽位），134个零标签复核，125个>5mm、9个保留接触条件；所有receipt从源输入独立复算一致，原motion/labels哈希未变。只是准备阶段核验，不能计为目标合格。
- v18最终基线明确为 **v14软姿态80 + 肢段方向初始化 + native均值碰撞12轮，仅新增源零标签复核**。最初曾从v17复制准备目录，随后在plan生成及启动前回到v14求解器；最终不含速度投影和精确姿态参数化，速度/姿态独立门槛仍保留。不可混淆。v17测试文件test_php_constraints已从未运行的v18准备目录删除，改用source_contacts测试。
- **v18已运行**：`php_segment_v18_source_contact_pilot_20261109`，PID3020029/session35217，冻结 `.cache/php_conversion/segment_v18/frozen_runner`，日志同父目录pilot.log，GPU1单worker/CPU1，8条完整4871帧/80槽位；46源码与plan核验并复制source_snapshot。20:50左右启动，预计10–15分钟。没有质量结果前不推广。
- 当前全库v11 PID2659104仍运行（冻结旧源标签行为保留）；20:51为72accepted/73rejected/7historical failed/2solving/7checking/135pending，本批663槽位，求解剩约51.5分钟，CPU并行。31地形专项 PID2867285为16accepted/1solving/14pending，126检查槽位通过（非新增126），剩约30.2分钟。7个历史程序异常均已有独立复验，不要重复处理；新ID另核实。
- 20:50全局实时去重673/2960，缺2287，55来源全变体通过（实时统计非独立最终审计）。最近独立533快照时点更早。主工作区求解器仍v11，不修改活跃冻结runner。下一轮先核实v18结果，若小批实质改善，再按用户要求生成四条完整新版同步录像并推广；若失败，按具体帧继续定位，不能靠放宽门槛或删除原始变体完成目标。

20:53补充：v18首条cfg0062完成10/10（仍为已有覆盖，不算新增），当前1accepted/1solving/6pending。全库v11已75accepted、本批692槽位，预计求解剩约48分钟；专项17accepted/1solving/13pending，剩约28分钟。全局实时去重729/2960，缺2231，60来源全变体通过；未替代最终独立覆盖核验。


### 21:32 v18 集成、v19 攀爬进展与覆盖更新

- 主源码现为 php-segment-v18，源零距离独立复核、肢段初始化、软躯干跟踪与 native 均值碰撞；默认 collision_rounds=12 与已验证小批一致。四条完整录像 1397 帧/50 fps/40 原始槽位及哈希、源时间、地形一致性均核验；53 项主工作区回归通过（129.711 秒）。证据 segment_v18_promotion.json、segment_v18_main_tests.txt、segment_v18_preview_summary.json。全库和其他活跃 frozen_runner 均未修改。
- v18 小批最终 8 条/4871 帧，50/80、5 accepted/3 rejected/0 failed，独立冻结版审计包含源 receipt 重算，60 输出哈希通过。相对 v14 无新增/丢失通过；低直行假膝支撑已修正，真实膝阶段保留，但手臂碰撞/滑动仍失败。此次集成不声称三条攀爬通过。
- 全库 v11 的全部 80 条攀爬/800 槽位已形成明确质量结论，0/800。7 个历史程序异常以不可变独立复验覆盖；segment_v11_all_climb_baseline.json 保留重叠拒绝原因。全部800均拒绝躯干三项指标，另有滑动、碰撞、支撑等问题。
- 新出现的两个台阶距离异常 041965edb7591d4d72b2c02a（high-step/run_left_45_1.5s_step1.0s，499帧）和 e298f17ec7dcca7f43c68ad7（run_right_90_1.5s_step1.0s，486帧）已在独立 *_distance_recheck_<id8>_separated 目录完成全部10变体验收与标签，分别10/10通过，原候选未改。主索引独立核验20槽位 segment_v11_recovered_steps.index.json；至此已出现的9个历史failed均已恢复，不要重复复验。
- 主 index 工具新增 assessment_only 恢复目录支持：核验原父plan哈希、精确任务、原failed状态、motion_changed=False、父/恢复候选哈希、评估代码快照后，在内存构造状态，不伪造plan或改写旧父批次。recovery_index_tests.txt 和 committed_index_guards_v2.txt 通过。主索引对 numeric version>=18 强制源零receipt从原输入独立重算。注意主 convert/check/audit 的 v18 检查仍为精确版本，v19 冻结版本已用通用 >=18 helper，未来推广时要一起同步。
- 较早独立覆盖快照 segment_v11_committed_coverage_20261109_2100.json/.index.json 为920/2960、缺2040、79来源全通过；包含运行中已提交输出和041恢复，不含后来的e298/v19等。require-all正确退出1；是核验期间覆盖下限，不是最终全库审计。当前实时统计 1510/2960，缺1450，时点2026-11-09T21:30:17.742391+08:00，不能混同历史快照。
- v11扩大地形31专项已完成，31 accepted/0 rejected/0 failed，240/310检查通过。独立审计 segment_v11_terrain_retry_expand_report.json；相对规划时哈希绑定覆盖去重，31个目标全部通过，新增31原始槽位，segment_v11_terrain_retry_expand_comparison.json。
- v18接替地形专项已启动：php_segment_v18_terrain_retry_20261109，PID4141811/session60903，GPU0单worker/CPU1，27个缺失原始槽位/11来源/14470帧；冻结 .cache/php_conversion/terrain_retry_v18/frozen_runner、日志run.log，46源码哈希已核验/snapshot复制。仅选sole terrain rejection且排除已有通过和在途重跑。计划 segment_v18_terrain_retry_plan.json。注意该计划不是此前准备但未单独启动的 v18_climb_expand 计划。
- v19 从v18建立独立runner：只调整自由臂目标在真实接触切换边缘的权重。每阶段n=min(8,phase_length//4)帧正半正弦平方渐变，仅邻接接触边界；全自由动作权重全1，短阶段至少一半保持全权重。位置使用该权重，速度用相邻权重乘积；自由臂独立验收仍对全部自由帧不加权，不改摆幅/时序/碰撞/姿态门槛。source_validation_required 使用数字版本>=18。56项回归120.619秒、CUDA图与固定前缀检查通过，segment_v19_tests.txt/segment_v19_graph_smoke.json。
- v19完整20条/13601帧/200槽位（15攀爬、5控制），原8+额外12攀爬，PID3444309/session9561，GPU1单worker/CPU1，12碰撞轮，.cache/php_conversion/segment_v19/frozen_runner及pilot.log。46代码身份已核验。额外12条由 segment_v18_climb_expand_plan.json 选出，但实际加入v19，没有启动独立v18攀爬扩大批次。
- v19首条低攀爬 forward_walk1.5s_step1.2s，894帧10/10通过；self从v18的16.7647mm降至1.7347mm，terrain1.6824mm，躯干RMSE0.7987°/P952.2669°/max2.2965°，左右手滑动0.02560/0.02848m/s。segment_v19_first_climb.index.json独立核验源receipt和输出，完整894帧视频及哈希核验 segment_v19_first_climb_preview.json。low-left45_walk1.5s_step1.2s 9/10，缺variant2。21:30已5accepted/2rejected/1solving/12pending、49槽位；新增forward_walk1.5s_step0.7s和left_90_walk0.6s_step1.2s各10。未收齐/未推广。
- **剩余重要限制**：v18右手释放350→351根位置跳变0.23677m/20ms，源仅0.007206m。v19首攀爬改善到0.129338m，仍过大；现有运动学验收没有独立根部速度/平滑性门槛。证据 segment_v18_contact_transition_diagnosis.json 与 first_climb_preview 明确记录。不能因为接触/摆臂/躯干通过就称整段质量或动力学可执行性已解决。下一步研究浮动根dp平滑项按50Hz物理尺度的速度/加速度约束，不能只删帧或放宽门槛。尚无v20实现。
- 21:30实际控制器：全库v11 PID2659104，147accepted/74rejected/9historical failed/58checking/2solving/6pending，1405本批槽位，GPU剩1.7分钟、CPU工作量14.7分钟；v18地形4accepted/3checking/1solving/19pending，30检查槽位，GPU14.8分钟/CPU20.9分钟；v19剩约20.3分钟GPU、9.9分钟CPU，各阶段重叠。4GPUworker/4CPU总预算。时间只代表批次，不代表全部通过目标完成。
- 后续：收齐v19并独立审计，检查四条完整控制后才考虑推广；检查全库新failed ID并恢复、继续未覆盖地形；另有全库low-step/right_45_walk0.8s_step0.5s同时terrain/right_knee support拒绝，不在sole-terrain专项，需单独用新版求解。继续处理根跳变，不遗漏。所有2960原始槽位仍必须通过，不能删变体、改源帧率或把新重试重复计数。


### 2026-11-10 13:50 中断核实、检查点恢复与 v20

上轮有实质进展：实现v20根部平滑隔离试验、完成59项测试与CUDA检查、启动8条完整小批，新增原始标签假零诊断并修复主验收器未来版本策略；随后会话中断，尚未把下述变化写入文档。本轮检查/proc确认四个控制器均已退出，不能把旧日志中的running当作仍在运行。

- 截至恢复前，实时去重1763/2960、缺1197，168来源全部变体通过；只是现有几何/摆臂/躯干门槛，不含根部平滑性完整验收。原始完整范围296条/2960槽位保持。
- 已核验四套冻结runner的全部源码哈希与原plan一致，并使用原参数--resume恢复，没有--retry-failed。checking恢复为solved后验收，solving恢复pending并读取窗口检查点，已有终态资产核验复用。全库v11无剩余GPU求解，只有CPU验收；9历史程序异常仍保留，均已有独立恢复，不重复。
- 新PID：全库v11 2840841；v18地形27专项2840842；v19接触切换20小批2840843；v20根部8小批2840844。各自日志为原cache子目录/resume_20261110_1350.log，启动命令证据resume_20261110_1350.json。总实际求解3GPUworker，验收最多5CPUworker；恢复前主机可用108GiB，资源充足。这取代之前自定4CPU调度安排，并非用户资源限制。
- v20位于.cache/php_conversion/segment_v20/frozen_runner，正式主求解器仍v18。基于v19，仅把原弱dp帧差平滑换成源相对根部修正的物理时间导数：0.25*mean((50*diff(dp))²)+0.001*mean((2500*diff2(dp))²)。常量体型平移不受惩罚，源自身运动保留；关节、接触、源零标签复核、自由臂和姿态验收不变。不是新的独立动力学准入门槛，也不声称已经解决跳变。
- 新增3个根部测试：常量平移零成本、不同采样率相同物理速度成本、尖锐释放跳变比平滑过渡成本大且梯度方向正确。初次测试漏把测试fps传给默认50Hz函数，修正测试调用后完整59项127.704秒通过，segment_v20_tests.txt；初次失败日志保留，不隐瞒。CUDA图/固定前缀逐字节不变检查segment_v20_graph_smoke.json通过。46运行源码与plan一致并复制source_snapshot。
- v20完整小批8条4871帧80槽位，包含四预览、三攀爬和长转弯cfg0062。run php_segment_v20_root_pilot_20261109；上轮PID630302仅到首条solving即中断，无质量结论。此次以同一plan恢复，不另建批次或篡改身份。
- 独立完整根部诊断segment_v18_v19_v20_root_diagnostics.json（后续可追加v20候选）绑定源/目标哈希，报告原50Hz下全部帧的速度和源相对速度/加速度。已通过既有门槛的v19攀爬仍有最大单帧0.123–0.187m，不能将现有quality_pass等同完整动作质量。因此v19尚不推广，即使更多攀爬通过现有门槛。
- 另一非攀爬失败motions-low-step/right_45_walk0.8s_step0.5s也定位为源标签假零：207/210/212帧右膝原0，严格原G1复核分别54.36/60.63/80.21mm；旧207:213假支撑六帧在验证副本中消失。source motion/labels哈希未改。segment_v18_step_source_contact_diagnosis.json只是准备阶段诊断，尚未对目标重跑，不能计通过；需要加入单独新版修复批次（不在sole-terrain27专项中）。
- 主source_contacts新增source_validation_required(version>=18)，check/audit均调用，未来v19/v20不会漏掉源receipt独立重算。4项source_contacts测试1.517秒通过，source_validation_future_versions_tests.txt；此前taskmemory提示主audit只检查精确v18已过时。此改变没有推广v19/v20求解器，也没有修改活跃冻结runner。
- v19已提交快照segment_v19_committed_2137.index.json核验6来源59槽位，其中5条攀爬49槽位；更晚实时结果增长，不混用时点。left45_walk1.5s_step1.2s缺variant2的具体拒绝是right_knee support_distance/support_slip，需要对应原地形求解；不是仅terrain_penetration。
- 下一步收齐v20比较跳变与全套门槛、收齐其余三个恢复批次并独立审计；对全库剩余缺失槽位去重规划后再跑。正式推广仍要求四完整同步录像验证。不得把2960目标缩小为已有合格子集。


13:54新增证据：完成全部296个v11候选的根部连续性诊断，segment_v11_all_root_diagnostics.json。问题不只攀爬：旧标准accepted的行走最大单帧位移0.36795m，高台阶0.36351m，低台阶0.22515m；最大额外速度约16.6m/s。不是窗口接缝专属，多个<512帧短动作也出现。现有几何/摆臂/躯干quality_pass无法证明整体连续性，所有旧计数只能称既有门槛通过；最终交付需要补充独立根部连续性检查，不能直接沿用旧通过标记。

v20首条cfg0062长转弯1142帧已10/10通过现有门槛，源相对根部诊断：v18/v19最大单帧0.30453m→v20 0.07047m；额外根速度max14.0873→2.08437m/s，额外加速度max805.01→60.5475m/s²，速度RMSE0.33485/P950.71252m/s。源速度max2.6238m/s，v20目标max3.52357m/s。segment_v18_v19_v20_root_diagnostics.json保留哈希和全段指标。其余小批尚待完成，不推广。


13:59结果与交接：

- 诊断出的low-step/right_45_walk0.8s_step0.5s源假膝接触专项已真正完成：php_segment_v18_step_repair_20261110，465帧10/10全部通过，独立审计segment_v18_step_repair_audit.json验证9输出哈希、源receipt、原始地形与标签。旧v11该源0/10，新增10个原始槽位；保留v18根部连续性尚未独立准入的限制。PID2861790已退出；日志.cache/php_conversion/terrain_retry_v18/step_run_20261110.log。46源码哈希与plan核验，source_snapshot完整。
- v20低攀爬直行已10/10通过既有全套验收：variant0 terrain最大2.12253mm/self1.93955mm，躯干三维RMSE0.88326°/P952.52514°/max2.55362°，手/肘保真全部通过。全段根部最大单帧0.051106m；源相对根速度RMSE0.218278/P950.442702/max1.31294m/s，加速度max30.1264m/s²。独立资产索引segment_v20_climb_result.index.json已验证该源和cfg0062共20槽位（已有槽位质量改善，不是20新增）。
- 完整v20攀爬录像results/php_segment_v20_climb_preview_20261110/comparison/01_low_climb_comparison.mp4，894帧50fps1920x640，同源时间/同地形/质量及录像哈希验证segment_v20_first_climb_preview.json通过。7.02s释放附近截图已查看。该录像及证据明确运动学、根部独立准入尚待完成，不宣称物理/策略跟踪通过。
- 全段根部速度/额外加速度对比图results/php_segment_v20_diagnostics_20261110/root_continuity.png已生成并查看，脚本.cache/php_conversion/segment_v20/plot_root.py；可见旧v18/v19孤立尖峰显著减少。所有指标绑定源/目标哈希，见segment_v18_v19_v20_root_diagnostics.json。
- 13:58:23实际运行：全库v11 PID2840841，188accepted/74rejected/9historicalfailed/25checking，验收约4.9分钟；v18地形PID2840842，20accepted/7checking，约6.6分钟；v19 PID2840843，11accepted/6rejected/3checking，约0.8分钟；v20 PID2840844，3accepted/1checking/1solving/3pending，29槽位通过，求解约2.7分钟/验收约2.2分钟。v20左45仍9/10，原variant2仍需专门求解。恢复日志统一resume_20261110_1350.log。
- 最新实时去重1986/2960、缺974（2026-11-10T13:58:24.016281+08:00）；只代表既有几何/摆臂/躯干门槛。根部问题已证实也影响非攀爬，最终合格数尚未确认，不能把这些槽位直接当作最终可用量。
- 下一轮收齐v19/v20/全库/地形专项；终态独立审计要使用当前主audit（已含>=18receipt校验）或相应冻结版，不使用旧精确v18版本。v20收齐后比较全部8源全部原始变体及根部指标，四条控制动作完整录像验证后才考虑推广；在最终统计中加入独立根部连续性验收，不能通过放宽指标规避跳变。当前尚未定义根部准入阈值，也没有推广v20，正式主求解器仍v18。


### 2026-11-10 14:11 正式v21与根部连续性准入

本轮有实质进展：收齐v19/v20及旧全库/地形批次；完成独立根部准入、四条完整录像与65项回归；正式源码更新v21并启动全部296源重跑。目标仍全部2960原始槽位，不标完成。

- v19终态20条13601帧，139/200、14accepted/6rejected/0failed；独立审计segment_v19_transition_pilot_report.json验证156输出哈希和源receipt。v20终态8条4871帧，69/80、7accepted/1rejected/0failed；独立审计segment_v20_root_pilot_report.json通过。v20低左45只缺variant2（右膝支撑距离/滑动），高攀爬run_right_90_0.6s_step0s全部10变体仍terrain/self拒绝；其他6来源全通过。源时间和地形均原样保留。
- 正式新模块supermimic/acp/root_continuity.py：root-continuity-v1，原始50Hz全部帧，target-source根位置修正的一/二阶秒制差分。固定工程筛查阈值：额外速度RMSE<=0.5m/s、P95<=1m/s、max<=2.5m/s；额外加速度max<=100m/s²。允许常量体型/高度偏移，保留源运动。该门槛用于排除跳变，不是电机/动力学/策略可执行性标准，也不是根姿态或骨盆保真全部解决。没有改变原碰撞、接触、自由臂、躯干门槛。
- 这些阈值在全库旧资产复筛及新全库求解前固定，不能随失败调整以凑覆盖。纯函数score_root_positions/score_root_files独立读原源和候选完整帧，绑定source/motion hashes；单帧瞬移、冻结根、末帧跳变、改变fps/帧数/NaN都必须拒绝；常量平移和共同坐标旋转不改变判断。
- 主源码VERSION现在php-segment-v21，求解目标与已验证v20相同（源相对根速度0.25/加速度0.001正则、自由臂接触切换渐变、躯干软姿态、源零标签复核），默认12collision rounds匹配小批。新版本>=21的CPU evaluate_task把根部评分加入quality_pass合取，根失败不导出合格资产；plan包含ROOT_POLICY，quality记录完整带哈希根评分和每变体root_continuity_pass。source_fingerprint为47文件。
- 主audit对>=21要求plan/report根策略存在并从源/候选独立重算整份评分；verify_fidelity_report验证新增根门槛合取，拒绝省略根失败或篡改布尔标记。主index对>=21也强制重算并绑定报告，旧报告可通过--require-root-continuity重新筛查。coverage严格模式验证每个条目的根证据与source/motion哈希绑定。最终交付必须使用--require-root-continuity --require-all，不能使用旧无根检查的通过数。
- 65项完整相关测试73.210秒通过segment_v21_tests.txt；包括新根检查、伪造证据/遗漏准入拒绝、原摆臂/躯干/接触/导出回归。单独根索引+CoverageTests为9项（root_continuity_index_tests.txt），完整65包含后来追加的合取/未来版本测试。CUDA求解目标与v20已经验证的相同，活跃旧runner未修改。
- v20全部8源均通过独立根筛查，和原几何/摆臂/姿态交集仍69槽位，segment_v20_root_continuity.json、segment_v20_strict_coverage.json/.index.json。后者全库分母2960，所以缺2891；只是此小批的覆盖，不是所有历史资产复筛结果。
- 四条完整控制新录像results/php_segment_v20_previews_20261110/comparison/all_comparisons.mp4，共1397帧50fps，40/40原变体；每条完整源时间、地形、视频/运动/质量哈希以及独立根筛查全部验证。segment_v20_preview_summary.json、segment_v20_preview_manifest.json。这是正式推广证据，单条攀爬录像不能代替这四条。
- 正式推广证据segment_v21_promotion.json；全库任务php_segment_v21_full_20261110于2026-11-10T14:09:48.859264+08:00启动，PID2923075，2GPU各1worker/2CPU验收，冻结.cache/php_conversion/segment_v21/frozen_runner，日志full.log。完整296条181236帧2960槽位及每源inventory哈希/帧区间验证、47源码哈希核验且source_snapshot复制。GPU0 worker2923173/GPU1 worker2923174均已ready并开始前2条，首次实测14:10为294pending/2solving。按小批粗估2–3小时，只是本轮，不承诺全部通过。
- 首次scan因选择文件复制到冻结docs而不是主docs导致FileNotFound，在plan生成前失败，日志scan.log保留；补齐主docs选择文件后scan_corrected.log验证296/181236/2960，随后才启动。没有篡改已有运行plan或重复控制器。
- 全库v11已终态212accepted/75rejected/9历史failed，2032既有槽位；加9个不可变恢复复验后全部296源181236帧/2960槽位有明确结论，2052旧门槛槽位通过，segment_v11_full_final_rejections.json。9个程序异常均已解决为质量结论。根连续性未包含在这2052中。
- v18地形27专项已终态27accepted/0failed，232/270检查通过；独立审计segment_v18_terrain_retry_report.json。相对规划覆盖哈希回执去重，27/27指定目标通过，新增29原始槽位（不是232新增），segment_v18_terrain_retry_comparison.json。额外2槽位来自同轨迹在非指定地形也通过。
- 独立旧资产全覆盖根复筛仍运行：.cache/php_conversion/strict_coverage_20261110.py，PID2917198/session23406，19父目录（优先v20，含两个accepted recovery），输出将为strict_root_coverage_20261110_1406.json/.index.json。14:11进程CPU100%、已5分钟，尚无结果，不能重启或当作卡死。该扫描列表在新v21plan创建前固定；下次重跑脚本须处理新的运行中v21（目前assert all_terminal会失败），可改显式allow_running并记录快照，或等其结束。
- 实时计数脚本.cache/php_conversion/live_coverage.py已改为包含独立根评分，按source+candidate哈希缓存到root_continuity_cache.json，旧三门槛数单独legacy_geometry_arm_posture_slots。首次填充缓存正在PID2925527/session80812运行；输出文件刷新前仍可能是旧时间/旧口径，不可误报。缓存只做实时视图，最终仍需全资产独立审计。
- 后续：先收齐严格覆盖和实时根统计、核验v21首个实际质量报告包含根门槛；旧v20四控制已完成，不再重复渲染。继续针对已知高攀爬碰撞和低左45原variant2支撑做隔离改进/专项求解，并以v21同门槛小批验证。不要仅等全库结束，所有2960原始槽位都要真正通过当前全部门槛。


14:15补充：

- v21全库已提交实际accepted，主独立index使用--allow-running --require-root-continuity成功核验3来源23槽位，segment_v21_first_committed.index.json；不是仅测试或计划。实际报告包含root-continuity-v1、根评分/哈希、每变体合取。cfg0021例9/10通过，根速度RMSE0.24381/P950.50755/max1.27784m/s，加速度max35.8823m/s²。
- 实时新口径首轮填充缓存完成，14:13:18为93/2960通过、缺2867；旧geometry+arms+torso口径2211。此数量变化是新增根检查剔除了大量旧跳变候选，不是原始分母改变。后续live_coverage.py可按source/candidate哈希复用缓存，只对新输出再计算，仍是provisional。全独立旧资产复筛PID2917198仍CPU100%运行、19父列表未变，必须继续等同一进程；不要重启。将来的该脚本需显式处理运行中v21，当前脚本assert父批终态。
- 已启动具体缺失攀爬专项php_segment_v21_climb_targeted_20261110，PID2935298，冻结仍segment_v21/frozen_runner，日志targeted.log，GPU1单worker/CPU1、24collision rounds（全库仍12）。两条1438完整帧：low-climb/left_45_walk1.5s_step1.2s用原solve_variant2并输出__terrain_02键；high-climb/run_right_90_0.6s_step0s用variant0。目标补11个缺失原槽位，每条仍验收全部原10变体，不能把20次检查计成20新增。47源码哈希验证并复制snapshot。没有修改优化损失/碰撞排除/验收门槛，增加轮数只是检验是否收敛不足；尚无质量结论。选择文件segment_v21_climb_targeted_selections.json。
- 全库选择全296源而不是等待旧复筛后删去已通过源，用同一v21导出门槛形成完整可追溯基线；已通过的v20小批仍可在最终strict索引中复用，新增求解不重复计覆盖。推广记录中的“等待严格清单”已被实际全库launch更新覆盖。


14:17覆盖复筛收齐（覆盖前述仍运行状态）：

- PID2917198/session23406已经正常结束，19父目录完整独立旧资产复筛结果为 **79/2960**、缺2881、7来源全变体通过；根门槛必需。证据strict_root_coverage_20261110_1406.json/.index.json。该列表在v21全库启动前固定，故不包含新全库后来提交的结果；不能把79误当作当前实时总数。
- 当前实时根口径（2026-11-10T14:16:09.324654+08:00）为 **153/2960**、缺2807；旧三门槛为2211。新全库实际提交结果补入后增加。实时缓存完成且正常更新，PID2925527已结束，不再需要等待首次填充。最终交付仍须更晚的独立全覆盖审计。
- 全库PID2923075与攀爬专项PID2935298继续运行。14:15:52全库8accepted/1checking/2solving/285pending，64本批槽位，求解估计79.4分钟/CPU工作量51.1分钟，阶段并行且样本类型变化会影响估计；对外保守本轮约2–3小时，不是全部成功承诺。专项GPU worker2935610已ready，1solving/1pending，暂无线下质量结果。


### 2026-11-10 14:27 攀爬专项收齐、地形小批和v22峰值试验

本轮有实质进展：低左45缺失原variant2补齐，定位高攀爬仅3帧碰撞且24轮无实质改善；启动4地形完整小批与隔离v22峰值试验。主目标全部2960仍未完成。

- v21攀爬专项php_segment_v21_climb_targeted_20261110已终态，PID2935298退出。两条1438帧、10/20检查通过、1accepted/1rejected/0failed；独立审计segment_v21_climb_targeted_report.json验证14输出哈希、源receipt和根连续性。主严格索引segment_v21_climb_targeted_committed.index.json已核验低左45完整10槽位。
- low-climb/left_45_walk1.5s_step1.2s以原solve_variant2求解、24碰撞轮，输出__terrain_02，全部10原变体通过。相对v20该源9/10新增1槽位，0丢失，不是新增10。solve_variant2 terrain1.80984mm/self1.43630mm，右膝支撑通过。segment_v21_climb_targeted_comparison.json记录同源/同完整帧对照。一个新完整轨迹覆盖该来源全部10变体，旧来源/编号保留。
- high-climb/run_right_90_0.6s_step0s在24轮后仍0/10，拒绝terrain/self；self7.02359→6.76384mm，terrain6.77787→6.79422mm，增加轮数没有可靠改进。源、门槛、碰撞排除未变。
- 原生模型对v20高攀爬逐帧碰撞诊断segment_v20_high_climb_collision_diagnosis.json（脚本.cache/php_conversion/segment_v21/diagnose_high_climb.py）显示>2mm共11条接触记录，集中202、204、213帧。213帧pelvis/hip_roll_r_link约7.02mm，右脚/world约6.78mm；202/204有手与地形接触。213源意图为右脚+左手，202/204为右脚+左膝+双手。因此不能全段抬高根部或删除手/膝接触来消除；自碰撞也不会因平移修好。诊断包含完整mesh沿若干轴的投影供复核，没有新增碰撞排除。
- 隔离v22基于冻结v21，仅native碰撞损失由100000*mean(逐帧最大violation)改为100000*(mean+0.02*amax)，强调少数最差帧。碰撞平面累积、源假零修复、自由臂接触切换渐变、根速度/加速度正则、全部独立门槛和12碰撞轮不变。这不是旧v15参数原样重试：当前已纠正假接触并具有根平滑和硬性根筛查，峰值权重也较小；结果仍必须实测，不能假定有效。
- v22的11项针对性回归通过segment_v22_gate_regression.txt（根/接触切换/根时间正则），CUDA图和固定前缀逐字节不变smoke通过segment_v22_graph_smoke.json；脚本额外断言两次均实际进入native碰撞refinement，确保峰值分支被执行。没有声称v22重新跑了主v21的65项全套测试。正式主源码仍v21，v22未推广。
- v22完整8条4871帧/80原槽位（四控制、三攀爬、长转弯cfg0062）已启动：php_segment_v22_collision_peak_pilot_20261110，PID2967165，GPU1单worker/CPU1，冻结.cache/php_conversion/segment_v22/frozen_runner，日志pilot.log；47源码哈希验证、snapshot复制。选择segment_v22_collision_peak_selections.json。14:25为1solving/7pending，尚无质量结论，预计10–15分钟。
- 同时启动v21地形小批php_segment_v21_terrain_pilot_20261110，PID2967164，GPU0单worker/CPU1、12碰撞轮，使用原冻结segment_v21/frozen_runner，日志.cache/php_conversion/terrain_retry_v21/pilot.log。4不同完整来源共4871帧，对应cfg0095 variant2、cfg0017 variant2、cfg0023 variant1、cfg0049 variant4；每条仍验收全部原10变体，目标新增4缺失槽位，不是40新增。计划从根筛查的旧79索引及当前v21已提交通过结果中扣除已有覆盖，绑定基线索引哈希及当前质量回执；segment_v21_terrain_pilot_plan.json和selections.json。47源码和snapshot已核验。14:25为1solving/3pending，尚无质量结果，预计10–15分钟。
- 两新批次启动记录segment_v21_terrain_v22_peak_launch.json。当前总GPUworkers=4（全库2+地形1+v22 1）、CPU验收workers=4；旧攀爬专项已结束，不是第四个常驻GPU批次。
- v21全库PID2923075持续运行，14:25:18为23accepted/2checking/2solving/269pending，200本批槽位；求解估计剩77.3分钟、CPU工作量51.5分钟，两阶段并行。当前已验收拒绝均仅terrain_penetration，没有因此放宽门槛。后续类别变化可能改变失败类型与ETA，仍需继续扫描新failed。
- 实时根口径（2026-11-10T14:25:28.339314+08:00）为270/2960、缺2690，旧三门槛2212仅作诊断。实时缓存绑定source/candidate哈希，仍非最终独立审计；最近完成的全旧资产独立快照79不含v21后来提交结果，不能混用时点。
- 下一轮核实三个活跃控制器及v22/地形结果；收齐后分别独立审计、同源质量与根指标对照。v22仅在四控制保持40/40且有实质新增/改善后才考虑推广，并生成完整新对照；失败则按具体帧继续改算法，不盲加轮数。地形小批有效后按最新根口径去重扩大缺失地形专项。全库持续运行，不能因观察超时重启，也不因部分样本成功结束全2960目标。


### 2026-11-10 14:46 地形扩展、前倾同源测量与v23隔离试验

本轮实际完成：v21地形小批和v22收齐并独立审计；新增5原槽位后启动37地形专项；否决抬升捷径；生成当前修正版与旧v3的完整前倾曲线；诊断残余碰撞后启动v23。主版本仍v21，目标2960全部通过尚未完成。

- 前倾测量 `posture_v3_v20_comparison.json` 通过原视频manifest哈希绑定四条完整源/目标，1397帧，50Hz。首秒原G1均2.9519°，旧v3天工分别13.5618/13.7516/14.9019/15.1579°，v20修正版约2.9558–2.9560°。全段前倾RMSE分别0.1628/0.0213/0.0159/0.0285°；这是一维诊断，不要与原独立三维躯干门槛混淆。原三维RMSE分别0.2072/0.0432/0.0298/0.0636°，仍全通过。图results/php_segment_v20_diagnostics_20261110/torso_lean_comparison.png已生成并查看；原G1与修正版曲线基本重合，保留源动作自身前倾。完整录像仍results/php_segment_v20_previews_20261110/comparison/all_comparisons.mp4，没有必要重复渲染。
- v21地形小批终态4accepted/0rejected/0failed，4871帧、29/40检查通过。独立审计segment_v21_terrain_pilot_report.json验证36输出哈希、源receipt、原地形/标签、根门槛。指定4个原变体全部通过，对规划时绑定的覆盖回执新增5原槽位（不是29新增），segment_v21_terrain_pilot_comparison.json。旧PID2967164已退出。
- 已启动php_segment_v21_terrain_expand_20261110，PID3014829，GPU0单worker/CPU1，仍用segment_v21/frozen_runner。37个缺失组合来自18完整源、40762帧、370检查，预期补37原槽位；按14:38左右已提交v21质量回执和旧严格索引去重，不能涵盖尚未完成的全库后续失败。计划segment_v21_terrain_expand_plan.json、selections.json、launch.json；脚本.cache/php_conversion/terrain_retry_v21/compare_expand.py。47源码哈希核验并复制source_snapshot，日志expand.log。14:44:36为2accepted/1solving/34pending，16检查通过，求解估计49.1分钟/CPU27.5分钟并行。每条仍完整时间和原全部地形，最终按目标去重统计。
- v22终态8源4871帧、69/80、7accepted/1rejected/0failed，独立审计segment_v22_collision_peak_report.json验证68输出哈希。与v20同8源相比新增0、丢失0，四控制仍40/40，高右90攀爬仍terrain/self：terrain6.789mm/self6.776mm。segment_v22_collision_peak_comparison.json明确不推广；旧PID2967165已退出。没有为了未推广的失败试验重复生成四录像。
- 根高度抬升可行性试验 `.cache/php_conversion/clearance_v21_probe/probe.py` 已完成4完整accepted seeds：cfg0011v1、cfg0013v6、cfg0021v6、cfg0029v4；固定关节/姿态/xy、最多8mm余量、原地形，用实际geometry+arms+posture+root评分。全部仍terrain失败，不计覆盖，没有正式plan/源receipt复验/标签/导出。证据segment_v21_clearance_probe.json。最坏原生接触分别为frame802左手48.507mm、frame580右脚28.047mm、frame121左手27.326mm、frame125右手33.014mm，法线z约0，均箱体侧面，解释垂直抬升无效。不要将这些诊断候选当作合格资产。
- `.cache/php_conversion/segment_v21/trace_witnesses.py` 已完成512帧高攀爬诊断、12轮，证据segment_v21_retained_witness_diagnosis.json。后期frame213 pelvis/hip_roll_r与world/ankle_roll_r同时约6.9–7.1mm；历史平面预测与新碰撞接近，没有证据指向陈旧平面是主要原因。该局部诊断没有覆盖完整源，不能计合格。原v20高攀爬髋部灵敏度segment_v20_high_climb_hip_sensitivity.json显示hip_pitch/roll不在关节限位，减小roll/增大pitch可降低该自碰撞，但须同时满足脚部支撑与整段平滑，不能直接单帧改角。
- 隔离v23位于.cache/php_conversion/segment_v23/frozen_runner，基于v21，仅将native碰撞损失从逐帧所有pair的最大值改为“每帧/无向body pair各自取历史平面最大值，再对pair求和/帧数”。重复历史平面不增大同pair权重，world不同surface仍同body pair取最大。固定100000系数；没有保留v22的0.02峰值，也未改变门槛、排除、contact、root、arm或torso。假设是让同帧多个碰撞都获得梯度，不声称已解决。
- 新48th runtime文件collision_objective.py仅在隔离runner，主源码未更新。新增3测试验证独立pair均有梯度、重复平面不放大、分离后无力；连同根/接触切换等共14测试通过segment_v23_gate_regression.txt。CUDA图、实际进入native refinement、固定前缀逐字节保留smoke通过segment_v23_graph_smoke.json。没有声称重跑v21完整65测试。
- v23完整小批8源4871帧/80原槽位（同四控制、三攀爬、cfg0062）于14:45:39启动：php_segment_v23_collision_pairs_pilot_20261110，PID3037823，GPU1单worker/CPU1、12轮；48源码哈希核验并复制snapshot。选择segment_v23_collision_pairs_selections.json、启动segment_v23_collision_pairs_launch.json；日志.cache/php_conversion/segment_v23/pilot.log。刚启动尚无结果，预计10–15分钟；不推广，待完整质量/根审计及同源比较后再决定，若推广仍先四完整新录像。
- 全库PID2923075持续运行。14:45:24为49accepted/4rejected/2solving/241pending；新拒绝4条均低攀爬右90，涉及terrain/self。已提交全部变体拒绝出现terrain73次/self30次/右膝support_distance1次/support_slip1次，原因可重叠；无根/躯干/自由臂拒绝，无程序failed。证据segment_v21_live_rejections_1446.json实际时间14:45:24，文件名仅标注近似分钟。14:44求解估计79.1分钟/CPU51.7分钟，类别变化使ETA上升。
- 实时严格去重14:44:26为483/2960、缺2477，33源全通过；是绑定源/目标哈希根筛查的provisional计数，最终仍须独立index/audit与--require-root-continuity --require-all。老三门槛数字不得代替。当前3个控制器、GPU4workers/CPU4workers，诊断进程已完成，均不重启。
- 下一步：收齐v23小批并独立审计，判断按pair计罚是否解决碰撞、是否引入root/arms/torso退化；主全库与37地形继续，收齐后按原source+variant去重，未覆盖的新失败另规划。若v23无新增通过，保留证据继续定位约束冲突，不盲加轮数或放宽门槛。所有2960仍需真正通过。


### 2026-11-10 15:34 v24正式修复、扩大结果与v25固定完整根部

上一目标轮和本轮均为实质进展：新算法、正式可追溯修复工具、全变体验收和完整录像已完成；新增合格资产，剩余问题继续修复。目标仍全部2960原始槽位，不标完成。

- **主源码现在php-segment-v24**，不是v21/v23/v25。新文件supermimic/acp/collision_objective.py、native_pose_polish.py、tools/polish_php_conversion.py、test_php_collision_pairs.py、test_php_native_polish.py已放入主工作区。php_conversion.py、convert_php.py、audit_php_conversion.py也更新。没有commit，其他旧修改未覆盖。50runtime源码哈希与冻结v24一致；promotion证据segment_v24_promotion.json。
- v23终态8源4871帧，69/80，7accepted/1rejected/0failed；独立审计segment_v23_collision_pairs_report.json通过68输出哈希。与v20相同通过数量，新增0/丢失0；但高攀爬右90自碰撞7.024mm→1.240mm，地形仍7.284mm，因此未单独推广。segment_v23_collision_pairs_comparison.json。其每帧/body pair内历史平面取max再对pair求和的native损失进入v24，系数100000，重复平面不放大，没有v22峰值项。
- v21全库已拒绝的6条原生全帧诊断segment_v21_rejected_native_diagnosis.json：每条碰撞仅1–5帧，包含手/膝/脚地形及跨肢自碰撞，原source intent保留。单帧可行性segment_v21_pose_feasibility.json和segment_v21_pose_fixed_rotation_feasibility.json证明高攀爬frame213可清除主要碰撞并保持支撑；仅诊断，不计覆盖。前者根旋转可变，后者根旋转固定，两者未达到优化器迭代收敛标记但实测几何已通过；不把单帧当整段合格。
- v24 native_pose_polish：先原生MuJoCo遍历全部帧，选现有非结构排除碰撞>2mm的帧；仅这些帧做SciPy least_squares，变量为root xyz+29关节，根四元数精确保留。每body pair的实际原生穿透残差目标1.5mm（不是修改验收阈值）；同时保持修复前支撑材料点、躯干朝向、自由臂躯干局部轨迹。root改动界限±0.1m，关节改动±0.4rad且在原joint limits内，max_nfev120；中心差分1e-5雅可比。只有实际最大穿透下降且有限才采用；未选帧精确不变。后续仍用原geometry/arms/torso/root全部门槛，优化器停止标记不能替代验收。
- v24四完整诊断种子（高右90、低左90_walk0.9_step0.7、低右90_walk1.5_step1.2、低forward_walk0.6_step1.2）原solve地形的全部门槛通过，segment_v21_native_polish_probe.json；只诊断不计覆盖。对应缓存.cache/php_conversion/native_polish_probe。root完全固定的后续诊断另述。
- 正式新CLI tools.polish_php_conversion 接受--seeds JSON列表 {parent,task_id}，--out、--workers、--scan/--resume。默认CPU2，不重复GPU IK。每任务绑定原父plan identity/sha、task、candidate/terrain/intent/quality哈希；只接受已提交accepted/rejected源、同模型/接触/摆臂/姿态策略，复核源motion/labels/terrain与G1模型文件。候选重新写干净motion字段及新version/task/seed_receipt/native_polish_policy；完整intent只换task_id，原始source receipt再由evaluate_task独立重算，全部原变体重新验收、重新标签/导出。每批运行前复制50源码snapshot。
- 新audit对于kind=native_pose_polish调用audit_seed_provenance：再次验证父资产/源资产，候选seed receipt和profile、原时间和root quaternion、未选帧、完整intent精确相等；随后常规audit再次检查全部质量、root独立评分、labels/terrain/export映射。v24只要求根四元数固定，位置允许受限修正；v25将要求整个根位姿固定，须兼容旧v24策略，不能用当前单个profile强行否定旧批次。
- v24正式小批php_segment_v24_native_polish_pilot_20261110已完成：12不同完整源8253帧/120原槽位，12accepted、119通过、0failed。前8种子来自v23小批，另4来自v21已拒绝低攀爬；原种子共69通过，新增50/丢失0。segment_v24_native_polish_comparison.json。独立审计segment_v24_native_polish_report.json验证108输出哈希、源receipt、root及新增seed provenance。旧PID3115781已退出。
- 该小批唯一缺失是低左45_walk1.5_step1.2的原variant2支撑（已由先前v21 targeted solve_variant2全10通过解决）。独立严格索引segment_v24_pilot_strict.index.json，父为v24小批+v21攀爬专项，**12原source、120/120槽位**。这不是全库120/2960总数。首次误用index --parents参数失败且无输出，pilot_index.log保留；改用位置参数后pilot_index_corrected.log成功。
- v24先66项测试通过（含新3native/3pair、root/arms/姿态/source/导出等），再5项原几何门槛安全测试通过，证据segment_v24_tests.txt与segment_v24_geometry_gate_tests.txt，共71分两次。首轮import ROOT位置错误导致test loader和scan在生成plan前失败；已改从robots导入，旧日志保留，正确测试和scan重跑通过。fresh GPU solve入口8帧+CPU polish hook smoke通过segment_v24_solve_integration_smoke.json：CUDA graph成功，根四元数精确保留；该8帧未触发需要修正的帧，实际修正分支由完整诊断/正式小批覆盖，不夸称smoke做了碰撞修复。
- 四条新版完整视频results/php_segment_v24_previews_20261110/comparison/all_comparisons.mp4，1397帧50fps1920x640、40/40原槽位，通过源/目标/quality/video哈希、完整时域/同地形及root独立重算。segment_v24_preview_summary.json、segment_v24_preview_manifest.json。4控制的native polish选帧均空，未改变种子动作数值。高右90完整568帧视频results/php_segment_v24_high_climb_preview_20261110/comparison/01_high_climb_comparison.mp4也验证，segment_v24_high_climb_preview.json；4.26s/frame213截图已查看，根/躯干/支撑保持，仍是运动学回放。
- **v24于15:15:17正式推广**，证据segment_v24_promotion.json绑定50代码和全部验证。现有v21全库/37地形冻结runner不修改、不重启。新完整GPU转换将自动调用native polish，已有资产可通过seed修复模式减少重复计算。
- 扩大php_segment_v24_native_polish_expand_20261110已收齐，11完整源8905帧/110槽位，7accepted/4rejected/0failed，**69通过，缺41**。源均原v21 solve地形仅terrain/self失败，规划时所有变体未覆盖；故新增69原槽位。独立审计segment_v24_native_polish_expand_report.json验证83输出哈希；比较segment_v24_native_polish_expand_comparison.json，保存每变体原因。旧PID3172541结束。此前口头把7accepted乘10误报70，已明确更正：实际6源10/10+1源9/10，不可再沿用70。
- 扩大遗留：right_90_walk0.9s_step0.7s缺variant6（left_knee support_distance/slip）。另外4源所有变体被root extra acceleration拒绝：forward_walk0.6s_step0.7s约171.55、left_90_walk0.6s_step0.7s135.19、right_45_walk0.6s_step1.2s182.36、right_45_walk1.5s_step0.7s108.61m/s²，阈值保持100。前两源某些非solve变体还存在手支撑滑动，不能称只剩root一个问题。修复几何不允许引入根瞬变。
- 新诊断.cache/php_conversion/native_joint_probe把变量减为29关节，root xyz和quaternion全帧精确不变。用**原始根已通过的种子**重新修复，绝不直接拿root已失败的v24输出锁住错误根。5完整来源（前述4诊断+forward_walk0.6s_step0.7s）都通过原solve地形的geometry/arms/torso/root，segment_v24_native_joint_probe.json；仍无正式plan/标签，不能计覆盖。未新增任何碰撞排除/放松门槛。
- 隔离v25冻结.cache/php_conversion/segment_v25/frozen_runner：基于v24，只将native polish改为固定完整根部，policy=native-joint-polish-fixed-root-v1，root_position_bound_m=0，其他目标/关节界限/门槛不变。audit_seed_provenance识别两种完整固定策略：旧v24根旋转固定与新v25完整根固定，候选policy必须对应declared已知策略；不能让新全根检查错误拒绝旧v24合法资产。segment_v25_root_guard.json证明伪造1mm根平移被拒绝、改policy被拒绝、12个旧v24 provenance仍正确通过。
- v25的71完整相关测试通过segment_v25_tests.txt；之后仅补了历史profile兼容审计，上述root_guard单独验证新审计分支和旧12资产，不声称71整套在该小改后又重跑。初始只scan的未启动目录已整体归档到.cache/php_conversion/segment_v25/initial_scan_before_compatibility_guard（只有plan和lock，无motion/manifest）；修正兼容代码后重新scan_final生成真实启动plan，代码身份无混用。
- **v25正式隔离小批运行中**：php_segment_v25_native_joint_pilot_20261110，PID3238894，2026-11-10T15:31:55启动，CPU2，不占GPU；16完整源11460帧/160原槽位，50源码hash及snapshot。种子是v24最初12个原种子+4个扩大root失败样本的原v21种子（均根本来通过），不是v24坏root输出。segment_v25_native_joint_seeds.json、launch.json，日志.cache/php_conversion/segment_v25/pilot.log。15:33已有2accepted/2checking/12pending，20检查通过但只是既有控制重复；未推广，主仍v24。按前完整批实测粗估8–12分钟，初期status脚本由两个短控制推算约32分钟受类别开销严重影响，等攀爬收齐后再更新。
- .cache/php_conversion/status_current.py和live_coverage.py已识别tools.polish_php_conversion进程；CPU-only求解与验收共用worker池，应相加估算，不能当GPU/CPU并行。旧GPU流水线仍阶段并行。
- 后续规划缓存.cache/php_conversion/segment_v24/plan_expand.py已改为**必须--tag新前缀**，拒绝覆盖已有规划证据，并读取/proc确认活跃native修复控制器、排除其未终态source/solve_variant，避免重复派发。原11来源规划在此修改前已经固定，文件未覆盖。下一批需新tag，例如segment_v24_native_polish_expand2；先收齐v25再选更合适profile。不要重跑旧命令覆盖原证据。
- 15:33:19活动：全库v21 PID2923075，89accepted/20rejected/2solving/185pending，834本批槽位，剩求解约72.4分钟/CPU42.7分钟并行；37地形PID3014829，29accepted/1solving/7pending，233检查通过，剩约12分钟/CPU6.5分钟。两个旧冻结仍运行，不能用main v24去resume它们。
- 实时严格去重15:32:14为997/2960、缺1963，94源全通过；仍是provisional源/候选hash绑定根评分，最终独立全覆盖审计未完成。目标要求全部2960，不将12源120/120或批次结束当完成。
- 下一轮：核验三实际控制器；收齐v25并独立源/资产/全部门槛审计，比较同16原种子的新增/退化。四完整控制录像必须生成并验证后才能推广v25。别忘两个非solve地形支撑遗留（低左45variant2已有旧修复、低右90_walk0.9_step0.7 variant6尚缺），以及root失败源其他变体手滑动。37地形结束后独立audit+规划回执去重比较。继续按新全库已提交失败规划后续修复，全部变体仍需达标；不用旧错误计数70或旧无root口径。


15:39补充：针对v24扩大中right_90_walk0.9s_step0.7s唯一缺失原variant6（左膝支撑距离/滑动），已规划完整原源fresh solve_variant6专项，并仍验收全部10变体。php_segment_v24_knee_terrain_20261110，PID3273562，GPU1单worker/CPU1，冻结仍v24、50代码hash与snapshot核验，日志.cache/php_conversion/segment_v24/knee.log；plan/selections/launch见segment_v24_knee_terrain_*。这是补1原槽位，不是新增10；不能改为重复variant0。正式v24已通过四控制/独立审计，使用其fresh GPU+native polish完整入口。当前共有4控制器：v21全库、v21地形37、v25CPU16、此v24专项；GPU4workers/CPU6workers。预计专项约3–5分钟，尚无结论；v25未推广。


15:42补充（覆盖v25仍求解中的旧状态）：v25小批已16accepted/0rejected/0failed，155/160检查通过；四个root加速度失败来源已恢复原合格根，剩5个原变体是接触门槛（其中低左45variant2已有v21专项覆盖）。主尚未推广v25：独立审计正在使用v25冻结runner执行，本轮audit session81597，日志.cache/php_conversion/segment_v25/audit.log，目标segment_v25_native_joint_report.json。四完整新视频渲染也正在运行，session10389，日志four_render.log，输出results/php_segment_v25_previews_20261110/comparison。渲染完成后执行.cache/php_conversion/segment_v25/verify_four_previews.py，并做同16原种子和已通过v24结果比较，才能考虑推广。15:39:45实时严格去重1109/2960、缺1851，尚非最终审计。v24膝variant6专项PID3273562刚启动，不能漏查。


### 2026-11-10 16:28 后台唯一执行者、v25推广与v26诊断

本轮与前轮为实质进展：v25正式推广、专项独立审计、新速度约束诊断通过。目标仍全部2960原变体；未完成、不标blocked。

- 已读取根AGENTS.md及/home/js/.local/share/php-guard/OPERATIONS.txt。本轮PHP_OVERNIGHT_OWNER=1，php-codex.service active，journal确认原thread 01a1bf00-c28d-7320-8999-8384558d17c1。服务前期initialize超时重启已恢复，不启动第二个Codex。标准acp环境tools.convert_php由sitecustomize自动注册guard，禁止改guard配置/限额/伪造MANAGED/绕过队列。已有全库guard job2b6364f43d92423090ee adopted PID2923075，GPU2 CPU2预约32GiB。旧37地形job8542a2219b3982976387已completed。共享96/104GiB及主机106/112GiB边界原样保留；取消须先guard cancel，不直接kill已接管控制器。
- 主源码已于15:48:58正式升v25，segment_v25_promotion.json绑定50源码与证据。仅php_conversion.py、native_pose_polish.py、polish_php_conversion.py三文件由核验过的冻结v25复制；此前主源码逐文件与v24一致，未覆盖其他修改。v25整根固定，旧v24只固定根旋转的已知profile兼容保留。独立审计16源11460帧155/160、144输出hash；同16来源较v24新增36丢失0、较原种子新增86丢失0，所有根导数指标逐值与原种子一致。证据native_joint_report/comparison、root_guard、tests、promotion。
- v25四完整录像results/php_segment_v25_previews_20261110/comparison/all_comparisons.mp4，1397帧50Hz1920x640、40/40，preview_summary/manifest独立核验。四条躯干三维RMSE0.20618/0.04304/0.03017/0.06463度。posture_v3_v25_comparison.json及results/php_segment_v25_diagnostics_20261110/torso_lean_comparison.png已生成且查看，原G1与新版曲线基本重合；这是完整同源FK诊断，不改验收、不表示动力学跟踪。
- v21地形扩大37已全部结束：18完整来源、多地形求解40762帧、291/370检查通过。独立segment_v21_terrain_expand_report.json核验333输出hash；expand_comparison.json对规划时绑定索引/质量回执去重，37目标全部通过、原槽位新增37，不把291当新增。新核验脚本.cache/php_conversion/terrain_retry_v21/verify_expand_final.py；旧compare_expand.py是生成规划脚本，不能重复执行覆盖旧规划。第一次核验被前台中断，无活动进程后重跑完成。
- v24膝variant6专项已完整770帧、10/10，独立segment_v24_knee_terrain_report.json及comparison.json验证基线9/10→10/10、新增1丢失0；原PID3273562结束，不重启。
- v25手地形专项php_segment_v25_hand_terrain_20261110于15:50启动、后已终态4accepted0failed，两个完整来源3170重复求解帧、16/40检查通过。四个指定原变体全部通过，segment_v25_hand_terrain_report/comparison验证新增4、剩0。forward_walk0.6_step0.7的v3仅该v3通过；left90_walk0.6_step0.7的v3/v4/v5各有[1,3,4,5,7]通过。与原v25小批互补覆盖两源20/20，不能拿单一地形轨迹替换全部变体。结合旧左45v2专项，原16源现有160/160；尚未生成本组合全索引。
- v25扩大php_segment_v25_native_joint_expand_20261110于15:49启动、已终态2accepted2rejected0failed，4完整源2654帧13/40。独立report28输出hash，comparison原缺40→新增13/余27。高right45_run1.2_step0.8通过9、v3手滑；低right90_walk0.6_step0.7通过[0,2,3,5]、余1/4/6/7/8/9右手滑（v1另支撑距离）。两条high left90_run1.0_step0和right90_run0.6_step0.8碰撞已清除，但joint_velocity比率1.02064/1.02593，全部20变体拒绝；原根/躯干/自由臂通过。不能放宽速度阈值。
- .cache/php_conversion/native_velocity_probe：在原根合格v21种子上，固定整根只调29关节，局部joint bounds与所有受影响np.gradient速度差分相交；内部用帧±2，端点严格单侧，按已更新out逐帧算，速度余量1e-7。两条完整1240帧诊断原solve地形全部geometry/arm/torso/root通过，segment_v25_native_velocity_probe.json。未改变fps/帧区间/地形/门槛；诊断不计覆盖。边界不可行保留原帧并记录，不伪造成功。
- 隔离v26冻结.cache/php_conversion/segment_v26/frozen_runner基于v25，整根+速度边界profile native-joint-polish-fixed-root-velocity-v1，兼容旧v24/v25 profile。20种子为v25最初16原种子+扩大4原v21种子（14114完整帧/200变体），segment_v26_native_velocity_seeds.json；未启动正式小批，不能宣称通过。主代码仍v25。
- 为CPU seed修复也使用受保护入口，仅隔离v26的tools.convert_php新增--polish-seeds分流，显式--workers-per-gpu 0 --cpu-workers N，调用polish main(argv)。guard真实预约0GPU+NCPU，无需改基础设施。仍标准python -m tools.convert_php，不直接运行未截获的polish controller。加入请求worker转发与不一致拒绝测试。
- 首次v26全套测试在前台中断后无终态/进程，不算通过；新3速度边界测试segment_v26_velocity_tests.txt通过。正确源码完整测试现运行session14165，输出segment_v26_final_tests.txt，须看终态。之前写测试相对路径因cwd重复失败，已修正；不能用旧截断日志宣称全套通过。
- 16:25:59全库191accepted/24rejected/81checking，无pending/solving，1810本批槽位，CPU验收估计16.4分钟。控制器2923075仍live且被guard接管；不重复启动。最近实时strict1784/2960是16:13:46旧快照，后续应更新，不与当前本批1810直接相加。最新运行详情以guard和/proc为准。
- 下一步：收齐v26完整测试，再通过标准受保护convert --polish-seeds入口scan/submit20源小批，明确CPU worker数与job identity；正式plan需50代码snapshot，独立audit/同20种子无退化比较/四完整新录像验证后才能推广。同时对已验收来源中尚缺的原地形编号做去重规划与专项，使用已推广v25或后续已验证版本；全库结束后独立全覆盖audit/index --require-root-continuity --require-all。所有2960目标未达成。


16:32补充：隔离v26完整76项测试已通过segment_v26_final_tests.txt（89.095秒），其中3速度边界与2统一入口测试；旧截断测试不计。标准受保护convert --polish-seeds scan20源14114帧/200槽位，16:30:07 guard job f5a4788c96313f7b4a2b正式running，CPU4 GPU0预约26GiB，实际child3488271，50源码/snapshot核验，native_velocity_launch.json。主仍v25。前4控制已accepted，四完整录像正在渲染session缓存日志four_render.log，后执行verify_four_previews.py。历史profile兼容/root伪造审计脚本verify_root_guard.py已启动，须收终态，证据segment_v26_root_guard.json。

v25新增地形扩大57求解（17完整来源、32586重复求解帧、570变体检查，目标补57原槽位）已通过guard job aa524df334eff026f014运行，实际child3493687，GPU2 CPU2预约32GiB。输出php_segment_v25_terrain_expand_20261110，冻结v25不变，50代码snapshot核验。segment_v25_terrain_expand_plan/selections/launch.json绑定当前root合格回执、旧严格索引，排除实时native修复来源；规划脚本.cache/php_conversion/segment_v25/plan_terrain_expand.py，不能重复覆盖旧规划。日志terrain_expand.log。三批合计预约90GiB，原96GiB共享上限不变；旧全库已无求解，但预约仍保守32GiB，不手改guard。

16:29:22严格实时去重2146/2960、缺814、202源全部通过，是provisional，尚非最终独立全库核验。16:31全库204accepted/24rejected/68checking，本批1940槽位，CPU验收余约12.8分钟。status_current.py已识别convert --polish-seeds的共享CPU池，不能除以0GPU；/proc识别只接受真实python -m控制器，避免把guard submit等待进程误当求解器。live_coverage.py亦修正同一识别，未改算法或guard。


16:33即时交接：v26 root_guard已完成，伪造根平移/篡改policy均拒绝，旧v24 12个和旧v25 16个完整seed provenance兼容验证通过，证据segment_v26_root_guard.json；无需重复跑。v26四完整render session69333仍在运行，日志four_render.log已01_straight完成，收齐后执行verify_four_previews.py，未渲染完成不能记录preview_verified。v25最初16源组合（v25pilot+v21climbtarget+v25handterrain）的独立严格index正在生成，session37701，日志.cache/php_conversion/segment_v25/pilot_index.log，输出segment_v25_pilot_strict.index.json；该组合应覆盖16源160原槽位，仍须看实际终态。16:32:54三个实际控制器均live，guard两新job均running（非queued）：v26 7accepted/4checking/9pending、69检查通过，共享CPU预计5.3分钟；v25terrain57为2solving/55pending；旧全库206accepted/24rejected/66checking、1960本批槽位，剩CPU12.2分钟。无需改guard或重启任何批次。全局最近strict2146是16:29:22快照，勿把批次数相加。接下来收齐v26后用其冻结auditor独立全20源audit，再与v25同20来源/原种子逐原variant比较，四录像通过后才推广；主仍v25。目标所有2960尚未完成，后台继续。


### 2026-11-10 16:45 v26正式推广、窗口衔接专项和后续支撑队列

本轮实质进展：v26完整小批/四录像/独立审计完成并推广；16源组合严格索引完成；针对新根部失败定位窗口衔接并启动不同配置的完整源专项。所有2960原槽位仍未完成，继续保持active，不标blocked。

- **主源码现在v26**，推广时间16:41:44，segment_v26_promotion.json绑定50runtime源码/76测试/全部证据。修改4runtime文件：php_conversion.py版本、native_pose_polish.py速度边界、polish_php_conversion.py三种已知profile兼容与argv入口、convert_php.py受保护--polish-seeds分流；新增tools/test_php_native_velocity.py、test_php_polish_entry.py。复制前主50文件与冻结v25完全一致，复制后与冻结v26完全一致，其他用户修改未覆盖，无commit，git diff --check通过。已有v21/v25冻结控制器不变，不用新主源码resume旧任务。
- v26正式20完整源14114帧/200原变体，20accepted0rejected0failed，188通过。segment_v26_native_velocity_report.json独立验证180输出hash、完整seed/source provenance及根连续性；comparison.json相比同20源v25（168）新增20丢失0，相比原种子新增119丢失0，所有根导数指标与原种子完全一致。两条高攀爬joint_velocity拒绝已恢复全部20；余12为接触支撑（其中5由旧专项已覆盖）。guard jobf5a4788c96313f7b4a2b completed退出0，旧child3488271已结束，不重启。
- 四完整v26录像results/php_segment_v26_previews_20261110/comparison/all_comparisons.mp4已完成1397帧50fps1920x640并独立核验40/40、源/目标/videohash、完整帧数及root。segment_v26_preview_summary/manifest.json；four_verify.log成功。完整独立audit session92527已终态verified，不重复跑。76测试和root_guard旧v24 12/旧v25 16兼容与篡改拒绝均完成，均在promotion绑定。
- v25最初16源组合（v25pilot+v21climbtarget+v25handterrain）的独立严格index已完成，segment_v25_pilot_strict.index.json确实16sources/160slots；pilot_index.log成功。只是这16源，不是全库总数。
- 新全部已提交拒绝统计segment_v21_live_rejections_1635.json（实际16:35:02）为210accepted/24rejected/62checking，原因可重叠：terrain255/self200、left_elbow amplitude10、joint_velocity10、root velocitymax10/accel30等。发现四个需额外处理来源，不可沿用此前“只有碰撞”。三root失败全部最坏accel在512帧：cfg0084（740帧，速度max2.5269/加速度107.96，joint速度1.1174）、lowforward_walk0.9_step1.2（789帧，103.92）、lowleft45_walk0.9_step1.2（818帧，101.26）。原window512/overlap48固定前缀的首未固定帧512正是衔接，冻结v25用window1024完整单窗重解，原全部源帧/地形/fps/门槛不变。
- 3完整单窗专项php_segment_v25_full_window_20261110，2347帧/30槽位；segment_v25_full_window_plan/selections/launch.json。guard job81b1bb8b4c3b9fc323f1，GPU1 CPU1预约17GiB，先queued，v26释放额度后自动running，child3541722，日志.cache/php_conversion/segment_v25/full_window.log。16:44首lowleft45已10/10，其余2checking，无求解剩余，预计约1–3分钟验收；须收终态独立audit+逐源比较后再计新增，不能提前宣称30全过。
- 唯一已提交自由臂失败highleft45_run1.0_step0s（605帧），left_elbow free phase[215,243)28帧，reference RMS0.0292915m/actual0.0384799m，ratio1.313686>1.3，corr0.9107、lag-2，其余门槛通过。只读原生FK诊断segment_v21_arm_phase_diagnosis.json记录每帧参考/实际、transition objective weights（边界7帧半正弦，首帧0.038），没有改候选/阈值。不应删掉短自由段或放宽幅度准入。可继续研究短自由阶段的相位均衡/幅度目标，或固定整根及其他关节、带速度与碰撞边界的自由臂局部修正；目前只是设计方向，未实现v27，不得声称修复。
- 16:44发现第五个质量例外：locomotion cfg0026_stand_2.0s_left_45_1.7s_right_90_1.2s_run_right_45_1.2s_stand_2.0s，共402帧，root velocityP95=1.0075457>1，max2.669742>2.5，RMS0.48724<0.5/acc92.407<100，worstaccframe305。这条不是分窗衔接，不能套512诊断；尚未规划修复。全部门槛仍保留，需要查源相对速度目标的峰值约束。
- 16:35尝试用旧geometry-only种子规划器生成segment_v26_native_velocity_expand，因当前没有符合且未在v26pilot中的新种子，assert seeds退出且未写计划/种子。这不是转换程序failed，不重试空计划。规划器.cache/php_conversion/segment_v24/plan_expand.py已识别convert --polish-seeds实际控制器并排除inflight。
- v26小批余12缺失，减去独立已覆盖的旧5，真正余7来自2源：highright45_run1.2_step0.8的v3，lowright90_walk0.6_step0.7的v1/4/6/7/8/9。对所有当前>=21root合格已提交回执去重，规划segment_v26_support_terrain_plan/selections/launch.json：7完整地形求解5089重复帧/70检查。已通过标准受保护入口提交，guard jobc67e3d2c554f207435d9，GPU1 CPU1预约17GiB，**当前queued**（现有预约81，增加17会超过96，正常等待）。输出php_segment_v26_support_terrain_20261110，冻结v26，50代码snapshot已核验，日志.cache/php_conversion/segment_v26/support.log。不要因无任务进展或没有childpid而重复启动；应看guard/journal。
- 57地形专项仍v25，guardaa524df334eff026f014 running child3493687，16:44:35为11accepted/2checking/2solving/42pending、73检查通过（非新增73），剩求解约27分钟/CPU20分钟并行。旧全库guard2b6364f43d92423090ee、child2923075为231accepted/25rejected/40checking，2205本批槽位，剩CPU6.5分钟。实际运行3批，预约32+32+17=81；新支撑17排队，不改上限。
- 最新strict实时快照16:43:13为2454/2960、缺506、234源全通过，缓存live_coverage_1642.json实际时点16:43。仍是provisional，不能代替最终独立全资产/全槽位audit --require-root-continuity --require-all；不把每批检查通过数相加。
- 下一步：收齐三条单窗及全库终态，分别独立audit并对30目标原编号比较；57地形和7支撑继续监控guard，完成后按规划回执/原source+variant去重。处理短自由臂幅度与新cfg0026根速度问题，不盲重复相同算法、不弱化门槛。主v26正式，v27尚未创建。所有原2960通过才完成。
