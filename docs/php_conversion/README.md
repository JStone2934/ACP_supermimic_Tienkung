# PHP → 天工数据转换

更新日期：2026-11-10（Asia/Shanghai）。本文档集服务于 PHP 数据转换的使用、开发与任务交接；当前目标机器人为天工 3.0 `tienkung3_29dof`。

转换以 PHP 的 G1 动作及配套箱体地形为输入，通过源机器人 FK、接触阶段提取和目标机器人任务空间优化，生成天工根部位姿与 29 个关节的动作，再使用目标模型进行独立质检和接触标注。两个机器人都有 29 个关节，不代表关节角可以直接复制。

**2026-11-10 16:45 当前状态：** 正式算法已于16:41推广为 **v26**：20条完整小批188/200通过，较v25新增20、无退化；76项测试、全批独立审计（180输出哈希）及四条完整录像1397帧40/40均通过。全部根位姿保持，局部修正增加关节速度边界。最新严格实时去重 **2454/2960**、缺 **506**（16:43:13，尚非最终独立全库审计）。原16源结合专项已独立索引验证160/160。当前全库剩40条验收约7–15分钟；57地形专项约剩27–40分钟；三条第512帧衔接失败改用完整单窗口，首条10/10，另两条验收中。另7个原支撑地形已入队，按原内存限制等待放行。唯一后台执行者php-codex.service，原目标与门槛不变，全部通过目标未完成。

**前倾问题已量化并修正四条样例：** 相同首秒G1躯干前倾2.9519°，旧v3天工13.56°–15.16°，已测量的v20修正版约2.956°；四条完整轨迹的前倾角RMSE为0.016°–0.163°。测量使用原生FK、中立姿态校准和同源朝向，不受录像相机影响。见[新旧曲线](../../results/php_segment_v20_diagnostics_20261110/torso_lean_comparison.png)、[完整同步录像](../../results/php_segment_v20_previews_20261110/comparison/all_comparisons.mp4)、[测量证据](evidence/posture_v3_v20_comparison.json)。该结论限已验证四条；全库仍须逐条验收。以下带时点内容为历史记录，最新运行详情见[任务交接](taskmemory.md)。

**21:32 更新：** 正式源码为 `php-segment-v18`：源接触假零复核、肢段方向初始化与躯干朝向跟踪已集成；53 项回归通过。四条完整动作共 1397 帧、40/40 原始地形槽位通过，见[新版同步录像](../../results/php_segment_v18_previews_20261109/comparison/all_comparisons.mp4)、[验收摘要](evidence/segment_v18_preview_summary.json)。四条躯干三维角度 RMSE 为 0.029°–0.109°；旧 v3 的额外前倾确实存在，详见下方同源测量。全库冻结 v11 继续运行，现有运行不会被源码更新改变。

实时去重暂计 **1510/2960**，尚缺 **1450**（统计时间 2026-11-09T21:30:17.742391+08:00；尚非最终独立全覆盖审计）。已完成 31 个指定地形重跑，31/31 目标通过，新增 31 个原始槽位；240/310 是重复检查次数，不能当作新增数量。[证据](evidence/segment_v11_terrain_retry_expand_comparison.json)。9 个历史程序异常均已独立恢复明确质量结论，其中新恢复的两条台阶共 20 槽位通过。

隔离 v19 接触切换试验正在运行 20 条完整动作；低攀爬直行首条已 10/10 通过并独立核验，[完整攀爬对照](../../results/php_segment_v19_climb_preview_20261109/comparison/01_low_climb_comparison.mp4)。但手部释放附近根部仍有 0.129 m/20 ms 的位移跳变，现有验收未覆盖根部动力学/平滑性；不能据此称攀爬质量完全解决。v19 尚未推广。另有 v18 的 27 个缺失地形专项在运行。全部 2960 槽位通过目标尚未完成。

**2026-11-10 13:50 恢复：** 上次中断后四个控制器已停止，现已核验冻结源码身份并从原检查点恢复。恢复前暂计1763/2960槽位通过、缺1197。隔离v20根部平滑试验已完成59项测试，正在跑8条完整动作；现有攀爬的根部跳变仍待解决，不能只凭既有quality_pass宣称动作质量完全通过。详见[最新交接记录](taskmemory.md)及[恢复命令](evidence/resume_20261110_1350.json)。以下带时点的条目均为历史记录。

**2026-11-10 13:59 质量进展：** v20首条完整攀爬10/10通过既有门槛，根部最大单帧位移从v19的12.9cm降至5.1cm，[完整同步录像](../../results/php_segment_v20_climb_preview_20261110/comparison/01_low_climb_comparison.mp4)。假膝接触台阶专项新增10/10通过。实时去重暂计1986/2960、缺974，但全库检查已证明部分旧合格行走/台阶也有根部跳变；该计数不等于最终可用量，必须补充独立连续性验收。v20小批和其他剩余验收继续运行，未推广、目标未完成。

**2026-11-10 14:11 正式v21：** 四条完整新对照1397帧、40/40原变体通过独立根部连续性检查，[录像](../../results/php_segment_v20_previews_20261110/comparison/all_comparisons.mp4)、[验证摘要](evidence/segment_v20_preview_summary.json)。65项回归通过，已启动包含根部准入的296条/2960槽位全库重跑，预计本轮约2–3小时。旧通过数正在按新准入复筛，不能沿用为最终数量；当前全部通过目标尚未完成。[正式推广记录](evidence/segment_v21_promotion.json)。

**14:15 计数口径更新：** 新增根部连续性筛查后，14:13实时暂计 **93/2960** 通过、缺 **2867**；旧三门槛2211不能直接用于最终交付。新v21全库已实际提交结果，首批独立索引核验23槽位，另有两条攀爬专项正在运行。完整旧资产复筛尚在运行，最终以包含根检查的独立覆盖审计为准。

最新v26对照：[四条完整动作](../../results/php_segment_v26_previews_20261110/comparison/all_comparisons.mp4)、[旧版与v25前倾曲线](../../results/php_segment_v25_diagnostics_20261110/torso_lean_comparison.png)、[此前失败的高速攀爬完整录像](../../results/php_segment_v24_high_climb_preview_20261110/comparison/01_high_climb_comparison.mp4)。这些是运动学参考，未宣称策略或动力学跟踪通过。

## 阅读入口

| 需要做什么 | 文档 | 内容 |
|---|---|---|
| 使用现有工具、重新转换小样本 | [usage.md](usage.md) | 环境、真实 CLI、输入输出、质检解释和复现命令 |
| 继续开发通用转换工具链 | [development.md](development.md) | 当前实现、缺口、开发顺序和验收条件 |
| 恢复任务上下文、追加记录 | [taskmemory.md](taskmemory.md) | 历史开发、转换、停止状态及记录模板 |
| 查找数据、模型、报告和权重 | [asset_list.md](asset_list.md) | 分类资产索引、7 个最终片段明细及来源 |

## 数据流

```text
G1 模型 + PHP motion/terrain + G1 接触标签
  → G1 FK / 双脚接触阶段提取
  → 天工 root + joints 任务空间重定向
  → motion + terrain + intent + manifest
  → 独立几何质检 / 可选自由根部 PD 诊断
  → 天工接触标签 / 运动学参考录像
  → 后续学习控制器训练与跟踪评估（单独开展）
```

目前标签和录像可用于诊断未通过严格验收的动作；它们的生成不表示数据已获得训练准入资格。正式数据集需要通过相应质量门槛并保留报告。

## 当前全量转换任务（2026-11-09）

**20:51 当前状态：** 实时去重673/2960通过、缺2287（20:50统计；最近独立覆盖快照为更早的533）。v17完成仍50/80，不推广。现已证明部分源接触标签的零距离是数值错误：[原G1分离证明](evidence/source_zero_contact_certificate.json)。修复在保留原文件的条件下重算源零标签，并由验收程序再次核验；53项测试及全变体准备预检通过，v18完整8条小批已启动，预计10–15分钟验证目标质量。全库v11约剩52分钟、31地形专项约30分钟求解，均并行；全部通过目标尚未完成。

**20:32 阶段记录：** 实时去重暂计555/2960通过、缺2405；最近独立[覆盖快照](evidence/segment_v11_committed_coverage_20261109_2028.json)已核验533，二者时点不同。v16完整小批[独立审计](evidence/segment_v16_constrained_pilot_report.json)为50/80，0新增；三攀爬仍碰撞拒绝，不推广。v17在52项分批测试及CUDA检查通过后启动同8条试验，预计10–20分钟；全库v11约剩71分钟求解，31地形专项约47分钟，均并行。7条历史程序异常均已独立复验成质量拒绝，全部通过目标仍未完成。

**20:29 阶段记录：** 已对运行中的已提交结果做独立[覆盖核验](evidence/segment_v11_committed_coverage_20261109_2028.json)：533/2960通过、缺2427，42条来源全部变体通过。此为核验快照下限，整体目标未完成。全库v11预计剩约74分钟求解；新增31个原始缺失地形槽位专项并行运行，约53分钟。v15已结束仍50/80，不推广；v16已验证躯干朝向和速度约束实现，但三条攀爬仍有碰撞拒绝，尚在收齐控制样本验收。当前6条历史程序异常均已在独立目录完成复验，转为明确质量拒绝。

**20:03 阶段记录：** 地形专项8个指定失败变体全部修复，较原v11补8槽位、扣除其他版本已有覆盖后全局新增7槽位；[独立审计](evidence/segment_v11_terrain_retry_pilot_report.json)、[逐编号对照](evidence/segment_v11_terrain_retry_comparison.json)。当前全库去重暂计449/2960通过、尚缺2511，全部目标未完成。全库v11仍运行，剩余求解约1.5–2小时；v15已启动8条攀爬/控制小批，验证残余碰撞的最差帧约束。三条本轮程序异常均已隔离复验完成，质量仍拒绝；最终距离修复通过[49项回归](evidence/segment_v13_distance_separation_tests.txt)及[真实穿透保留测试](evidence/distance_real_penetration_test.txt)。

**19:52 阶段记录：** v13肢段方向初始化显著改善攀爬，v14再加强姿态/碰撞修正后，三条攀爬的躯干门槛全部通过（RMSE约0.90°–1.01°、最大约3.63°）；但碰撞/接触仍拒绝，8条小批均为50/80，无新增通过槽位，生产求解器仍保持v11。见[v13审计](evidence/segment_v13_init_pilot_audit.json)、[v14报告](evidence/segment_v14_pose_polish_pilot_report.json)、[v14对照](evidence/segment_v14_pose_polish_comparison.json)。全库v11与8条地形专项重跑均在运行；19:56跨正式版本去重暂计435/2960，尚缺2525（实时统计，未替代最终覆盖审计），全库剩余求解约94分钟。距离查询新增零点见证/不一致符号见证过滤，最终[48项回归](evidence/segment_v13_distance_final_tests.txt)通过，未放宽几何或保真门槛。旧v4全量[独立审计](evidence/segment_v4_final_baseline_audit.json)已完成，仅作旧口径基线。

**19:24 阶段记录：** 全库v11控制器 PID2659104 仍在运行；19:23实测本批234个槽位通过，跨新版结果去重暂计275/2960（尚缺2685，实时统计未替代最终审计）。按当前吞吐估计求解还需约113分钟，验收并行；这不是全部通过的完成承诺。v12摆腿目标小批已结束，40/80低于v11的50/80，cfg0062自碰撞退化，三攀爬仍不通过，因此不推广；[独立报告](evidence/segment_v12_swing_pilot_report.json)、[同源对照](evidence/segment_v12_swing_comparison.json)。旧v4全库已结束，仅用作历史几何/摆臂基线。

**前倾直接对比：** 使用同一原始动作、原生FK及中立轴校准重新比较v3/v11。四条开头首秒：G1约2.95°，旧天工13.56°–15.16°，新版约2.97°；全段前倾RMSE由11.37°–16.20°降至0.058°–0.182°。这是前倾分量，区别于全三维姿态RMSE。见[逐帧曲线](../../results/php_segment_v11_previews_20261109/comparison/torso_lean_v3_v11.png)、[数据](evidence/posture_v3_v11_comparison.json)。已验证四条/40变体，不代表攀爬和全部数据解决。

**19:05 阶段状态：** 正式源码已更新到 `php-segment-v11`，按髋宽/肩宽适配原接触面落点。8条小批50/80通过，四条预览保持40/40，cfg0062从0/10变为10/10；65项回归及独立审计通过。[最新版四条同步录像](../../results/php_segment_v11_previews_20261109/comparison/all_comparisons.mp4)、[小批报告](evidence/segment_v11_width_pilot_report.json)、[覆盖审计](evidence/segment_v11_coverage.json)。当前合计98/2960通过，2862尚未通过新标准。19:03启动 `php_segment_v11_full_20261109` 全库重跑，完整覆盖296条/181236帧/2960槽位；复用8条身份完全相同的小批终态，其余288条求解，2GPU各1worker/2CPU验收，日志 `.cache/php_conversion/segment_v11/full.log`。按小批实测估计求解约3小时，攀爬仍有已知失败，批次结束不能视为全目标完成。

**18:53 阶段结论：** 正式 `php-segment-v9` 已将躯干三维姿态纳入独立验收。14条完整动作10476帧中，88/140原始槽位通过；四条预览仍为40/40，且包含新增姿态检查。65项回归通过，独立审计验证106个输出哈希；见 [扩大试验报告](evidence/segment_v9_pilot_report.json)、[四条正式新版对照](../../results/php_segment_v9_previews_20261109/comparison/all_comparisons.mp4)。[当时覆盖审计](evidence/segment_v9_v10_coverage.json)为88/2960通过；旧几何/摆臂计数不能补齐。v10峰值损失试验只有30/70且直行退化，不推广。全库首轮v4因worker退出而中断后，已从原检查点以较低并发恢复，仍仅作为诊断基线。

**新增躯干前倾问题与小批修正（18:23 核验）：** 四条旧预览首秒 G1 躯干前倾均约 2.95°，v3 天工约 13.56°–15.16°；v4 恢复摆臂后仍约 10.61°–10.88°。对应骨盆没有增加前倾，主要是腰部/躯干姿态偏差。测量使用两模型原生 FK、各自中立姿态校准及同一源朝向，排除了摄像机视角影响。见 [姿态诊断](evidence/preview_posture_diagnosis_v3_v4.json)。独立 `php-segment-v9-posture-probe` 在 v6 稳定小批求解器上增加源躯干/骨盆朝向目标；四条完整动作 40/40 仍通过原几何与摆臂门槛，输出独立审计通过，躯干前倾 RMSE 降至 0.08°–0.28°、绝对误差 P95 为 0.16°–0.41°。见 [修正前后指标](evidence/posture_v9_pilot_comparison.json)、[审计](evidence/posture_v9_pilot_audit.json)。骨盆动态偏差仍存在；现有自动准入门槛尚未包含躯干/骨盆姿态，因此 40/40 不等于整体姿态保真通过。试验未覆盖旧数据，也未推广至全库。

**v4 历史批次（已结束）：** 用户要求恢复自由摆臂、按肢段适配、增加手/肘轨迹/摆幅/时序验收，四条完整样本验证后推广，并要求全部 2960 个原始地形槽位通过。`php-segment-v4` 已取消硬固定，按各骨段长度重建参考；四条完整动作、1397 帧、40/40 变体同时通过原几何门槛和新增摆臂门槛，独立审计通过。见 [小批报告](evidence/segment_v4_preview_a_report.json)、[指标摘要](evidence/segment_v4_preview_summary.json)及 [新版同步对照](../../results/php_segment_v4_previews_20261109/comparison/all_comparisons.mp4)。全库首轮 `php_segment_v4_full_20261109` 于 17:13（Asia/Shanghai）启动，已结束；已发现部分非求解地形的脚部穿透，正在用独立地形求解验证修复。**目标尚未完成，四条样本通过不代表全库通过。**

**v3 动作还原诊断（历史批次仍保留）：** 旧四条预览中，天工双臂的 14 个关节全程恒定，原版 G1 则都有变化。v3 求解器在无手/肘支撑时选择 `neutral/clearance`，每次更新后强制重置手臂；旧质检没有自由摆臂的跟踪误差门槛。因此下述 v2/v3 “合格”仅指几何/接触检查，不能当作全身动作还原合格。见 [手臂诊断](evidence/preview_arm_diagnosis_v3.json)。旧数据没有被覆盖，也不计入新增保真门槛下的合格覆盖。

用户已授权代码接线及全量转换。当前工作区为 `ACP_supermimic_Tienkung`、分支 `feat/dual-5090-scheduling`，实际设备为两张 RTX 5090（每卡约 32 GB）。新入口为 `python -m tools.convert_php`，原 pilot 继续保留用于回归。

已盘点 296 条源动作、181236 帧、2960 个地形槽位；逐文件去重后有 2510 个不同的动作/地形组合，所有原始变体编号仍保留。见 [输入清单](evidence/source_inventory.json)。

`php-fullbody-v2` 已完成代表批次 `representative_v7`：8/10 片段、78/100 变体合格，高台阶 10/10 变体通过；攀爬穿透和自碰撞大幅下降，但仍有接触质量拒绝。见 [前后对照](evidence/representative_v7_comparison.json)、[审计](evidence/representative_v7_audit.json)及 [39 项测试](evidence/tests_v7.txt)。

全量批次 `data/tienkung3_29dof/motions/php_full_v2_20261109` 已于 15:09 结束：111 条有合格变体、184 条质量拒绝、1 条验收异常，合格变体 910 个。原失败候选经近接触距离重试修复后，在独立目录完成 10 个变体复验与标签生成，结果为质量拒绝；原批次保持不变。见 [异常复验](evidence/distance_recheck.json)和 [296 条完整拒绝统计](evidence/full_v2_rejection_summary.json)。

后续 `php-fullbody-v3` 小批从 30/120 提高到 96/120 合格变体，6 条长轨迹从 25/60 提高到 52/60，均无原合格变体丢失，独立审计与 43 项回归通过。攀爬样本仍失败且部分指标退化，不推广到该类别。新批次 `data/tienkung3_29dof/motions/php_repair_v3_20261109` 已完成 146 条行走/台阶重跑：133 条有合格变体、13 条质量拒绝、0 条程序异常；合格变体从 210 提高到 1093，新增 904、退化 21。保持旧版合格结果优先的索引已核验：全库两版合计覆盖 **205 条动作、1814 个合格变体**，原有 910 个全部保留。索引是可追溯资产映射，不是直接可加载的合并训练数据目录。见 [重跑对照](evidence/repair_v3_rerun_comparison.json)、[独立审计](evidence/repair_v3_rerun_audit.json)及 [覆盖核验](evidence/repair_v3_coverage.json)。见 [短轨迹对照](evidence/repair_v3_pilot_comparison.json)、[长轨迹对照](evidence/repair_v3_long_comparison.json)。

## 历史状态（2026-11-07）

以下实验结果来自 2026-11-07 的试验；2026-11-08 仅核对文件并整理文档。

| 项目 | 状态与证据 |
|---|---|
| 天工模型、接口、FK 与接触计算 | M0–M3 接入及数值检查已通过，见[接入进度](../tienkung3_29dof/progress.md) |
| PHP 小样本转换 | 站立 1、直行 2、转向 2、低台阶 2；共 7 个片段、1220 帧、50 Hz，逐片段有效时长合计 24.26 s |
| 几何初筛 | 7/7，通过时排除了两对原生模型小腿—脚部凸包重叠 |
| 严格几何验收 | 0/7，小腿—脚部原始重叠约 34–36 mm 尚未解决 |
| 自由根部 PD 跟踪 | 0/7 达标；原始 G1 的同协议基线也失败，不能仅由 PD 失败断定学习控制器无法跟踪 |
| 接触标签、参考录像 | 已生成；录像是运动学回放，不是训练策略效果 |
| 单站立短训练 | 日志到 250 轮，持久化 checkpoint 为 200 轮；用户停止，未评估该 checkpoint |
| 五片段平地训练 | 日志到 100 轮后 CUDA OOM，无 checkpoint |
| 全量转换与复杂接触 | 未开始 296 条全量转换；手支撑、攀爬和其余地形变体未完成验证 |

质量数字见[最终质量报告](../tienkung3_29dof/php_pilot/evidence/quality_final.json)，实验状态见[收尾 manifest](../tienkung3_29dof/php_pilot/evidence/run_manifest.json)。当前全体样本为开发集，没有独立泛化评估结论。

## 文档维护约定

- 11 月 8 日的新任务已明确授权全量转换；先验证代表样本，再批量处理并记录所有成功、拒绝与失败。历史训练没有恢复。
- `usage.md` 只把现有接口列为可执行示例；未实现能力写入 `development.md`，并标注待开发。
- 每轮转换在 `taskmemory.md` 追加记录，更新 `asset_list.md` 中的用途、状态和证据链接。模型、数据或参数改变后建立新输出版本。
- 既有[完整迁移计划](../tienkung3_29dof/migration_plan.md)、[PHP pilot 设计](../tienkung3_29dof/php_pilot/design.md)和[收尾记录](../tienkung3_29dof/php_pilot/README.md)保留为历史来源；不回写历史实验结果来代表新结果。
- 数据、模型、权重和视频主要保存在 Git 忽略的本地目录；复制代码仓库不等于复制这些资产，详见资产清单。
