# PHP 转换资产清单

核对日期：2026-11-09（Asia/Shanghai）。返回[文档入口](README.md)。路径均相对仓库根目录；“存在”描述本机核对时的状态，不保证其他机器已同步。

## v3 修复资产

| 资产 | 用途与状态 |
|---|---|
| [完整拒绝统计](evidence/full_v2_rejection_summary.json) | 含独立异常复验的 296 条、2960 个原始变体槽位，按动作/类别/原因汇总；原因可重叠 |
| [距离异常回归姿态](evidence/distance_near_touch.json)、[复验结果](evidence/distance_recheck.json) | 原失败候选保持不变，10 个变体验收和标签生成完成，结果为质量拒绝 |
| `data/tienkung3_29dof/motions/php_full_v2_20261109_distance_recheck/` | assessment-only 产物、候选哈希、父任务和代码溯源；不能传给 `convert_php --resume` |
| `.cache/php_conversion/repair_v3/pilot/` | 12 条完整动作，30→96 个合格变体；[对照](evidence/repair_v3_pilot_comparison.json)、[审计](evidence/repair_v3_pilot_audit.json) |
| `.cache/php_conversion/repair_v3/long_pilot/` | 6 条长动作，25→52 个合格变体；[对照](evidence/repair_v3_long_comparison.json)、[审计](evidence/repair_v3_long_audit.json) |
| [43 项回归记录](evidence/tests_v3_repair.txt) | 包含真实近接触异常姿态、机器人契约、转换、pilot、流水线及交付检查 |
| `data/tienkung3_29dof/motions/php_repair_v3_20261109/` | 已完成 146 条重跑，133 accepted、13 rejected、0 failed；[审计](evidence/repair_v3_rerun_audit.json)、[对照](evidence/repair_v3_rerun_comparison.json)；源码冻结在 source_snapshot |

“最终输出”表示 2026-11-07 收尾所选定的数据版本，**不表示通过严格几何或策略跟踪验收**。大文件保存在 Git 忽略目录中，新机器需要按[使用手册](usage.md)下载、重建或另行同步。

## 1. 工具与文档

### 2026-11-08 新增资产

| 路径 | 用途/状态 | 来源与证据 |
|---|---|---|
| `supermimic/acp/php_conversion.py` | 通用九类接触求解与窗口恢复 | 当前工作分支；原 pilot 未替换 |
| `tools/convert_php.py`、`tools/check_php_conversion.py` | 双 GPU/CPU 队列、逐变体验收与正式子集导出 | plan/task/quality manifest 绑定身份与产物哈希 |
| `tools/test_php_conversion.py` | FK 原生对照、侧面投影、穿透/滑动/超速/假手支撑拒绝及过期产物检查 | 与旧 pilot、机器人契约、流水线共 36 项回归通过；日志暂在 `.cache/php_conversion/tests_v1.log` |
| [source_inventory.json](evidence/source_inventory.json) | 全部 296 源动作、地形与源标签路径和 SHA-256 | 181236 帧；2960 槽位；2510 不同动作/地形组合 |
| [collision_policy.json](evidence/collision_policy.json) | 踝部模型层级、4 组代表姿态和原始重叠 | 保留历史模型身份，仅新增有版本的几何验收排除规则 |
| `.cache/php_conversion/smoke/` | 100 帧站立算法检查 | 1000 次迭代约 3.8 s；有结构排除时通过，原始碰撞仍失败 |
| `.cache/php_conversion/representative_v1/` | 首轮批量接线失败证据 | numpy int64 接触段索引无法 JSON 序列化；候选/窗口保留，未计为验收成功 |
| `.cache/php_conversion/representative_v2/` | 修复序列化后的十样本双卡验证 | 历史验证批次；已由 representative_v7 代表测试取代 |

上述缓存中的候选、标签、窗口与日志均为本地资产，不进入 Git。小型配置、盘点和验收证据进入 Git；完整全量输出完成后在此追加实际路径与覆盖数。

| 资产 | 用途/生成入口 | 版本与证据 | 本地 / Git |
|---|---|---|---|
| [下载工具](../../tools/download_acp_data.sh)、[G1 MJCF 转换](../../tools/urdf_to_mjcf.py) | 下载 PHP/G1，转换 G1 模型并生成源标签 | 下载器固定 PHP 和 G1 上游版本，见下表 | 存在 / 已跟踪 |
| [天工资产工具](../../tools/prepare_tienkung3.py)、[RobotSpec](../../supermimic/acp/robots.py) | 冻结模型、派生资产及身份参数 | M0–M3 实现 `ca74a80`；资产 manifest | 存在 / 已跟踪 |
| [重定向器](../../tools/retarget_php_pilot.py) | 7 片段 root/joints 优化 | `php-pilot-taskspace-v2`，pilot 实现 `c79c31e` | 存在 / 已跟踪 |
| [质检器](../../tools/check_php_pilot.py) | 独立几何、PD 和筛选导出 | 源码哈希见 [pilot 运行清单](../tienkung3_29dof/php_pilot/evidence/run_manifest.json) | 存在 / 已跟踪 |
| [标签生成](../../supermimic/acp/make_labels.py)、[动作微调](../../supermimic/acp/refine_motion.py) | 目标模型标注及已有目标动作微调 | 身份/标签依赖见 [usage](usage.md)；微调还未维护 pilot intent | 存在 / 已跟踪 |
| [参考打包](../../tools/record_motion_reference.py)、[渲染](../../supermimic/acp/render.py) | 运动学或已有轨迹视频 | 源码哈希见 pilot 运行清单 | 存在 / 已跟踪 |
| [pilot 测试](../../tools/test_php_pilot.py)、[机器人契约测试](../../tools/test_robot_contracts.py)、[流水线测试](../../tools/test_acp_pipeline.py) | 来源筛选、错误拒绝及兼容回归 | [历史 20 项结果](../tienkung3_29dof/php_pilot/evidence/tests.txt)；本轮未重跑 | 存在 / 已跟踪 |
| [本目录](README.md) | 当前使用、开发、记录与资产入口 | 首次整理 2026-11-08 | 存在 / 本轮新增文档 |

## 2. 源数据与机器人模型

| 资产路径 | 用途与数量 | 来源 / 生成方式 / 哈希依据 | 本地 / Git |
|---|---|---|---|
| [data/php_parkour/](../../data/php_parkour) | 296 个 G1 motion、296 个配套 terrain | `tools/download_acp_data.sh`；PHP 提交 `c7c9ce3fc75a8754a7e403c58de3b6933254d01b`；[本地 release-assets 清单](../../.cache/setup/php/release-assets.json)记录上游文件校验信息 | 存在 / `data/` 忽略 |
| [data/php_parkour_contacts/](../../data/php_parkour_contacts) | 296 份源标签及 summary，供 pilot 提取接触意图 | 下载脚本末尾调用 G1 `make_labels`；pilot 用到的源标签哈希见[来源清单](../tienkung3_29dof/php_pilot/evidence/selection_and_retargeting.json) | 存在 / `data/` 忽略 |
| [data/g1_description/](../../data/g1_description) | 源 URDF、meshes、`g1_29dof.xml` | 下载工具固定 `lvhaidong/LAFAN1_Retargeting_Dataset@ce1572906efe6157840e8474d5a0d7aa87481e74`；[download-manifest.json](../../data/g1_description/download-manifest.json)；MJCF 由 `tools/urdf_to_mjcf.py` 生成 | 存在 / `data/` 忽略 |
| [天工 upstream](../../data/tienkung3_29dof/upstream) | 原始归档、天工 3 URDF/网格及控制参数引用 | `Open-X-Humanoid/TienKung_URDF@4c2c061d72dcdab60d74e45d3efb17246a3b6f62`；`prepare_tienkung3` 下载 | 存在 / `data/` 忽略 |
| [天工 generated](../../data/tienkung3_29dof/generated) | `robot.urdf`、`robot.xml`、OBJ meshes；固定头部后的 29 DoF 模型 | `prepare_tienkung3`；[资产 manifest](../../data/tienkung3_29dof/manifests/assets.json)包含原始/派生文件哈希 | 存在 / `data/` 忽略 |
| [天工 manifests](../../data/tienkung3_29dof/manifests) | `assets.json` 与 `robot.json`，资产、控制与身份配置 | [资产证据副本](../tienkung3_29dof/evidence/assets.json)、[机器人证据副本](../tienkung3_29dof/evidence/robot.json)进入 Git；本地清单用于实际加载 | 存在 / 本地忽略、证据副本已跟踪 |

PHP 分类数为：locomotion 100、low-step 56、high-step 60、low-climb-76 40、high-climb-76 40。此次未逐项复制全部源文件和网格的哈希，以既有清单为依据。

天工模型头部 yaw/pitch 固定在零位；裸模型 `nq=36, nv=35, nu=0`，训练实体另添加 29 个执行器。历史 pilot 身份为：

```text
robot_id:     tienkung3_29dof
model_hash:   b6105ab23a8494624068ab304c73e51d79a10b6504768730b741e42bf935ef3a
config_hash:  7edd0695f24b307df7a282729428d99b7da33f8a471ea1dd16206422070eb25d
data_version: php-pilot-taskspace-v2
control_dt:   0.02 s
```

身份来源为[运行清单](../tienkung3_29dof/php_pilot/evidence/run_manifest.json)。它用于识别历史版本，不应手工写入新数据来绕过模型检查。资产/几何或控制配置变更后重新计算身份、记录新版本。

## 3. 输出数据与历史中间版本

以下均在本地存在且被 `data/` 忽略。各版本的质量与用途以相应报告为准，不能仅根据目录名选入训练。

| 路径 | 用途与状态 | 生成入口 / 版本 / 证据 |
|---|---|---|
| [motions/php_pilot_final/](../../data/tienkung3_29dof/motions/php_pilot_final) | 收尾最终 7 片段；7 motion、7 terrain、7 intent 及 manifest；严格几何未通过 | pilot v2 各轮结果收尾汇总；[本地 manifest](../../data/tienkung3_29dof/motions/php_pilot_final/manifest.json)、[Git 来源清单](../tienkung3_29dof/php_pilot/evidence/selection_and_retargeting.json) |
| [motions/php_pilot_final_contacts/](../../data/tienkung3_29dof/motions/php_pilot_final_contacts) | 最终 7 份目标标签及 summary；用于诊断，不代表已通过准入 | 目标模型 `make_labels`；[标签统计证据](../tienkung3_29dof/php_pilot/evidence/final_labels.json)，各 NPZ 内含 motion/terrain 哈希 |
| [motions/php_pilot_v1/](../../data/tienkung3_29dof/motions/php_pilot_v1)、[php_pilot_v1_remaining/](../../data/tienkung3_29dof/motions/php_pilot_v1_remaining) | 早期失败/中间结果，保留用于诊断 | [quality_v1](../tienkung3_29dof/php_pilot/evidence/quality_v1.json)、[remaining](../tienkung3_29dof/php_pilot/evidence/quality_v1_remaining.json) |
| [motions/php_pilot_v2/](../../data/tienkung3_29dof/motions/php_pilot_v2) | v2 平地数据；当前 pilot 测试的站立夹具依赖此处 | `retarget_php_pilot`；各片段 v2 报告在 pilot evidence 中；最终清单记录选用版本 |
| [motions/php_pilot_steps_v2/](../../data/tienkung3_29dof/motions/php_pilot_steps_v2)、[php_pilot_steps_clearance_v2/](../../data/tienkung3_29dof/motions/php_pilot_steps_clearance_v2) | 台阶中间版本及外展手臂修正版；final 采用后者对应配置 | [step_a_v2](../tienkung3_29dof/php_pilot/evidence/step_a_v2.json)、[step_b_v2](../tienkung3_29dof/php_pilot/evidence/step_b_v2.json)、[clearance_a](../tienkung3_29dof/php_pilot/evidence/step_clearance_a.json)、[clearance_b](../tienkung3_29dof/php_pilot/evidence/step_clearance_b.json) |
| [motions/php_pilot_stand_v2/](../../data/tienkung3_29dof/motions/php_pilot_stand_v2)、[对应标签](../../data/tienkung3_29dof/motions/php_pilot_stand_v2_contacts) | 探索性单站立训练输入副本，1 片段；不能与最终根目录的数据身份混用 | [standing_training_gate](../tienkung3_29dof/php_pilot/evidence/standing_training_gate.json)、[standing_args](../tienkung3_29dof/php_pilot/evidence/standing_args.json)；副本根目录无独立 manifest |
| [motions/php_pilot_flat_train_v2/](../../data/tienkung3_29dof/motions/php_pilot_flat_train_v2)、[对应标签](../../data/tienkung3_29dof/motions/php_pilot_flat_train_v2_contacts) | 探索性五片段平地训练输入副本 | [flat_training_gate](../tienkung3_29dof/php_pilot/evidence/flat_training_gate.json)、[flat_args](../tienkung3_29dof/php_pilot/evidence/flat_args.json)；副本根目录无独立 manifest |
| [source_php_pilot/](../../data/tienkung3_29dof/source_php_pilot) | 同片段 G1 基线数据，7 片段；不是天工转换输出 | [源 G1 PD 报告](../tienkung3_29dof/php_pilot/evidence/source_g1_pd.json)，源文件/帧区间见来源清单；当前无单独 CLI 重建该裁剪副本 |
| [motions/bootstrap/](../../data/tienkung3_29dof/motions/bootstrap)、[contacts/bootstrap/](../../data/tienkung3_29dof/contacts/bootstrap) | M0–M3 合成站立链路，非 PHP 重定向 | `tools.generate_standing_motion` + `make_labels`；[合成数据证据](../tienkung3_29dof/evidence/synthetic_data.json) |

### 七个最终片段

下表片段名链接到最终 motion，“标签”链接到对应目标标签。所有片段均为地形变体 **0**、50 Hz；帧区间为零基 `[start,end)`，时长为 `(T-1)/50`。

| 最终片段 / motion | G1 来源 | 源帧区间 | 帧数 | 有效时长 | 手臂模式 / 迭代数 | 目标标签 |
|---|---|---|---:|---:|---|---|
| [standing/stand](../../data/tienkung3_29dof/motions/php_pilot_final/standing/stand_motion.npz) | [cfg0000][source0] | [0,100) | 100 | 1.98 s | neutral / 1000 | [标签](../../data/tienkung3_29dof/motions/php_pilot_final_contacts/standing/stand.npz) |
| [walking/forward_a](../../data/tienkung3_29dof/motions/php_pilot_final/walking/forward_a_motion.npz) | [cfg0002][source2] | [100,240) | 140 | 2.78 s | neutral / 1000 | [标签](../../data/tienkung3_29dof/motions/php_pilot_final_contacts/walking/forward_a.npz) |
| [walking/forward_b](../../data/tienkung3_29dof/motions/php_pilot_final/walking/forward_b_motion.npz) | [cfg0014][source14] | [100,160) | 60 | 1.18 s | neutral / 1000 | [标签](../../data/tienkung3_29dof/motions/php_pilot_final_contacts/walking/forward_b.npz) |
| [turning/left45](../../data/tienkung3_29dof/motions/php_pilot_final/turning/left45_motion.npz) | [cfg0000][source0] | [100,170) | 70 | 1.38 s | neutral / 1000 | [标签](../../data/tienkung3_29dof/motions/php_pilot_final_contacts/turning/left45.npz) |
| [turning/right45](../../data/tienkung3_29dof/motions/php_pilot_final/turning/right45_motion.npz) | [cfg0003][source3] | [100,200) | 100 | 1.98 s | neutral / 1000 | [标签](../../data/tienkung3_29dof/motions/php_pilot_final_contacts/turning/right45.npz) |
| [low_step/forward_a](../../data/tienkung3_29dof/motions/php_pilot_final/low_step/forward_a_motion.npz) | [forward_walk0.5s_step0s][step05] | [0,363) | 363 | 7.24 s | clearance / 1200 | [标签](../../data/tienkung3_29dof/motions/php_pilot_final_contacts/low_step/forward_a.npz) |
| [low_step/forward_b](../../data/tienkung3_29dof/motions/php_pilot_final/low_step/forward_b_motion.npz) | [forward_walk0.8s_step0s][step08] | [0,387) | 387 | 7.72 s | clearance / 1200 | [标签](../../data/tienkung3_29dof/motions/php_pilot_final_contacts/low_step/forward_b.npz) |
| 合计 | 7 个派生片段，来自 6 个源文件 | — | 1220 | 24.26 s | — | 7 份 |

[source0]: ../../data/php_parkour/motions-locomotion/cfg0000_stand_2.0s_left_45_1.4s_forward_2.3s_right_90_0.9s_right_45_0.6s_right_45_2.0s_left_90_0.6s_left_45_2.6s_run_right_45_1.2s_left_45_1.3s_run_right_90_1.6s_stand_2.0s_motion.npz
[source2]: ../../data/php_parkour/motions-locomotion/cfg0002_stand_2.0s_forward_1.2s_forward_1.6s_left_45_1.0s_right_45_2.0s_stand_2.0s_motion.npz
[source14]: ../../data/php_parkour/motions-locomotion/cfg0014_stand_2.0s_forward_1.3s_right_45_0.6s_forward_2.2s_stand_2.0s_motion.npz
[source3]: ../../data/php_parkour/motions-locomotion/cfg0003_stand_2.0s_right_45_2.0s_run_right_45_1.2s_run_right_90_2.9s_forward_1.3s_stand_2.0s_motion.npz
[step05]: ../../data/php_parkour/motions-low-step/forward_walk0.5s_step0s_motion.npz
[step08]: ../../data/php_parkour/motions-low-step/forward_walk0.8s_step0s_motion.npz

每个 motion 同目录还有 `<片段>_terrain.npy` 与 `<片段>_intent.npz`；来源链接所指文件同目录也有相应原始 terrain。所有源路径、源标签哈希、帧区间及输出 motion/terrain/intent 哈希见[selection_and_retargeting.json](../tienkung3_29dof/php_pilot/evidence/selection_and_retargeting.json)。

各片段质量结果位于[quality_final.json](../tienkung3_29dof/php_pilot/evidence/quality_final.json)的 `clips["类别/片段"]`：7 个均为 `pilot_screen_pass=true`、`strict_geometry_pass=false`、`pd.tracking_pass=false`。所有最终 motion/terrain/intent 文件与清单哈希在本次核对中一致；源 motion 哈希也一致。

## 4. 报告、轨迹、录像与权重

| 资产 | 内容/状态 | 来源与哈希依据 | 本地 / Git |
|---|---|---|---|
| [pilot evidence/](../tienkung3_29dof/php_pilot/evidence) | 来源、最终/中间质量、标签统计、训练日志/参数/准入记录、测试结果 | [run_manifest.json](../tienkung3_29dof/php_pilot/evidence/run_manifest.json)记录源码、报告、checkpoint 和视频哈希 | 存在 / 已跟踪 |
| [M0–M3 evidence/](../tienkung3_29dof/evidence) | 资产、FK/接触数值、兼容检查、合成训练与评估 | [接入运行清单](../tienkung3_29dof/evidence/run_manifest.json) | 存在 / 已跟踪 |
| [reference.npz](../../results/php_pilot_final/reference.npz)、[reference.mp4](../../results/php_pilot_final/reference.mp4) | 7 片段最终运动学参考 | reference 打包与 render；MP4 哈希见 pilot 运行清单 | 存在 / NPZ、MP4 忽略 |
| `results/php_pilot_final/<类别>/<片段>_pd.npz` | 7 份自由根部 PD 回放轨迹 | `check_php_pilot`；逐路径见 quality_final 的 `pd.trajectory` | 存在 / NPZ 忽略 |
| [flat_reference.mp4](../../results/php_pilot_v2/flat_reference.mp4) | v2 五片段参考预览，非训练策略 | [pilot 收尾说明](../tienkung3_29dof/php_pilot/README.md) | 存在 / MP4 忽略 |
| [站立训练目录](../../logs/tienkung3_29dof/php_pilot_stand_v2) | args、train_log.csv、`ckpt_000200.pt`、`last.pt`；checkpoint 为 200 轮，未评估 | [standing_args](../tienkung3_29dof/php_pilot/evidence/standing_args.json)；run_manifest 的 checkpoint_sha256 对应 `last.pt`，data_hash 对应训练输入 | 存在 / `logs/` 与 PT 忽略 |
| [平地训练目录](../../logs/tienkung3_29dof/php_pilot_flat_v2) | args、train_log.csv；100 轮日志后 OOM | [flat_training.log](../tienkung3_29dof/php_pilot/evidence/flat_training.log)；**没有 checkpoint** | 日志存在 / `logs/` 忽略 |
| [integration_stage1_v3](../../logs/tienkung3_29dof/integration_stage1_v3)、[旧 tiny policy 视频](../../results/tienkung3_tiny_policy.mp4) | M0–M3 的 4 轮链路检查；不是 PHP 200 轮策略 | [training_smoke](../tienkung3_29dof/evidence/training_smoke.json)、[render_smoke](../tienkung3_29dof/evidence/render_smoke.json) | 存在 / 日志及视频忽略 |

截至 2026-11-08：没有该轮训练后的策略评估录像，也没有该轮 PD MP4。正式全量天工 PHP 数据集、独立 train/validation/test 清单和跨机器任务队列尚未交付；不要将原迁移计划中的目标路径当作已存在资产。

2026-11-08 只读检查确认两个站立 checkpoint 的 `iteration` 均为 200，`data_hash` 均为 `7ebf428edca084738cc417db16eb2b3886ec66daa9aab11e83009fa946b91ba2`。它们是分别序列化的文件，文件哈希不同：`last.pt` 与历史清单中的 `ad4c6f5168eb2a8f29e75611b4906e6393e484f71b84e1683ee2991acf96d1a3` 一致；`ckpt_000200.pt` 当前 SHA-256 为 `cc82f06fcaf0c2a9ee7caf41824b97c5d714a906a01c799a1edb48fd1bfd5587`。读取元信息不是策略评估。

## 5. 新增与替换资产的维护规则

- 新增资产记录用途、来源、输入哈希、生成命令、版本与输出；将日志/报告链接到对应 taskmemory 事件。
- 用明确状态区分源资产、中间失败、探索性数据、最终收尾版本、正式验收通过和未生成；保留被替代版本的证据。
- 身份或文件哈希不一致时先查版本，不能复用旧标签、旧准入报告或旧流水线完成标记。
- 运行报告中 `source_motion_hash` 的历史命名实际指目标 motion；原始 G1 来源用 selection manifest 的 `source_hash` 查找。
- 本清单只描述文件存在性及证据，不自动下载、删除、清理或重建资产。另一台机器缺失的本地链接应按来源恢复，不能推断文件已进入 Git。

`php_repair_v3_20261109/delivery_report.json` 保存本次子集的逐变体报告；`accepted_variant_index.json` 是旧版优先的 205 条/1814 变体资产索引，[覆盖核验](evidence/repair_v3_coverage.json)保存 910 原版 + 904 新增的统计。索引不是合并后的 MotionLib 数据目录。

## 合格动作参考录像（2026-11-09）

`results/php_success_previews_v3_20261109/` 保存 4 条 1280×720、50 fps 的完整运动学回放：`01_straight.mp4`（5.38 s）、`02_turn.mp4`（5.20 s）、`03_low_step.mp4`（8.26 s）、`04_high_step.mp4`（9.10 s）。所选动作均 10/10 变体几何通过，录像使用原始变体 0。来源批次、动作/地形/质量/视频哈希见 [录像清单](evidence/success_preview_v3.json)。这些是转换参考录像，不是策略或实机效果；帧数对应完整源轨迹，没有截短。

### 原版 G1 与天工同步对照

`results/php_success_previews_v3_20261109/comparison/` 保存 4 条 `*_comparison.mp4` 及 27.94 秒的 `all_comparisons.mp4` 合集。左侧原版 G1，右侧天工；1920×640、50 fps，相同原始地形变体 0、相同帧和公共相机位姿，保留真实世界尺寸。全部为运动学回放，不包含策略/动力学仿真。帧数、来源与视频哈希见 [对照录像清单](evidence/success_comparison_v3.json)。复现：`python -m tools.render_php_comparison results/php_success_previews_v3_20261109`。

对照后确认这四条转换轨迹的双臂 14 关节均全程固定；“合格动作”在此仅指既有几何/接触门槛，不能代表原始动作完整保留。[手臂诊断](evidence/preview_arm_diagnosis_v3.json)保存逐关节角度极差、初始角度、手/肘接触帧数及体型标定。录像与旧数据保持不变。

## v4 自由摆臂验证及运行资产（2026-11-09）

- `data/tienkung3_29dof/motions/php_segment_v4_preview_a_20261109`：四条完整动作、1397 帧、40/40 变体通过几何和摆臂验收；含冻结源码。报告为 [segment_v4_preview_a_report.json](evidence/segment_v4_preview_a_report.json)。
- `results/php_segment_v4_previews_20261109/comparison/`：四条新版同步对照及 `all_comparisons.mp4`，1920×640、50 fps、合计1397帧/27.94秒；均为运动学回放，视频标注两项验收结果。[视频身份清单](evidence/segment_v4_preview_comparison.json)。
- `data/tienkung3_29dof/motions/php_segment_v4_full_20261109`：296 条全库首轮正在运行，尚非全部通过。日志 `.cache/php_conversion/segment_v4/full.log`。
- `data/tienkung3_29dof/motions/php_segment_v4_variant_probe_a_20261109`：首个长轨迹失败地形 1、6 的专用求解已完成，两个指定变体均通过；去重后 10/10 原始变体通过。[槽位核验](evidence/segment_v4_variant_probe_coverage.json)。
- `.cache/php_conversion/segment_v4/frozen_runner`：与四条小批匹配的求解/检查源码，用于隔离正在运行的任务与后续改进。新、旧批次不能混同统计。
- `data/tienkung3_29dof/motions/php_segment_v5_contact_pilot_20261109`：14 条短时膝/肘擦碰分类试验正在运行；独立冻结代码 `.cache/php_conversion/segment_v5/frozen_runner`，日志 `.cache/php_conversion/segment_v5/pilot.log`。不以启动作为改进成功证据。

## 躯干前倾修正试验（2026-11-09）

- `data/tienkung3_29dof/motions/php_posture_v9_pilot_20261109`：四条完整动作1397帧，40/40原几何与摆臂验收通过；独立审计验证36个输出哈希，包含45个冻结源码文件 `source_snapshot`。试验基于v6加显式躯干/骨盆朝向项，尚未并入工作区v8或全库。
- [新版同步录像](../../results/php_posture_v9_previews_20261109/comparison/all_comparisons.mp4)：四条视频合辑，1920×640、50fps、1397帧/27.94秒，运动学回放；[视频清单](evidence/posture_v9_preview_manifest.json)。旧视频原样保留。
- [逐帧前倾曲线](../../results/php_posture_v9_previews_20261109/comparison/torso_lean_comparison.png)、[姿态指标](evidence/posture_v9_pilot_comparison.json)、[独立审计](evidence/posture_v9_pilot_audit.json)、[v6→v9隔离对照](evidence/posture_v9_ablation.json)。躯干前倾RMSE降到0.08°–0.28°，骨盆动态误差仍需改进；不要将几何/摆臂通过解释成全部姿态或动力学通过。

## 正式三项验收与宽度适配试验（18:53）

- `php_segment_v9_pilot_20261109`：14条、10476帧，88/140通过几何/摆臂/躯干三项，独立审计106哈希。[报告](evidence/segment_v9_pilot_report.json)、[65项回归](evidence/segment_v9_final_tests.txt)。
- [正式v9四条同步录像](../../results/php_segment_v9_previews_20261109/comparison/all_comparisons.mp4)：四控制40/40，1397帧/50fps/1920×640；[身份清单](evidence/segment_v9_preview_manifest.json)、[指标](evidence/segment_v9_preview_summary.json)。
- `php_segment_v10_peak_pilot_20261109`：7条、30/70，直行及攀爬失败，不推广。[报告](evidence/segment_v10_peak_pilot_report.json)。
- [当前全库覆盖](evidence/segment_v9_v10_coverage.json)：88/2960，缺2872；历史几何/摆臂通过项不代替新姿态通过项。
- `php_segment_v4_full_20261109_distance_recheck`：不改原候选完成原异常的10变体验收，结果为质量拒绝，程序异常已修复；[核验](evidence/segment_v4_distance_recheck.json)。
- `php_segment_v11_width_pilot_20261109`：8条宽度适配小批进行中，源码在对应source_snapshot；[准备检查](evidence/segment_v11_width_preparation.json)，尚无通过结论。

## v11已验证资产与全库重跑（19:05）

- `php_segment_v11_width_pilot_20261109`已结束：5 accepted/3 rejected、50/80；[独立报告](evidence/segment_v11_width_pilot_report.json)、[65项回归](evidence/segment_v11_tests.txt)。三条攀爬未通过。
- [最新版同步录像](../../results/php_segment_v11_previews_20261109/comparison/all_comparisons.mp4)，1397帧/50fps/1920×640；[清单](evidence/segment_v11_preview_manifest.json)、[四样本三项验收指标](evidence/segment_v11_preview_summary.json)。
- [正式覆盖审计](evidence/segment_v11_coverage.json)：合并v11与v9后98/2960，缺2862，保持原始来源/帧/槽位身份。
- `php_segment_v11_full_20261109`：296条全库重跑进行中，复用8条完全同身份终态（`reused_pilot.json`），剩288条；完整冻结源码在source_snapshot。日志 `.cache/php_conversion/segment_v11/full.log`，真实进程状态见taskmemory最新记录并重新核实。


## 19:24 前倾复核与摆腿试验

- [同源前倾曲线](../../results/php_segment_v11_previews_20261109/comparison/torso_lean_v3_v11.png)、[原生FK数据](evidence/posture_v3_v11_comparison.json)：四条v3/v11与G1逐帧比较。
- [v12独立小批报告](evidence/segment_v12_swing_pilot_report.json)、[v11/v12对照](evidence/segment_v12_swing_comparison.json)、[65项回归](evidence/segment_v12_tests.txt)、[准备阶段不变量](evidence/segment_v12_swing_preparation.json)：40/80，退化10槽位，不推广，源码保存在独立run/source_snapshot。
- 旧v4全库已结束；全量独立审计仍运行，待输出 `segment_v4_final_baseline_audit.json`。当前全库重跑为v11，状态和PID见taskmemory最新记录。


## 19:52 初始化、姿态修正及数值恢复

- [v4全量旧口径审计](evidence/segment_v4_final_baseline_audit.json)：2960全部检查、1950通过，仅诊断基线。
- [v13初始化预检](evidence/segment_v13_initialization_precheck.json)、[完整小批审计](evidence/segment_v13_init_pilot_audit.json)、[v11/v13对照](evidence/segment_v13_init_comparison.json)：50/80，攀爬误差下降但未通过。
- [v14独立报告](evidence/segment_v14_pose_polish_pilot_report.json)、[v13/v14对照](evidence/segment_v14_pose_polish_comparison.json)：50/80，三攀爬姿态全部通过，仍有碰撞/接触拒绝。
- [48项最终回归](evidence/segment_v13_distance_final_tests.txt)、[零见证fixture](evidence/distance_zero_witness.json)、[符号不一致fixture](evidence/distance_signed_witness.json)、[四个距离fixture测试](evidence/distance_signed_witness_tests.txt)。
- [v13异常复验](evidence/segment_v13_distance_recheck.json)、[v11第1条复验](evidence/segment_v11_distance_recheck_288081a3.json)、[v11第2条最终复验](evidence/segment_v11_distance_recheck_bf725426_signed.json)：均保持原运动，验收结束但质量拒绝。
- [地形专项计划](evidence/segment_v11_committed_terrain_retry_plan.json)、[8条启动选择](evidence/segment_v11_terrain_retry_pilot_selections.json)：实际对应失败地形重新求解，完整9089帧，结果待收齐；run为`php_segment_v11_terrain_retry_pilot_20261109`。
- v15源码和selections已准备但未运行、未验证；不是可用算法或已通过资产。


## 20:03 补充

- [专项完整审计](evidence/segment_v11_terrain_retry_pilot_report.json)、[专项对照](evidence/segment_v11_terrain_retry_comparison.json)：8个指定槽位全部通过，全局新增7。
- [分离轴修复49项回归](evidence/segment_v13_distance_separation_tests.txt)、[保留真实穿透测试](evidence/distance_real_penetration_test.txt)、[第三异常最终复验](evidence/segment_v11_distance_recheck_d2855473_separated.json)。
- v15已从准备转为运行，run `php_segment_v15_collision_peak_pilot_20261109`；尚无通过结论，最新PID与源码状态见taskmemory。


## 20:29 新增资产

| 资产 | 状态与用途 |
|---|---|
| [运行中覆盖快照](evidence/segment_v11_committed_coverage_20261109_2028.json)及同名index | 已独立核验533/2960，缺2427；运行中下限，不是最终交付 |
| [索引防护检查](evidence/committed_index_guards.txt) | 未显式允许运行中父批次、源错配、候选错配、变体编号错配均拒绝 |
| [31地形专项计划](evidence/segment_v11_terrain_retry_expand_plan.json) | 15来源、33854完整帧、31原始缺失槽位；独立批次运行中 |
| [v15独立报告](evidence/segment_v15_collision_peak_pilot_report.json)、[对照](evidence/segment_v15_collision_peak_comparison.json) | 50/80，0新增，不推广 |
| [v16测试](evidence/segment_v16_tests.txt)、[CUDA检查](evidence/segment_v16_graph_smoke.json)、[碰撞诊断](evidence/segment_v16_collision_diagnosis.json) | 53测试通过，三攀爬仍碰撞拒绝，完整小批在收尾验收 |
| `php_segment_v11_full_20261109_distance_recheck_{c4bcaa27,e8dd4a97,c89ee388}_separated` | 三个新增异常均完成隔离复验，全部0/10质量拒绝；原候选和历史failed保持 |

20:32 更新：v16已收齐，50/80、60hash核验，[独立报告](evidence/segment_v16_constrained_pilot_report.json)、[v14/v15/v16对照](evidence/segment_v16_constrained_comparison.json)。v17已通过52项分批检查和[CUDA检查](evidence/segment_v17_graph_smoke.json)，完整小批 `php_segment_v17_velocity_pilot_20261109` 正在运行。第7异常cb043e0a也已完成独立复验，[证据](evidence/segment_v11_distance_recheck_cb043e0a_separated.json)。


## 20:51 源接触数值修复证据

| 资产 | 状态与用途 |
|---|---|
| [源零距离诊断](evidence/source_zero_contact_diagnosis.json)、[独立分离证明](evidence/source_zero_contact_certificate.json) | 原G1第488/491帧实际分离31.5/32.7mm，原标签假零 |
| [测试现场](evidence/source_zero_contacts_fixture.json)、[53项回归](evidence/segment_v18_tests.txt) | 保留真实接触、原标签不变、错配复核记录必须拒绝 |
| [全变体准备预检](evidence/segment_v18_all_variants_precheck.json) | 8来源53不同地形、134零标签重算；不代表目标质量通过 |
| [v17审计](evidence/segment_v17_velocity_pilot_report.json)、[对照](evidence/segment_v17_velocity_comparison.json) | 50/80、60hash，0新增，不推广 |
| `php_segment_v18_source_contact_pilot_20261109` | 完整8条/80槽位正在运行，46代码snapshot；最终独立审计须用其冻结runner以核验源receipt |


## 21:32 新资产

- [v18四条完整对照](../../results/php_segment_v18_previews_20261109/comparison/all_comparisons.mp4)：1397帧、40/40槽位；[摘要](evidence/segment_v18_preview_summary.json)、[主源码推广记录](evidence/segment_v18_promotion.json)。
- [v19首条完整攀爬对照](../../results/php_segment_v19_climb_preview_20261109/comparison/01_low_climb_comparison.mp4)：894帧，原10变体通过；[独立索引](evidence/segment_v19_first_climb.index.json)、[录像验证与根部跳变限制](evidence/segment_v19_first_climb_preview.json)。v19尚未推广。
- [31地形专项审计](evidence/segment_v11_terrain_retry_expand_report.json)、[去重新增31槽位](evidence/segment_v11_terrain_retry_expand_comparison.json)。
- [920槽位独立历史快照](evidence/segment_v11_committed_coverage_20261109_2100.json)，非最终全覆盖。
- [两条异常恢复的20槽位索引](evidence/segment_v11_recovered_steps.index.json)、[80条攀爬旧v11完整拒绝统计](evidence/segment_v11_all_climb_baseline.json)。
- 活跃新增批次 `php_segment_v18_terrain_retry_20261109`（27目标地形）、`php_segment_v19_transition_pilot_20261109`（20完整动作），计划和运行参数见taskmemory最新条目；旧运行中的全库v11保留其冻结代码。


## 2026-11-10 新结果

- [v20完整低攀爬同步录像](../../results/php_segment_v20_climb_preview_20261110/comparison/01_low_climb_comparison.mp4)、[894帧录像核验](evidence/segment_v20_first_climb_preview.json)：原10变体通过既有门槛，根部平滑明显改善，尚非动力学验证。
- [完整根部对比曲线](../../results/php_segment_v20_diagnostics_20261110/root_continuity.png)、[逐源根部指标](evidence/segment_v18_v19_v20_root_diagnostics.json)、[296条v11根部诊断](evidence/segment_v11_all_root_diagnostics.json)：证明旧quality_pass不能替代根部连续性验收。
- [假膝接触台阶专项审计](evidence/segment_v18_step_repair_audit.json)：465完整帧、10/10原变体通过，新增10槽位；[源标签诊断](evidence/segment_v18_step_source_contact_diagnosis.json)。
- [v20的59项测试](evidence/segment_v20_tests.txt)、[未来版本源receipt验收回归](evidence/source_validation_future_versions_tests.txt)。正式求解器仍v18。


## v21完整门槛资产

- [四条新版完整对照](../../results/php_segment_v20_previews_20261110/comparison/all_comparisons.mp4)、[含独立根筛查的40槽位验证](evidence/segment_v20_preview_summary.json)。
- [v21正式推广](evidence/segment_v21_promotion.json)、[65项回归](evidence/segment_v21_tests.txt)、[全库启动与身份核验](evidence/segment_v21_full_launch.json)。
- [v20小批独立审计](evidence/segment_v20_root_pilot_report.json)、[根筛查后69槽位覆盖](evidence/segment_v20_strict_coverage.json)、[v21实际首批23槽位索引](evidence/segment_v21_first_committed.index.json)。
- [旧v11全296源最终拒绝统计（含9恢复）](evidence/segment_v11_full_final_rejections.json)：2052为旧门槛，不含根连续性。
- [v18地形27专项审计](evidence/segment_v18_terrain_retry_report.json)、[目标27/27、去重新增29](evidence/segment_v18_terrain_retry_comparison.json)。
- 全库运行 `php_segment_v21_full_20261110`，两条攀爬专项 `php_segment_v21_climb_targeted_20261110`。所有新计数使用根部准入；实时统计不是最终独立交付证明。


## 14:27 专项证据

- [两攀爬专项独立审计](evidence/segment_v21_climb_targeted_report.json)、[去重新增1槽位对照](evidence/segment_v21_climb_targeted_comparison.json)：低左45补齐10/10，高右转仍拒绝。
- [高攀爬碰撞逐帧定位](evidence/segment_v20_high_climb_collision_diagnosis.json)：202/204/213帧。
- [4地形重跑计划](evidence/segment_v21_terrain_pilot_plan.json)、[地形/v22小批启动](evidence/segment_v21_terrain_v22_peak_launch.json)、[v22 CUDA检查](evidence/segment_v22_graph_smoke.json)、[11项针对回归](evidence/segment_v22_gate_regression.txt)。两小批尚未收齐，不能计预期通过。


### 2026-11-10 14:46 新证据

- 当前四条完整前倾对照：evidence/posture_v3_v20_comparison.json；图../../results/php_segment_v20_diagnostics_20261110/torso_lean_comparison.png。录像沿用已独立验证的v20四条。
- 地形4目标独立审计及新增5槽位：segment_v21_terrain_pilot_report.json、segment_v21_terrain_pilot_comparison.json；37组合扩展的plan/selections/launch均在evidence。
- v22无新增且不推广：segment_v22_collision_peak_report.json、segment_v22_collision_peak_comparison.json。
- 仅诊断、不计覆盖：segment_v21_clearance_probe.json、segment_v21_retained_witness_diagnosis.json、segment_v20_high_climb_hip_sensitivity.json。
- 隔离v23：segment_v23_gate_regression.txt、segment_v23_graph_smoke.json、segment_v23_collision_pairs_launch.json；尚无终态结果。


### 2026-11-10 15:34 新修复资产

- v24正式12源：data/tienkung3_29dof/motions/php_segment_v24_native_polish_pilot_20261110；证据segment_v24_native_polish_report.json、comparison.json、segment_v24_promotion.json。119/120，结合v21专项严格索引segment_v24_pilot_strict.index.json覆盖12源120槽位。
- v24完整四对照：../../results/php_segment_v24_previews_20261110/comparison/all_comparisons.mp4；高攀爬完整568帧：../../results/php_segment_v24_high_climb_preview_20261110/comparison/01_high_climb_comparison.mp4；对应preview_summary/manifest/high_climb_preview证据已核验。
- v24扩大11源：php_segment_v24_native_polish_expand_20261110，69/110；expand_report与expand_comparison记录新增69、剩41，不能沿用错误70。
- v25隔离16源：php_segment_v25_native_joint_pilot_20261110运行中；segment_v25_tests.txt、root_guard.json、native_joint_launch.json。正式主代码仍v24。
- 单帧/完整诊断probe只供算法研究，不计交付覆盖：segment_v21_pose_feasibility.json、pose_fixed_rotation_feasibility.json、segment_v21_native_polish_probe.json、segment_v24_native_joint_probe.json。


2026-11-10 16:28最新资产：v25推广及四录像已验收（segment_v25_promotion、preview_summary/manifest）；v21地形扩大37新增原槽位（segment_v21_terrain_expand_report/comparison）；v24膝专项新增1（segment_v24_knee_terrain_report/comparison）；v25手专项新增4（segment_v25_hand_terrain_report/comparison）；v25扩大4源13/40（segment_v25_native_joint_expand_report/comparison）。最新前倾图results/php_segment_v25_diagnostics_20261110/torso_lean_comparison.png及posture_v3_v25_comparison.json。v26仍隔离，只有速度诊断与测试，不计覆盖。


2026-11-10 16:45：v26正式证据segment_v26_promotion/native_velocity_report/native_velocity_comparison/preview_summary/preview_manifest/final_tests/root_guard；四完整对照results/php_segment_v26_previews_20261110/comparison/all_comparisons.mp4。v25原16源严格组合index为segment_v25_pilot_strict.index.json（160/160）。运行/排队新增v25_full_window与v26_support_terrain，分别3源30与7求解70检查，不能当已通过。诊断segment_v21_live_rejections_1635.json、segment_v21_arm_phase_diagnosis.json。


2026-11-10 17:23新增：segment_v27_promotion/free_arm_report/free_arm_comparison/root_guard/preview_summary/preview_manifest.json及full_tests.txt为正式v27证据；results/php_segment_v27_previews_20261110/comparison/all_comparisons.mp4为最新四条完整同步录像。segment_v21_full_report.json已验证296源/2960检查/2593通过，segment_v25_full_window_report/comparison.json新增30，segment_v25_terrain_expand_report/comparison.json目标57/57，segment_v26_support_terrain_report/comparison.json目标7/7。remaining_20261110_1717.json记录17:20:28 provisional2933/2960、缺27；remaining_20261110_1721_reasons.json记录全部原因。segment_v28_root_peak_plan/selections/launch.json为隔离实验5源1799帧50槽位，运行中、未正式推广。


2026-11-10 17:37：正式v29证据segment_v29_promotion、root_tail_report、root_tail_comparison、preview_summary/manifest、root_preview_summary.json及full_tests.txt（83通过）。最新四完整原G1/天工同步录像results/php_segment_v29_previews_20261110/comparison/all_comparisons.mp4；根速度最后难例402帧录像results/php_segment_v29_root_preview_20261110/comparison/01_cfg0026_comparison.mp4；新旧前倾曲线results/php_segment_v29_diagnostics_20261110/torso_lean_comparison.png，数据posture_v3_v29_comparison.json。segment_v27_terrain_remaining_report/comparison.json为最后17专项：17完整专项求解5810重复帧170检查149通过、目标17/17、新增17，153输出hash已验证。all_original_variants_final_plan_20261110.json登记14个终态父批及最终audit命令；其最终.json/.index.json仍在生成，当前仅实时2960候选全部覆盖，尚未标目标完成。


2026-11-10 17:47:44最终状态：独立全库覆盖验收已完成，296源/181236帧/2960原地形槽位全部通过、剩余0。evidence/all_original_variants_final_20261110.json为最终报告，同名.index.json为原source_variant→motion/terrain/export_slot/labels完整映射；evidence/delivery_summary_20261110.json绑定交付证据。主v29、83项测试、小批50/50与四完整录像40/40通过，原门槛与完整源数据不变。最新四对照results/php_segment_v29_previews_20261110/comparison/all_comparisons.mp4；最后根难例results/php_segment_v29_root_preview_20261110/comparison/01_cfg0026_comparison.mp4。索引保留14个已终态父批的合格资产，勿把专项重复项计作新增或当作直接MotionLib目录；PD/策略/实机验证不在已通过声明内。
