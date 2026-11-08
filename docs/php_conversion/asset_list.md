# PHP 转换资产清单

核对日期：2026-11-08（Asia/Shanghai）。返回[文档入口](README.md)。路径均相对仓库根目录；“存在”描述本机核对时的状态，不保证其他机器已同步。

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
| `.cache/php_conversion/representative_v2/` | 修复序列化后的十样本双卡验证 | 正在运行；最终状态以该目录 manifest 为准 |

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

没有本轮训练后的策略评估录像，也没有本轮 PD MP4。正式全量天工 PHP 数据集、独立 train/validation/test 清单和跨机器任务队列尚未交付；不要将原迁移计划中的目标路径当作已存在资产。

2026-11-08 只读检查确认两个站立 checkpoint 的 `iteration` 均为 200，`data_hash` 均为 `7ebf428edca084738cc417db16eb2b3886ec66daa9aab11e83009fa946b91ba2`。它们是分别序列化的文件，文件哈希不同：`last.pt` 与历史清单中的 `ad4c6f5168eb2a8f29e75611b4906e6393e484f71b84e1683ee2991acf96d1a3` 一致；`ckpt_000200.pt` 当前 SHA-256 为 `cc82f06fcaf0c2a9ee7caf41824b97c5d714a906a01c799a1edb48fd1bfd5587`。读取元信息不是策略评估。

## 5. 新增与替换资产的维护规则

- 新增资产记录用途、来源、输入哈希、生成命令、版本与输出；将日志/报告链接到对应 taskmemory 事件。
- 用明确状态区分源资产、中间失败、探索性数据、最终收尾版本、正式验收通过和未生成；保留被替代版本的证据。
- 身份或文件哈希不一致时先查版本，不能复用旧标签、旧准入报告或旧流水线完成标记。
- 运行报告中 `source_motion_hash` 的历史命名实际指目标 motion；原始 G1 来源用 selection manifest 的 `source_hash` 查找。
- 本清单只描述文件存在性及证据，不自动下载、删除、清理或重建资产。另一台机器缺失的本地链接应按来源恢复，不能推断文件已进入 Git。
