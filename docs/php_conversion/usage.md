# PHP 转换工具链使用手册

更新日期：2026-11-09。返回[文档入口](README.md)。先阅读以下通用入口；后续编号章节保留历史 pilot 的复现方式。

## v4 摆臂与全部原始地形验收

新版 `php-segment-v4` 的导出同时要求几何及自由摆臂保真通过；旧版 `geometry_pass` 不等于新版 `quality_pass`。小批 `php_segment_v4_preview_a_20261109` 已通过 40/40，首轮全库 `php_segment_v4_full_20261109` 仍在运行。不要对活跃控制器再次启动转换。

复现新版四条视频（新建输出目录）：

```bash
python -m tools.render_php_comparison results/php_success_previews_v3_20261109 \
  --run data/tienkung3_29dof/motions/php_segment_v4_preview_a_20261109 \
  --out results/php_segment_v4_previews_20261109/comparison
```

检查完成批次的原始变体覆盖；可以追加多个相同保真政策的专用变体修正批次：

```bash
python -m tools.audit_php_variant_coverage \
  data/tienkung3_29dof/motions/php_segment_v4_full_20261109 \
  data/tienkung3_29dof/motions/php_segment_v4_variant_probe_a_20261109 \
  --out .cache/php_conversion/segment_v4/coverage.json
python -m tools.plan_php_variant_retries \
  .cache/php_conversion/segment_v4/coverage.json \
  --out .cache/php_conversion/segment_v4/missing_selections.json
```

上述全库覆盖命令需等对应批次终态后运行。它检查原始 296 条/2960 槽位；即使输入只有通过的四条小批，也会明确报告剩余 2920，而不会把小批当全库。最终验收增加 `--require-all`，仍有任一缺失槽位时返回非零状态。每个修正 selection 保留完整源帧区间，使用独立输出键及 `solve_variant`；新的转换批次通过现有 `--selections` 接口执行。相同原始几何的别名只需求解一次，但所有原编号均逐槽位计入覆盖；多次重试不重复计数。

冻结求解代码位于 `.cache/php_conversion/segment_v4/frozen_runner`，原资产通过 `data` 符号链接引用；继续运行旧版本时从该目录执行，使用绝对输入/输出路径及原配置。新的算法改动另建源码快照和输出目录，不能使正在运行的求解与验收使用不同版本。

## v3 分类别修复重跑

原 v2 全量批次已结束且保持不可变。近接触距离异常的原候选复验保存在 `data/tienkung3_29dof/motions/php_full_v2_20261109_distance_recheck/`：`provenance.json` 绑定父任务、候选哈希和验收代码，`result.json` 为复验结果。这是 assessment-only 目录，不是可恢复的求解批次。

两轮小批证据见 [12 条对照](evidence/repair_v3_pilot_comparison.json)和 [6 条长轨迹对照](evidence/repair_v3_long_comparison.json)。新批次 `data/tienkung3_29dof/motions/php_repair_v3_20261109` 仅重跑 [146 条行走/台阶](evidence/repair_v3_rerun_selections.json)，不含未验证改善的攀爬。恢复必须携带原 selections：

```bash
python -m tools.convert_php \
  --out data/tienkung3_29dof/motions/php_repair_v3_20261109 \
  --selections docs/php_conversion/evidence/repair_v3_rerun_selections.json \
  --devices 0,1 --workers-per-gpu 2 --cpu-workers 6 --resume
```

恢复前检查控制器是否仍在运行。源码、模型及求解配置需与该批次一致。此批次是子集，不可用 296 条全库 inventory 声称完整覆盖；新旧候选分别保留，不用 v3 候选直接覆盖 v2 已通过变体。原 `deliver_php_conversion` 只支持同代码的失败任务重试，不能用它合并此次算法变更。

## 2026-11-09 批次、审计和交付

本轮原始批次为 `data/tienkung3_29dof/motions/php_full_v2_20261109`；最终汇总交付目录为同级 `php_full_v2_20261109_delivery`，采用 `php-fullbody-v2`。实际启动参数为两张 5090、每卡 4 个 worker、12 个 CPU worker、512 帧窗口、48 帧重叠和 1000 次初始迭代。多身体接触窗口另执行最多 8 轮、每轮 400 次的原生碰撞点修正；仅足部接触的有地形轨迹执行最大 8 mm 的平滑根高修正。原始地形、采样率、帧数和验收门槛不变。

检查完成批次及生成详细报告：

```bash
python -m tools.audit_php_conversion \
  data/tienkung3_29dof/motions/php_full_v2_20261109_delivery \
  --inventory docs/php_conversion/evidence/source_inventory.json \
  --out docs/php_conversion/evidence/full_v2_audit.json

python -m tools.report_php_conversion \
  data/tienkung3_29dof/motions/php_full_v2_20261109_delivery \
  --inventory docs/php_conversion/evidence/source_inventory.json \
  --out data/tienkung3_29dof/motions/php_full_v2_20261109_delivery/delivery_report.json
```

审计检查全部源文件的完整帧区间、全部原始地形槽位、候选/导出/标签/质量哈希、变体映射及来源分组。`delivery_report.json` 按类别汇总，并列出每个动作及变体的拒绝原因和接触指标。求解源码另冻结在批次的 `source_snapshot/`，与 `plan.json` 的源码哈希对应。恢复任务必须使用原源码和配置。

训练或下游数据消费者只能从 `accepted/` 配合 `accepted_contacts/` 取数据，并读取同批次 quality、collision policy 和 splits。全库处理完成不等于全库合格；`development` 划分及 PD/学习跟踪未运行的状态仍适用。

原始批次若发生程序错误，使用新的 run ID 与求解配置单独重试，不能覆盖原错误或放宽检查。`tools.deliver_php_conversion` 只允许用完整、同输入同模型的重试替换 failed 任务；不能替换已接受或已拒绝的结果。交付目录保存各父批次 plan/manifest、逐任务来源和全部尝试记录，属于只读汇总，不是可传给 `convert_php --resume` 的求解目录。

## 通用全量入口

本机使用 `/home/js/miniconda3/envs/acp_supermimic/bin/python`；工作区为 `/home/js/robot/locomotion/ACP_supermimic_Tienkung`。新入口一次串接求解、逐变体独立验收、标签、合格子集及汇总，不需要另行对候选数据运行筛选脚本。

```bash
conda activate acp_supermimic
cd /home/js/robot/locomotion/ACP_supermimic_Tienkung

# 只扫描：写 plan，不创建任何 motion 输出。
python -m tools.convert_php --scan --out .cache/php_conversion/plan_01

# 每次新实验选择新目录；两卡各一个 worker，CPU 独立验收池。
python -m tools.convert_php \
  --out data/tienkung3_29dof/motions/php_full_v1 \
  --devices 0,1 --workers-per-gpu 1 --cpu-workers 12 \
  --window-frames 512 --overlap 48 --iterations 1000

# 恢复时保持源码、输入、模型与求解配置相同；可调整设备/CPU 并发。
python -m tools.convert_php \
  --out data/tienkung3_29dof/motions/php_full_v1 --resume \
  --devices 0,1 --cpu-workers 12
```

任务目录说明：`plan.json` 保存冻结输入、源码哈希及配置；`tasks/*.json` 保存逐任务状态和输出哈希；`checkpoints/*.npz` 保存已完成窗口；`candidates/` 与 `candidate_contacts/` 保存所有已转换动作及诊断标签；`quality/` 保存逐变体结果；`accepted/` 与 `accepted_contacts/` 只含合格变体；`manifest.json` 汇总覆盖数；`splits.json` 记录来源家族与当前 development 状态。候选动作保留原始 10 个地形槽位，即使几何完全相同也保留编号映射。

`--only` 按源动作键子串筛选。`--selections file.json` 接收列表，每项含 `source_key`、`key`（输出类别/名）、`start`、`end` 和可选 `solve_variant`。见 [十个代表样本配置](evidence/representative_selections.json)。接触 intent 现为 `(V,T,9)`，另含 offsets/anchors/normals/surfaces，不应交给只懂 `(T,2)` 的旧 pilot 检查器。

`--max-attempts` 默认 2；求解运行错误重试后仍失败则记录为 failed，队列继续，最后以退出码 2 报告。几何拒绝是有效检查结果，退出码 0 只代表批次运行完成，**不代表所有动作合格**。恢复会校验每个完成任务的产物哈希；输入、代码、模型、策略或参数变化必须使用新目录。`--retry-failed` 显式重试已有失败任务。

`geometry_pass` 使用有依据的踝部相邻组件排除，`strict_raw_geometry_pass` 仍包含原始两对重叠；两种结果同时保存。读取 `accepted` 时必须同时保留 collision policy 与 quality 报告，不能将其描述为未排除任何碰撞对的严格通过。动力学和学习策略跟踪本轮未运行。

## 1. 历史 pilot：环境与输入

所有命令从仓库根目录执行。已有本地环境时：

```bash
cd /home/js/robot/locomotion/Acp_supermimic_vibed
conda activate acp_supermimic
```

在其他机器上将 `cd` 改为实际仓库位置。[environment.yml](../../environment.yml)锁定 Python 3.11、torch 2.9.0+cu128、mjlab 1.6.0、MuJoCo/mujoco-warp 3.11.0、Warp 1.16.0 等依赖，并包含 ffmpeg。本机解释器为 `/home/js/miniforge3/envs/acp_supermimic/bin/python`；只用解释器绝对路径不会自动将 ffmpeg 加入 PATH。

已有资产时，先按[资产清单](asset_list.md)确认以下输入，无需重新构建：

| 输入 | 默认位置 | 用途 |
|---|---|---|
| G1 URDF、网格及 MJCF | `data/g1_description/` | 源 FK 与 G1 接触几何 |
| PHP 源动作和地形 | `data/php_parkour/<类别>/` | 当前 pilot 按固定路径选择 7 个片段 |
| G1 源接触标签 | `data/php_parkour_contacts/<类别>/<源片段>.npz` | 求解器读取距离及接触阶段，不可用天工标签替换 |
| 天工原始与派生模型 | `data/tienkung3_29dof/upstream/`、`generated/` | 目标 FK、几何质检与渲染 |
| 天工模型清单及参数 | `data/tienkung3_29dof/manifests/` | 模型哈希、关节顺序、执行器与身份校验 |

**新机器重建入口**如下。只在需要重建时运行；源数据下载脚本会下载全部 PHP 源资产并生成 G1 标签，但不会执行 G1 → 天工全量重定向。已有天工模型时不要把重建作为日常转换的前置步骤。

```bash
conda env create -f environment.yml
conda activate acp_supermimic
bash tools/download_acp_data.sh
python -m tools.prepare_tienkung3
```

[下载脚本](../../tools/download_acp_data.sh)还会生成 G1 MJCF；[天工资产工具](../../tools/prepare_tienkung3.py)固定模型上游提交，生成无损 OBJ、固定头部的 URDF/MJCF 和 manifests。本链路不使用旧 Isaac Lab 的 `tools/setup.sh`。下载源和版本见资产清单；几何修改后应更新派生模型与清单，并重新质检、标注。

## 2. 历史 pilot：命令与能力边界

| 模块 | 主要参数与实际行为 |
|---|---|
| `tools.retarget_php_pilot` | `--out`、`--only`、`--iterations`（默认 1000）、`--device`（默认 `cuda:0`）、`--arm-mode`（`auto/neutral/tracked/clearance`）；固定 G1 → 天工，无 `--robot` 参数 |
| `tools.check_php_pilot` | `--motion-dir`、`--out`、`--record-dir`、`--only`、`--no-pd`、`--export-screened`、`--allow-baseline-self-overlap`；默认同时做几何和 PD |
| `supermimic.acp.make_labels` | `--robot`、`--motion_dir`、`--out`、`--workers`（默认 12）、`--clips`；`--threshold` 默认 0.005 m，`--max_dist` 默认 0.1 m |
| `tools.record_motion_reference` | 位置参数 `motion_dir out`；`--robot` 默认天工；打包 50 Hz 运动学轨迹，使用第 0 地形变体 |
| `supermimic.acp.render` | 位置参数 `traj out`；`--robot`、`--per_folder`（默认 1）、`--solid-label` 等 |
| `supermimic.acp.refine_motion` | `--robot`、`--motion_dir`、`--contacts_dir`、`--out`、`--clips`、`--device`、`--iters` 等；针对已经是目标机器人的动作做微调 |

注意参数风格：检查器使用 `--motion-dir`，标注与微调模块使用 `--motion_dir`。`--only` 是 pilot 中的完整片段键，如 `turning/left45`；`--clips` 在标注/微调中按名称子串筛选。不要把渲染模块同名的 `--clips` 当成标注过滤器。

当前重定向只能选择[7 个预定义片段](asset_list.md#七个最终片段)，没有任意输入路径、任意帧区间或地形变体参数。`auto` 对低台阶使用 `clearance`，其余使用 `neutral`；`tracked` 虽可调用，但不属于最终 7 片段已验证配置，也不代表手支撑能力已实现。

## 3. 在新目录复现小批量转换

以下各节在同一个已激活环境的 Bash 会话中执行。每次复现修改 `php_run`，并保持输出目录唯一；本例不会写入历史 `php_pilot_final` 或 evidence。

```bash
set -e
php_run=php_repro_20261108_01
php_base="data/tienkung3_29dof/motions/${php_run}"
php_results="results/${php_run}"
if [ -e "$php_base" ] || [ -e "$php_results" ]; then
  echo '输出目录已存在，请更换 php_run 后重新运行。' >&2
  exit 1
fi

python -m tools.retarget_php_pilot \
  --out "$php_base/flat" --device cuda:0 --iterations 1000 \
  --arm-mode neutral \
  --only standing/stand walking/forward_a walking/forward_b turning/left45 turning/right45

python -m tools.retarget_php_pilot \
  --out "$php_base/steps" --device cuda:0 --iterations 1200 \
  --arm-mode clearance --only low_step/forward_a low_step/forward_b
```

平地输出为 5 个片段，低台阶输出为 2 个片段，各自有 manifest。`neutral` 每次优化更新后将双臂还原默认姿态；`clearance` 还将左右 shoulder roll 固定为 +0.5/−0.5 rad。它们保留根部、腰和腿的任务空间转换，但没有保留源摆臂风格。

求解器在输出目录已有 `manifest.json` 时拒绝运行，不能先写平地再向同一目录追加台阶。当前也没有可靠的中途恢复、并发写 manifest 或自动合并功能；保留两个独立数据根目录，分别检查和标注即可。异常留下的部分目录也应保留诊断，选择新目录重跑。

## 4. 独立质检与结果判读

先只运行几何检查，输出全身地形/自碰撞及双脚支撑指标：

```bash
for php_group in flat steps; do
  python -m tools.check_php_pilot \
    --motion-dir "$php_base/$php_group" \
    --out "$php_results/${php_group}_geometry.json" --no-pd
done
```

检查器需要与动作同目录、同名的 `_intent.npz`。不要只复制 motion 和 terrain 后直接调用它。这里使用独立 MuJoCo 接触检查与原生几何求距；不是读取求解器 loss 来判断合格。

**命令退出码为 0 不代表质检通过。** 当前检查器记录失败而不据此返回非零退出码，且无匹配动作时可能不生成报告。必须确认报告存在、`clips` 数量为预期的 5/2，并逐条阅读以下字段：

| 字段 | 含义 |
|---|---|
| `pilot_screen_pass` | 排除两对已知小腿—脚部重叠后的探索性初筛 |
| `strict_geometry_pass` | 同时满足初筛条件和未单独排除上述两对的原始自碰撞门槛 |
| `baseline_self_collision_exception` | 原始自碰撞没有通过门槛；结合 `self_pair_max_m` 定位原因 |
| `rejection_reasons` | 初筛失败原因；此数组为空仍可能 `strict_geometry_pass=false` |
| `source_motion_hash` | 此报告中实际是**被检查的天工 motion** 哈希；原始 G1 哈希见来源 manifest |

检查保留 MuJoCo 的同焊接体、父子体默认排除，并另外报告原始结果及两对 pilot 例外结果。现有阈值如下：

| 指标 | 条件 |
|---|---|
| 关节硬限位误差 | ≤1e-4 rad |
| 根部四元数范数误差 | 加载时要求 ≤1e-4 |
| 地形/自碰撞穿透 | 最大 ≤5 mm，且超过 2 mm 不得连续达到 3 帧 |
| 意图支撑段表面距离 | P95 ≤10 mm |
| 连续支撑帧脚底水平速度 | P95 ≤0.05 m/s |
| 关节速度 | 相对逐关节参考上限的最大比值 ≤1 |

当前报告中的支撑距离与滑动只覆盖双脚；不得将它解释为手、膝等接触验收。

若需 PD 诊断，去掉 `--no-pd`，使用新报告和轨迹路径：

```bash
for php_group in flat steps; do
  python -m tools.check_php_pilot \
    --motion-dir "$php_base/$php_group" \
    --out "$php_results/${php_group}_with_pd.json" \
    --record-dir "$php_results/${php_group}_pd"
done
```

该协议为自由根部关节位置 PD，无根部外力扶持，沿用 ACP 关闭物理自碰撞的设置；50 Hz 参考、200 Hz 物理。`pd.tracking_pass` 要求没有提前终止，根部误差 RMSE ≤0.1 m、最大 ≤0.2 m，关节误差 RMSE ≤0.15 rad。报告同时给出饱和、速度超限与轨迹路径。PD 诊断不能替代学习控制器验收；源 G1 同协议结果见[基线报告](../tienkung3_29dof/php_pilot/evidence/source_g1_pd.json)。

### 严格筛选与探索性导出

需要建立严格几何通过子集时，可显式调用：

```bash
python -m tools.check_php_pilot \
  --motion-dir "$php_base/flat" \
  --out "$php_results/flat_export_gate.json" --no-pd \
  --export-screened "$php_base/flat_screened"
```

此示例没有启用例外。按当前模型及历史结果，可能没有任何片段导出；空结果不能作为训练集。导出仅复制 motion/terrain/intent，不复制运行 manifest 或生成新汇总清单，也不检查 PD 是否达标；应另外保留原 manifest 与此导出报告。

`--allow-baseline-self-overlap` 只有和 `--export-screened` 一起使用时才允许“初筛通过、严格检查失败”的片段进入探索性子集。它不修复模型、不改变严格结果，也不忽略其他初筛失败。需要此类试验时必须在新一轮记录中明确例外及用途，不作为正式验收默认配置。

## 5. 标签与运动学录像

以下命令为两个原始转换批次生成诊断标签和参考录像，**不会过滤几何失败片段**。正式合格子集应将输入替换为实际非空、通过对应门槛的筛选目录。

```bash
for php_group in flat steps; do
  python -m supermimic.acp.make_labels \
    --robot tienkung3_29dof --motion_dir "$php_base/$php_group" \
    --out "${php_base}/${php_group}_contacts" --workers 1
  python -m tools.record_motion_reference \
    "$php_base/$php_group" "$php_results/${php_group}_reference.npz" \
    --robot tienkung3_29dof
  python -m supermimic.acp.render \
    "$php_results/${php_group}_reference.npz" "$php_results/${php_group}_reference.mp4" \
    --robot tienkung3_29dof --per_folder 2 --solid-label 'reference IK'
done
```

`make_labels` 通过 CPU 多进程处理输入文件携带的每个地形变体，但本次 pilot 输出只保留第 0 变体。`--workers 1` 是小样本起点，不是性能上限。标签包含九类接触及 other_contact，不能据此声称求解器已经优化了所有接触类型。`--per_folder 2` 保留每类最多 2 个片段，能覆盖本例全部 7 个片段。

报告、标签和录像工具可能覆盖同名输出，因此重做某一阶段时也应使用新的报告文件或目录。

## 6. 文件约定与失效关系

| 文件 | 内容与依赖 |
|---|---|
| `<片段>_motion.npz` | `joint_pos=(T,36)`：root xyz + root quat wxyz + 29 关节；另有 joint_names、fps、robot_metadata、data_version、source、source_frames |
| `<片段>_terrain.npy` | `(K,V,10)`：center xyz + quat wxyz + full size xyz；世界坐标、米，箱体仅绕 yaw；pilot 导出 `V=1`，平地为 `(0,1,10)` |
| `<片段>_intent.npz` | `contact=(T,2)` 和 `foot_target=(T,2,3)`；源动作映射得到的双脚接触意图，供独立质检使用，不是最终九类接触标签 |
| `<批次>/manifest.json` | 求解版本、目标身份、来源动作/地形/源标签哈希、帧区间、变体、迭代参数和输出 motion 哈希；最终收尾清单另补有 terrain/intent 哈希 |
| `<标签根>/<类别>/<片段>.npz` | 目标几何计算的九类距离、接触点/法向/表面及接触标记；绑定 robot_metadata、motion_hash、terrain_hash |
| `summary.json` | 标签统计，不是数据质量准入报告 |
| 质检 JSON / 轨迹 NPZ | 当前检查对象的几何及可选 PD 结果；参考轨迹另记录 purpose 和来源哈希 |

时长使用 `(T-1)/50`，帧区间为零基、右端不包含。关节单位为 rad，轨迹为 50 Hz。加载器按关节名重排，校验身份、有限值和四元数范数，并连续化四元数符号。`MotionLib` 按目标模型计算 FK 和速度；pilot 不导出 G1 的 body pose/velocity 缓存。通用 motion loader 支持重采样，不代表 pilot 求解器支持任意 fps：后者明确要求输入 50 Hz。

模型/几何变化后更新模型身份并重做相应转换与质检；motion 或 terrain 改变后必须重做标签、报告和参考录像。不要仅改元数据来让旧结果通过身份校验。

## 7. 微调与训练的边界

[refine_motion](../../supermimic/acp/refine_motion.py)接收已经属于目标模型的 motion 与匹配标签，通过目标模型的接触约束微调 root/joints。它使用第 0 地形变体，不提供跨机器人映射，也不能替代完整全身自碰撞验收。

天工分支当前只输出新的 motion 和 terrain，**不复制或重新生成 pilot `_intent.npz`**，也没有完整的原始重定向溯源清单。因此微调结果不能直接串接现有 pilot 检查器当作完整复现流程；先补齐并核对接触意图及来源管理，再独立质检和重新标注。此功能属于后续开发集成点。

训练与策略评估单独开展，入口为 [train_stage1.py](../../supermimic/acp/train_stage1.py) 和 [evaluate.py](../../supermimic/acp/evaluate.py)。先阅读[历史站立参数](../tienkung3_29dof/php_pilot/evidence/standing_args.json)、[平地参数](../tienkung3_29dof/php_pilot/evidence/flat_args.json)及[停止记录](taskmemory.md)。已有参数出现过 OOM，不能直接当成已验收配置；本手册没有自动恢复训练步骤。

## 8. 常见问题

| 现象 | 检查与处理 |
|---|---|
| 找不到源动作或标签 | 当前源码固定读取 `data/php_parkour` 和 `data/php_parkour_contacts`；检查对应源文件名、类别和标签，不要用目标标签填补 |
| 找不到 intent | 检查 motion 同级 `_intent.npz`；目标标签不能替代 intent，微调输出还未自动维护它 |
| 模型身份/资产哈希不匹配 | 核对代码、generated 和 manifests 是否来自同一版本；恢复匹配资产或建立新版本，不编辑哈希绕过检查 |
| 标签过期 | 核对 motion/terrain 哈希，使用当前目标模型重新生成标签；输出到新目录 |
| 输出目录拒绝复用 | 更换 run ID；现有重定向器没有追加或恢复开关，保留原 manifest |
| 命令结束但没有片段 | 核对固定 `--only` 键、目录层级 `<根>/<类别>/<片段>_motion.npz` 和报告数量 |
| 初筛通过但严格失败 | 查看原始 self penetration 与碰撞对；当前两对模型重叠未解决，不自动启用例外 |
| 生成标签后仍无法跟踪 | 标签生成只计算几何，需分别核对动作质量、执行器和策略能力 |
| 渲染找不到 ffmpeg | 激活 conda 环境并检查 `command -v ffmpeg`；绝对 Python 路径不会自动配置外部程序 PATH |
| CUDA OOM | 保存任务配置与日志，定位显存峰值后缩小负载；训练 OOM 与离线重定向是不同运行链路 |

复现结束后按 [taskmemory 模板](taskmemory.md#追加记录模板)记录命令、版本和结果，并更新[资产清单](asset_list.md)。

## v3 已完成结果核验与索引复现

146 条重跑已完成且独立审计通过。以下命令只读取既有候选/标签并生成报告或索引，不重新求解：

```bash
python -m tools.report_php_conversion \
  data/tienkung3_29dof/motions/php_repair_v3_20261109 \
  --out data/tienkung3_29dof/motions/php_repair_v3_20261109/delivery_report.json

python -m tools.index_php_conversion_variants \
  data/tienkung3_29dof/motions/php_full_v2_20261109 \
  data/tienkung3_29dof/motions/php_repair_v3_20261109 \
  --out data/tienkung3_29dof/motions/php_repair_v3_20261109/accepted_variant_index.json
```

父批次顺序决定优先级：上例保留旧版 910 个通过项，再加入新版 904 个新增通过项。索引中的 `source_variant` 是原始变体编号，`export_slot` 是对应父批次已筛选 terrain/labels 中的槽位，不可混用。它不是直接可加载的合并训练数据目录。


## 运行中结果的覆盖快照

`index_php_conversion_variants` 和 `audit_php_variant_coverage` 默认拒绝仍在运行的父批次。显式传入 `--allow-running` 后，只收集已提交 `accepted` 状态、完成哈希与标签核验的结果；等待、求解、验收、拒绝和异常项均不能填补覆盖。快照会标记 `snapshot_of_running_parents: true`，它是核验期间已提交结果的覆盖下限，不是全库结束证明。

核验绑定原始 source/terrain/labels、完整帧区间、50 Hz、候选/报告、导出轨迹和地形编号；重复重跑不能增加原始槽位总数。`--require-all` 在未覆盖全部 2960 槽位时仍返回非零退出码，即使已生成有效快照。运行中 manifest 以一次读取的原始字节计算身份哈希，避免状态刷新造成读取内容与哈希不一致。


覆盖索引也接受只做验收的独立恢复目录（包含 `provenance.json`、`result.json` 和 `source_snapshot`，不需要伪造 `plan.json`）。工具会核验原父计划、原任务、父候选及恢复副本一致、验收代码快照，以及新报告、导出轨迹、地形编号和标签。恢复结果填补同一个原始 source/variant 槽位，不增加分母；原批次的历史failed标记保持不变。这类目录不能用 `convert_php --resume` 恢复求解。


## v21 根部连续性准入（2026-11-10）

正式导出要求几何、自由臂、躯干姿态及root-continuity-v1同时通过。根检查对原50Hz全段计算target-source根位置修正的速度RMSE/P95/最大值和加速度最大值，阈值分别0.5/1/2.5m/s及100m/s²；这些是工程连续性筛查，不是动力学可执行性证明。常量体型/高度平移允许，原源动作/帧/地形不改。

旧quality_pass并未包含根连续性。最终覆盖核验必须显式增加`--require-root-continuity --require-all`：

```bash
python -m tools.audit_php_variant_coverage <run1> <run2> --require-root-continuity --require-all --out <coverage.json>
```

运行中只看已提交结果时另加`--allow-running`，仍不表示全库完成。index工具也支持`--require-root-continuity`，会从原源与候选重新计算并排除跳变轨迹，保留原source/variant编号和全部原始分母。>=21的质量报告和plan必须声明根策略；独立audit重算评分与哈希绑定，不能只接受报告里的布尔值。


### 14:46 当前结果解释

正式版本仍v21；v22虽保持69/80但没有改善，未推广，v23为隔离运行中试验。地形小批29/40是重复检查数，实际补5原始source/variant槽位；扩展37组合仍完整时间与原全部地形。clearance probe与512帧witness trace只供诊断，无正式导出，不能直接并入训练覆盖。最新四条原版/修正版录像及前倾曲线链接见README当前状态，旧v3目录仍保留，不表示已被修复覆盖。


### 原生碰撞修复入口（v25正式；历史直接入口）

`tools.polish_php_conversion`用于复用已核验的完整候选。种子JSON为`[{"parent":"/absolute/path/to/run","task_id":"原task id"}]`；只使用已提交accepted/rejected任务。

```bash
python -m tools.polish_php_conversion --seeds seeds.json --out new_run --workers 2 --scan
python -m tools.polish_php_conversion --seeds seeds.json --out new_run --workers 2 --resume
python -m tools.audit_php_conversion new_run --out audit.json
```

该入口绑定父plan/motion/terrain/intent/quality与原始源文件，重算全部原始变体、接触标签、根连续性和其余门槛，不把局部优化器的收敛状态当作验收结论。v24允许受限根平移并精确保留根旋转，扩大样本已出现根加速度拒绝；拒绝样本不得导出为合格。v25隔离试验固定整个根位姿，必须从原根已合格的种子重新开始，不能固定一个已发生跳变的v24输出。

运行中的v21全库/地形与v25隔离任务必须使用它们各自冻结runner；不能因为主源码升级而以主入口直接resume旧任务。所有最终覆盖仍须独立audit/index及--require-root-continuity --require-all。


后台唯一执行与队列（2026-11-10）：先读取根AGENTS.md和php-guard/OPERATIONS.txt，并检查guard.py status。正常转换保持python -m tools.convert_php，经启动钩子登记。隔离v26正在验证统一入口 `python -m tools.convert_php --polish-seeds <seeds.json> --out <run> --workers-per-gpu 0 --cpu-workers 4 --scan`，实际提交加--resume去掉--scan；只在相应冻结runner内使用。CPU修复也明确登记worker预约，不直接启动绕过队列的polish控制器。排队不是失败，不重复启动同输出，取消使用guard cancel。正式主版本目前v25，旧冻结批次只能使用各自原入口/代码身份恢复。


2026-11-10 16:45：上述统一入口已随v26正式推广，可在主源码的新输出目录使用；CPU修复命令明确 `--polish-seeds <seeds.json> --workers-per-gpu 0 --cpu-workers 4`，实际运行去掉--scan并加--resume。旧直接tools.polish_php_conversion命令仅作为历史复现记录，当前新批次统一走受保护convert入口。不要用主v26恢复任何v21/v25冻结批次。新source/variant专项也先查guard和已提交质量回执，队列queued不代表失败。
