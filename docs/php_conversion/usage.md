# PHP 转换工具链使用手册

更新日期：2026-11-08。返回[文档入口](README.md)。本页命令根据当前源码核对，供后续主动运行；本轮文档整理没有执行下载、转换、标注、训练或渲染。

## 1. 环境与输入

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

## 2. 当前命令与能力边界

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
