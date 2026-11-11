# 站立瞄准资产列表

核对日期：2026-11-11（Asia/Shanghai）。返回 [文档入口](README.md)；使用路线见 [plan.md](plan.md)。

路径均相对仓库根目录。状态区分为 **本地存在**、**历史记录存在但本地缺失**、**计划生成**；外部资料单列。运行资产的缺失路径只用代码文字列出，不伪装成已存在文件的链接。

新增 [详细课程目录](../../config/standing_aiming_curriculum/README.md) 已保存单 actor 控制、IK 监督、课程与接入验收设计；监督数据、瞄准权重及课程代码尚未生成。

## 1. 本地存在：源码与配置资料

| 资产 | 用途 |
|---|---|
| [模型准备脚本](../../../tools/prepare_tienkung3.py) | 固定上游版本、29 关节顺序、默认姿态与 PD 参数；本地运行模型已生成 |
| [关节预览脚本](../../../tools/preview_aiming_joints.py) | 双视角、固定组与可动组配色、末端三关节依次 ±45° 回放 |
| [RobotSpec](../../../supermimic/acp/robots.py) | 机器人身份、模型加载、控制周期和关节配置 |
| [FK](../../../supermimic/acp/kinematics.py) | 全树正运动学、`right_tcp_link` 位姿及后续指向计算 |
| [仿真环境](../../../supermimic/acp/env.py) | 关节目标执行、观测历史、站立状态与 reset |
| [任务与奖励](../../../supermimic/acp/task.py)、[MDP](../../../supermimic/acp/mdp.py) | 指令、PaperReward、参考跟踪、终止及 critic 特征 |
| [网络](../../../supermimic/acp/networks.py)、[replay](../../../supermimic/acp/buffer.py) | 全身动作输出、归一化、目标与动作数据流 |
| [Stage1 训练](../../../supermimic/acp/train_stage1.py) | 世界模型、策略和 critic 更新，想象 rollout |
| [持臂站立任务](../../../supermimic/acp/standing.py) | 统一动作覆盖、固定世界参考奖励及任务配置 |
| [持臂站立评估](../../../tools/evaluate_standing.py)、[结果导出](../../../tools/report_standing_finetune.py) | 加载检查点，连续 30 秒评估，比较图与录像 |
| [standing 参考生成器](../../../tools/generate_standing_motion.py) | 生成常值姿态参考；现有输出用途为接口检查，不能视为站立策略 |
| [评估](../../../supermimic/acp/evaluate.py)、[渲染](../../../supermimic/acp/render.py) | 后续加入瞄准指标、目标点和指向显示 |
| [机器人配置快照](../../config/tienkung3_29dof/evidence/robot.json) | 29 关节名字、默认角度、50 Hz 控制和 PD 参数的历史记录 |
| [资产来源快照](../../config/tienkung3_29dof/evidence/assets.json) | 上游 SHA、文件哈希、模型约定与 `self_collision=False` 记录 |
| [天工接入说明](../../config/tienkung3_29dof/README.md)、[设计](../../config/tienkung3_29dof/design.md) | 已有机器人接口和 FK 的背景 |

配置快照用于查阅，运行加载器实际读取 `data/tienkung3_29dof/`；不能把快照所在目录直接当作完整模型目录。持臂站立已接入，方向 IK 留待下一阶段。

## 2. 本地存在：可参考的历史证据

| 资产 | 使用方式 |
|---|---|
| [最新 Stage1／Stage2 诊断](../../report/20261111_stage1_stage2_analysis/README.md) | 判断初始化候选；优先于旧文档中的训练进度说明 |
| [同条件评估摘要](../../report/20261111_stage1_stage2_analysis/evidence/summary.json) | 核对历史数据，不作为新任务效果 |
| [Stage1 实际参数](../../report/20261111_stage1_stage2_analysis/evidence/training/stage1_args.json) | 查阅历史机器人、动作／标签目录和训练位置 |
| [Stage1 资产列表](../../stage1/asset_list.md) | 查找原训练产物；其中旧文档路径可能尚未随搬迁更新 |
| [接入阶段结果](../../config/tienkung3_29dof/progress.md) | FK、模型和合成站立链路的历史结果 |
| [小批量 PHP 试验](../../config/tienkung3_29dof/php_pilot/README.md) | 早期站立检查点的背景；不能当作已验收的站立初始化 |

最新报告的 ±2 秒窗口、动作噪声 0.1 协议下，Stage1 行走／低台阶失败数为 0/1000、8/560，Stage2 为 56/1000、31/560。这是参考数据上的历史评估，说明 Stage1 值得作为初始化候选，不代表新任务或未见目标上的成功率。

## 3. 本地运行资产与预览

2026-11-11 已创建 `acp_supermimic` 环境，并准备以下资产；数据和预览产物不纳入 Git。

| 本地路径 | 内容 |
|---|---|
| `data/tienkung3_29dof/generated/robot.xml` | 天工运行模型 |
| `data/tienkung3_29dof/manifests/robot.json`、`assets.json` | 运行配置与来源记录 |
| `data/tienkung3_29dof/motions/bootstrap/`、`contacts/bootstrap/` | 601 帧合成站立参考与接触标签；不是持臂训练成果 |
| [关节总览图](../../../results/analysis/standing_aiming/joint_overview.png) | 固定组、可动组与 TCP 指向轴 |
| [逐关节视频](../../../results/analysis/standing_aiming/joint_sweep.mp4) | 16.5 秒双视角运动学回放 |
| [预览配置](../../../results/analysis/standing_aiming/joint_definition.json) | 固定角度、可动关节限位、局部轴和初始指向 |
| `.cache/environment-check/` | 环境与 GPU 仿真验证记录 |
| `logs/imported/5090/stage1_fixed8_from100k_20261111/ckpt_100500.pt` | 已导入并验证的微调源权重 |
| `data/tienkung3_29dof/standing_aiming/` | 2001 帧／40 秒持臂参考及双脚接触标签 |
| `logs/tienkung3_29dof/standing_aiming_from100500_20261111/train/` | 当前本机微调配置、日志、检查点与周期评估 |
| `.cache/standing_aiming/` | 任务接线、短训练和保存加载验证 |

从仓库根目录重新生成预览：

```bash
conda activate acp_supermimic
python -m tools.preview_aiming_joints
```

## 4. 历史记录存在但本地缺失

以下历史训练资产尚未加载或复测。

| 历史资产路径 | 用途与获取方向 |
|---|---|
| `data/tienkung3_29dof/stage1/php_all_variants_20261110/` | 历史正式训练数据；首版站立不要求恢复全量 PHP 数据 |
| `logs/tienkung3_29dof/stage1_php_all_spheres224_20261110/train/ckpt_060000.pt` | 历史 Stage1 60k 初始化候选；从原训练资产获取后才能使用 |
| `logs/tienkung3_29dof/php_pilot_stand_v2/ckpt_000200.pt` | 早期探索性站立检查点；历史未完成策略评估，首版不默认使用 |

当前主线已取得 100500 权重和持臂参考，无需恢复以上历史资产即可微调。

## 5. 计划生成：首版产物

以下为当前导出目标及后续瞄准产物位置。是否已生成以 task memory 最新记录为准。

| 建议位置 | 内容 |
|---|---|
| `results/analysis/standing_aiming_finetune/` | 最佳与最终策略重新加载评估、漂移曲线、比较摘要与最佳录像 |
| `data/tienkung3_29dof/standing_aiming/ik_supervision/<dataset_id>/` | 拟新增的 IK 监督序列、教师配置、划分与质量摘要 |
| `logs/tienkung3_29dof/standing_aiming_curriculum/<run>/` | 拟新增的单 actor、WM／critic、训练辅助、课程及优化器状态 |
| `results/standing_aiming_curriculum/<run>/` | 拟新增的靶面与角误差摘要、实际轨迹和演示录像 |

首版保留一份实际配置、可加载权重、简短效果记录和录像即可，不预设大量独立证据文件。

## 6. 外部资料

| 来源 | 本任务用途 |
|---|---|
| [固定版本天工 URDF](https://github.com/Open-X-Humanoid/TienKung_URDF/blob/4c2c061d72dcdab60d74e45d3efb17246a3b6f62/tiangong3_urdf/urdf/tiangong3.urdf) | 已在前述调研中核对右臂关节链和限位；与模型准备脚本固定 SHA 对应 |
| [HOMIE](https://arxiv.org/html/2502.13013v2) | 参考身体策略适应上肢姿态变化的思路 |
| [HugWBC](https://arxiv.org/html/2502.03206v2) | 参考外部上肢接管和奖励冲突处理 |
| [Modern Robotics：数值 IK](https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-1-of-2/) | 末端方向求解的基础方法 |

外部资料用于理解控制方法；首版不要求引入这些项目的代码或训练框架。
