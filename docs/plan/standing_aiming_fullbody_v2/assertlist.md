# 全身站立瞄准 v2：涉及资产列表

核对：2026-11-13（Asia/Shanghai）。实施计划：[plan.md](plan.md)；进度：[taskmemory.md](taskmemory.md)。文件名按用户要求保留为 `assertlist.md`，用途是资产清单。

本表区分**本地存在、历史证据、计划生成**。2026-11-13 代码实施后的实际资产更新见第 8—9 节；第 1—6 节的“本次核对”指之前的文档整理快照。路径相对仓库根目录说明，链接相对本文。存在性已在文档整理时核对；存在不等于已通过 v2 验证。未生成资产使用代码路径，不创建指向不存在文件的链接。本文没有重新加载权重或运行历史评估。

## 1. 机器人、物理参数与运动学

| 资产/路径 | 状态 | 用途与复用方式 | 限制 |
|---|---|---|---|
| [运行模型 data/tienkung3_29dof/generated/robot.xml](../../../data/tienkung3_29dof/generated/robot.xml) | 本地存在 | 复用自由浮动机器人与 29 维关节控制 | 解除动作覆盖不需要焊接/解焊基座或改变关节拓扑 |
| [运行配置 manifests/robot.json](../../../data/tienkung3_29dof/manifests/robot.json) | 本地存在 | 关节顺序、默认姿态、动作尺度、PD、力矩与速度参数、控制周期 | 运行时来源；速度参数不等于已实现目标软件限速 |
| [模型来源 manifests/assets.json](../../../data/tienkung3_29dof/manifests/assets.json) | 本地存在 | 模型版本、几何处理和资产身份 | 后续改变模型时需重新检查 checkpoint 兼容性 |
| [robots.py](../../../supermimic/acp/robots.py) | 本地存在 | RobotSpec、机器人维度、接触部位映射 | 按名称建立索引，避免硬编码分组 |
| [kinematics.py](../../../supermimic/acp/kinematics.py) | 本地存在 | 真实/预测状态的 FK，TCP、双脚和胸口基准计算 | 使用实际状态，不能用固定指令角替代实际角 |
| [terrain.py](../../../supermimic/acp/terrain.py) | 本地存在 | 平地和后续地形的高度扫描 | 平地扫描不等于全零；policy 与 WM 的扫描布局分别保留 |
| [模型准备说明](../../config/tienkung3_29dof/README.md) | 本地存在，背景资料 | 查阅机器人接入与资产来源 | 历史说明不能代替当前运行配置核对 |
| [指向轴与关节预览定义](../../../results/analysis/standing_aiming/joint_definition.json) | 本地存在，历史几何证据 | 复核 `right_tcp_link` 局部 `-Z` 与关节命名 | 旧固定组配色和末端三关节分组不适用于 v2 |

## 2. 当前控制与训练代码

以下为当前源码接入位置，不表示它们已实现 v2。

| 资产/路径 | 状态 | 用途与复用方式 | 必须调整的边界 |
|---|---|---|---|
| [env.py](../../../supermimic/acp/env.py) | 本地存在，已有修改 | 物理引擎、PD 下发、状态读取、实际转移 | v2 reset、终止、地形、回合计时需绕开参考动作依赖 |
| [mdp.py](../../../supermimic/acp/mdp.py) | 本地存在，已有修改 | 保留 99 维观测、状态转换、物理上下文及限位计算 | tracking reward、reference termination、reference critic 不用于 v2 |
| [task.py](../../../supermimic/acp/task.py) | 本地存在 | 接触部位顺序、坐标转换、站位指令与扫描约定 | 使用程序化脚位锚点，不读取逐帧接触标签；旧 3 维 goal 不是速度 |
| [networks.py](../../../supermimic/acp/networks.py) | 本地存在 | Policy、WM、Critic 与归一化基础 | actor 指令布局显式迁移；critic 重新初始化 |
| [buffer.py](../../../supermimic/acp/buffer.py) | 本地存在，已有修改 | 实际转移存储、历史恢复、窗口 mask | 增加并恢复目标/锚点/终止上下文，不能只扩展存储字段 |
| [standing.py](../../../supermimic/acp/standing.py) | 本地存在，旧任务 | 初始站姿常量、站立指标、冻结统计等可参考 | 不复用右臂固定动作覆盖和参考站立奖励 |
| [aiming.py](../../../supermimic/acp/aiming.py) | 本地存在，未跟踪源码 | TCP 几何、靶面射线指标、旧输入和权重布局 | 仍是 4 固定＋3 专用处理、110 维指令；不能直接当 v2 |
| [train_aiming.py](../../../supermimic/acp/train_aiming.py) | 本地存在，未跟踪源码 | WM 更新、可微想象、critic 拟合和保存机制 | 清理参考奖励/终止/critic/教师；核对 finetune 的 WM 来源 |
| [train_stage1.py](../../../supermimic/acp/train_stage1.py) | 本地存在，已有修改 | 通用模型训练和历史 standing 迁移实现参考 | 不把旧 reference frame 采样直接用于无参考任务 |
| [aiming_curriculum.py](../../../supermimic/acp/aiming_curriculum.py) | 本地存在，未跟踪源码 | 旧目标采样、窗口统计和验收实现参考 | 删除固定四关节指标；本阶段不启用动态目标课程 |

## 3. 评估、数据准备与运行工具

| 资产/路径 | 状态 | 用途与复用方式 | 限制 |
|---|---|---|---|
| [evaluate_aiming.py](../../../tools/evaluate_aiming.py) | 本地存在，旧评估 | 完整 30 s 轨迹与指标输出方式 | 旧固定关节指标和动态场景不能原样用于 v2 |
| [render_aiming.py](../../../tools/render_aiming.py) | 本地存在 | 目标、射线、靶面与轨迹录像参考 | 需支持新任务元数据与统一动作 |
| [export_aiming.py](../../../tools/export_aiming.py) | 本地存在 | actor 导出参考 | 去掉固定四维、terminal_speed 和专用三维历史假设 |
| [test_aiming.py](../../../tools/test_aiming.py) | 本地存在，旧测试 | 了解旧接线验证 | 固定行零梯度、三维限速等是旧任务预期，v2 需独立测试 |
| [evaluate_standing.py](../../../tools/evaluate_standing.py) | 本地存在 | 旧持臂站立基线复测入口 | 旧右臂固定角度指标不适用于 v2 |
| [prepare_aiming.py](../../../tools/prepare_aiming.py) | 本地存在 | 旧 IK 数据与监督迁移背景 | 只训练末端三自由度，不作为 v2 必经步骤 |
| [prepare_chest_aiming.py](../../../tools/prepare_chest_aiming.py) | 本地存在 | 胸口水平目标几何和旧迁移背景 | 复用几何思路，不能继承固定关节约束 |
| [validate_aiming_supervision.py](../../../tools/validate_aiming_supervision.py) | 本地存在 | 历史监督质量证据的生成方式 | 监督通过不是 v2 动力学成功 |
| [watch_aiming.py](../../../tools/watch_aiming.py)、[check_aiming_health.py](../../../tools/check_aiming_health.py) | 本地存在，旧服务使用 | 识别现有运行与健康监控 | 本次不修改服务、不启动第二个训练控制器 |

## 4. 已有检查点、配置与运行身份

大文件通常位于 Git 忽略目录，只在本机可用。**本次核对存在性，没有给运行中的文件建立不可变快照；正式迁移前需记录副本和哈希。**

| 资产/路径 | 状态 | 用途与复用方式 | 限制 |
|---|---|---|---|
| [持臂站立 best.pt](../../../logs/tienkung3_29dof/standing_aiming_from100500_20261111/train/best.pt) | 本地存在，历史初始化资产 | 已有站立 actor/WM，作为回归基线与独立备选 | 全右臂七维曾被固定；不能直接认为七维输出可用 |
| [站立 args.json](../../../logs/tienkung3_29dof/standing_aiming_from100500_20261111/train/args.json)、[task_config.json](../../../logs/tienkung3_29dof/standing_aiming_from100500_20261111/train/task_config.json) | 本地存在 | 核对旧来源、物理参数和历史实验设置 | 是固定持臂任务，不是 v2 配置 |
| [胸前瞄准 train/best.pt](../../../logs/tienkung3_29dof/standing_aiming_curriculum/chest_forward_20261112/train/best.pt) | 本地存在，后台可能更新 | v2 优先验证的 actor/WM 迁移候选 | 110 维指令、四行固定；需按字段迁移并固定来源版本 |
| [胸前瞄准 train/last.pt](../../../logs/tienkung3_29dof/standing_aiming_curriculum/chest_forward_20261112/train/last.pt) | 本地存在，持续更新 | 旧任务恢复与近期状态参考 | 不作为未标版本的 v2 源，不直接 resume 成新任务 |
| [运行 config.json](../../../logs/tienkung3_29dof/standing_aiming_curriculum/chest_forward_20261112/config.json)、[train/config.json](../../../logs/tienkung3_29dof/standing_aiming_curriculum/chest_forward_20261112/train/config.json) | 本地存在，运行资产 | 核对旧任务实际配置；checkpoint 内配置也需对照 | 旧任务目前无迭代上限，v2 初次实验另设预算 |
| [migration.json](../../../logs/tienkung3_29dof/standing_aiming_curriculum/chest_forward_20261112/migration.json) | 本地存在，历史迁移记录 | 查旧胸前目标迁移来源 | 不是 v2 迁移报告 |
| [status.json](../../../logs/tienkung3_29dof/standing_aiming_curriculum/chest_forward_20261112/status.json) | 本地存在，持续变化 | 查询旧任务最新状态 | 不复制其中轮数为 v2 进展，不把快照当永久当前状态 |
| [导入的 Stage1 ckpt_100500.pt](../../../logs/imported/5090/stage1_fixed8_from100k_20261111/ckpt_100500.pt) | 本地存在，历史来源 | 追溯站立微调的原始来源 | 不是默认新任务起点，旧历史能力需按对应协议解释 |

核对时用户级 `aiming-training.service` 为 `active/running`，入口是 `tools.watch_aiming --run-dir .../chest_forward_20261112`。服务存在与训练运行不代表新任务已经启动。

## 5. 旧参考、接触、监督数据和几何

| 资产/路径 | 状态 | 用途与复用方式 | 限制 |
|---|---|---|---|
| [standing_motion.npz](../../../data/tienkung3_29dof/standing_aiming/motions/synthetic/standing_motion.npz) | 本地存在，旧常值参考 | 查旧 reset 姿态与历史训练来源 | v2 用初始状态常量和锚点，不读取其逐帧轨迹作监督 |
| [standing 接触标签](../../../data/tienkung3_29dof/standing_aiming/contacts/synthetic/standing.npz)、[标签摘要](../../../data/tienkung3_29dof/standing_aiming/contacts/summary.json) | 本地存在 | 查旧双脚接触定义 | v2 程序化生成双脚目标，无需这套标签文件 |
| [IK 监督数据目录](../../../data/tienkung3_29dof/standing_aiming/ik_supervision/20261111_c0/) | 本地存在，历史数据 | 含监督集、增广、配置与旧学生检查点 | 三维 IK 标签不是 v2 全右臂必需监督来源 |
| [监督集描述 dataset.json](../../../data/tienkung3_29dof/standing_aiming/ik_supervision/20261111_c0/dataset.json)、[监督验证 best.validation.json](../../../data/tienkung3_29dof/standing_aiming/ik_supervision/20261111_c0/best.validation.json) | 本地存在，历史证据 | 区分数据身份与监督结果 | 不能将监督成绩换算成自由全身物理验收 |
| [旧 IK target_pool.pt](../../../data/tienkung3_29dof/standing_aiming/ik_supervision/20261111_c0/target_pool.pt) | 本地存在 | 旧目标可达性背景 | 不限制 v2 只能在旧三关节可达域学习 |
| [胸前 target_pool.pt](../../../logs/tienkung3_29dof/standing_aiming_curriculum/chest_forward_20261112/target_pool.pt)、[geometry.json](../../../logs/tienkung3_29dof/standing_aiming_curriculum/chest_forward_20261112/geometry.json) | 本地存在，旧任务几何 | 胸口中心、水平前向、中央目标的几何参考 | 旧可达性检查仍基于三自由度；v2 要检查完整右臂 |

新世界模型训练数据来自 `(S_t, a_exec, S_next)` 实际物理转移。以上 motion、接触标签和 IK 数据均不应成为构建 v2 环境和训练 WM 的必要外部输入。

## 6. 历史文档与评估证据

| 资产/路径 | 状态 | 参考内容 | 适用边界 |
|---|---|---|---|
| [旧计划](../standing_aiming/plan.md)、[旧 taskmemory](../standing_aiming/taskmemory.md)、[旧验收](../standing_aiming/validation.md) | 本地存在，历史方案 | 用户需求演变、固定持臂站立与课程训练记录 | 最新自由右臂要求以本目录为准；不改写历史 |
| [旧课程设计](../../config/standing_aiming_curriculum/design.md)、[课程](../../config/standing_aiming_curriculum/curriculum.md)、[运行记录](../../config/standing_aiming_curriculum/run.md) | 本地存在，旧实现说明 | 输入布局、连续奖励、数据和运行证据 | 旧固定关节、IK 初始化、动态课程不能原样继承 |
| [持臂站立独立评估 best.json](../../../results/analysis/standing_aiming_finetune/best.json) | 本地存在，历史通过证据 | 固定右臂下的 30 s 站立结果 | 本次未复测，不属于 v2 通过证据 |
| [胸前新靶位 baseline.json](../../../results/standing_aiming_curriculum/chest_forward_20261112/baseline.json)、[轨迹](../../../results/standing_aiming_curriculum/chest_forward_20261112/baseline.npz)、[录像](../../../results/standing_aiming_curriculum/chest_forward_20261112/baseline.mp4) | 本地存在，历史失败基线 | 旧策略迁移新靶位前站立通过、瞄准未通过 | 不是当前后台最新成绩，也不是 v2 基线 |
| [Stage1/Stage2 历史诊断](../../report/20261111_stage1_stage2_analysis/README.md) | 本地存在，历史报告 | 初始化来源与旧能力背景 | 参考动作任务上的结果不等于新瞄准任务能力 |

## 7. v2 计划生成的资产

以下路径统一以新任务身份分开管理；本表已按 2026-11-13 实际实施更新。`<run_id>` 在实际启动时使用日期时间与实验标识，避免与旧实验混用。

| 计划路径 | 当前状态 | 生成阶段与用途 | 约束 |
|---|---|---|---|
| `supermimic/acp/aiming_fullbody.py` | 已实现并通过接线测试 | M1—M4：任务上下文、统一动作、107 维指令、奖励与终止 | 与旧任务保持独立语义 |
| `supermimic/acp/train_aiming_fullbody.py` | 已实现并完成短闭环/恢复/容量检查 | M2—M5：迁移、实际采集、WM、actor 与 critic 更新 | 明确源 checkpoint，禁止旧任务直接 resume |
| `tools/evaluate_aiming_fullbody.py` | 已实现，初始 A 基线评估失败并留档 | M4/M6：完整静态目标评估与证据输出 | 无固定肩肘指标，无隐藏教师执行 |
| `docs/config/standing_aiming_fullbody_v2/default.json` | 已生成 | M1—M4：正式配置初值 | 与 plan 的输入、奖励、课程和版本对应 |
| `logs/tienkung3_29dof/standing_aiming_fullbody_v2/<run_id>/source/` | 两类源快照已生成 | M2：固定源 checkpoint 副本、哈希、迁移报告 | 不引用持续变化且未标版本的源文件 |
| `logs/tienkung3_29dof/standing_aiming_fullbody_v2/<run_id>/config.json` | 已生成 | M4 起：实际运行配置 | 保存实际生效值，不只保存默认配置 |
| `logs/tienkung3_29dof/standing_aiming_fullbody_v2/<run_id>/train/` | init 已生成；短训练在 smoke/capacity 目录；正式权重未生成 | M4—M6：初始化、best/last/checkpoint、优化器和课程状态 | 首轮新建 critic 与计数；后续仅同版本 resume |
| `results/standing_aiming_fullbody_v2/<run_id>/wm_validation/` | 初始诊断已生成；正式适配待执行 | M5：真实轨迹与 1/8/想象长度预测误差 | 与训练轨迹按 episode 分离，记录物理单位 |
| `results/standing_aiming_fullbody_v2/<run_id>/evaluation/` | 初始失败基线、轨迹与录像已生成 | M6：静态场景 JSON、NPZ 轨迹、曲线与视频 | 保留完整 30 s 及失败场景，不删起始调整段 |
| `results/standing_aiming_fullbody_v2/<run_id>/export/` | 初始 actor 导出/重载已验证；未获部署资格 | M6 后：actor、输入布局与控制元数据 | 不依赖世界模型、critic、IK 或固定关节覆盖执行 |

文档整理本身的验收记录保存在 [taskmemory.md](taskmemory.md)。未来每项资产实际生成后再更新状态、实际路径与证据；训练失败也保留配置和输出，不把“已生成权重”写为“任务成功”。

## 8. 2026-11-13 实际生成与检查结果

- [运行说明与正式参数](../../config/standing_aiming_fullbody_v2/README.md)、[默认配置](../../config/standing_aiming_fullbody_v2/default.json)。
- [准备检查总览](../../../results/standing_aiming_fullbody_v2/preflight_20261113/preflight.json)、[最终代码身份](../../../results/standing_aiming_fullbody_v2/preflight_20261113/code_identity.json)。
- [默认源迁移报告](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/preflight_20261113/source/migration.json)、[默认初始化](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/preflight_20261113/train/init.pt)。
- [站立源迁移报告](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/preflight_20261113_standing/source/migration.json)、[备选初始化](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/preflight_20261113_standing/train/init.pt)。
- [8 环境短闭环](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/preflight_20261113/smoke_final/completion.json)、[恢复验证](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/preflight_20261113/smoke_resume/completion.json)、[512 环境短容量检查](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/preflight_20261113/capacity/completion.json)。其中 `passed=false` 表示任务未验收，不表示程序运行失败。
- [胸前源 30 秒基线](../../../results/standing_aiming_fullbody_v2/preflight_20261113/evaluation/initial_A.json)、[完整轨迹](../../../results/standing_aiming_fullbody_v2/preflight_20261113/evaluation/initial_A.npz)、[录像](../../../results/standing_aiming_fullbody_v2/preflight_20261113/evaluation/initial_A.mp4)、[误差曲线](../../../results/standing_aiming_fullbody_v2/preflight_20261113/evaluation/initial_A.png)。
- [站立源 30 秒基线](../../../results/standing_aiming_fullbody_v2/preflight_20261113_standing/evaluation/initial_A.json)、[七关节离线可达性检查](../../../results/standing_aiming_fullbody_v2/preflight_20261113/reachability.json)。
- [actor 导出元数据](../../../results/standing_aiming_fullbody_v2/preflight_20261113/export/runtime.json)、[契约与物理测试日志](../../../results/standing_aiming_fullbody_v2/preflight_20261113/checks/contracts.log)。

正式 M5 数据适配、课程训练、最终 27 点联合物理通过及合格部署权重仍未生成。资产存在与短测试通过不能替代这些结果。


## 9. 2026-11-13：32 步正式运行资产

正式训练已启动，M5 数据适配进行中，M6 物理验收尚未完成。第 8 节为启动前记录。

- [正式启动身份](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113/launch.json)、[生效参数](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113/train/config.json)、[运行代码身份](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113/train/code_identity.json)。
- [控制台日志](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113/stdout.log)、[最近进度](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113/train/status.json)、[详细指标](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113/train/metrics.jsonl)。
- [10 项契约与物理/梯度测试](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113/checks/contracts.log)、[512 环境 H32 容量短测试](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113/checks/capacity/completion.json)、[保存恢复短测试](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113/checks/resume/completion.json)。
- [正式第 100 轮独立 WM 诊断](../../../logs/tienkung3_29dof/standing_aiming_fullbody_v2/h32_20261113/train/wm_validation/000100.json)，包含 1/8/16/32 步，诊断不阻塞优化器更新。

服务为 `aiming-fullbody-v2.service`，旧固定臂服务已停止，旧产物保留。正式 checkpoint 每 500 轮保存；最终可部署权重尚未验收。
