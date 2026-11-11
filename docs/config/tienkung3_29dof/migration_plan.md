**天工 3.0 29 DoF 完整仿真迁移计划**

日期：2026-11-07。本文为原始完整迁移计划；用户已批准先实施 M0–M3 + 合成站立链路。实时实施状态以 [progress.md](progress.md) 为准，恢复上下文见 [taskmemory.md](taskmemory.md)。

依据：[G1 → 天工 3.0 迁移调研](../tienkung3_migration_analysis.md)。本计划收敛该报告中的候选路线，确定以天工 29 DoF 完成 ACP 三阶段仿真迁移。文中的工程门槛是后续验收目标，不是已经获得的实验结果。

本文最初编写时仅交付计划，以下命令当时均未执行。后续实施的实际接口和已完成证据见 [README.md](README.md) 与 progress.md；不要将完整 M0–M7 目标视为本次全部交付。

**1. 目标、范围与固定决策**

最终目标：在保留 G1 可运行的前提下，使天工 3.0 29 DoF 在现有 mjlab / MuJoCo Warp + ACP / SuperMimic 中完成参考接触跟踪、接触规划和程序地形目标到达。首个可运行里程碑是使用最终 29 DoF 模型完成平地 Stage 1 闭环。

| 决策项 | 本计划采用的方案 |
|---|---|
| 开发分支 | `feat/tienkung3-29dof-acp`，从实施时已核对的本地 `main` 创建 |
| 机器人标识 | `tienkung3_29dof`；保留 `g1` 为现有默认配置 |
| 机器人来源 | `Open-X-Humanoid/TienKung_URDF` 的天工 3.0 URDF 与配套网格 |
| 模型提交 | `4c2c061d72dcdab60d74e45d3efb17246a3b6f62`，不用浮动的 latest/main 代替 |
| 自由度 | 双腿 12 + 腰部 3 + 双臂 14 = 29；不计 freejoint |
| 头部处理 | head yaw/pitch 在零位固定，保留 link、质量、惯量和几何 |
| 仿真后端 | 保留 mjlab / MuJoCo Warp，不迁移旧 Isaac Lab 后端 |
| 初始频率 | 策略 50 Hz；物理 200 Hz，4 子步；必要时验证物理 400 Hz、8 子步 |
| 动作路线 | 官方平地动作适配 → 现有 PHP G1 动作的任务空间重定向 → 接触清理与重新标注 |
| 训练初始化 | 天工 Stage 1 从头训练；Stage 2/3 仅继承前阶段的天工模型 |
| 兼容原则 | 不直接加载 G1 权重、归一化统计、replay buffer 或接触缓存 |
| 首批任务顺序 | 站立/平地步行 → 低台阶 → 高台阶 → 双手支撑 → 攀爬 |
| 正式范围 | Stage 1–3、数据工具、仿真评估与可复现记录 |
| 后续范围 | 实机通信、在线感知、推理部署和实机验收，单独立项 |

不增加 23 DoF 正式实现分支。平地动作中的末端关节可以保持默认姿态，但模型仍有完整 29 个受控关节。

模型机械参数以[冻结 URDF][urdf]为准；[TienKung-Lab 配置][lab-robot]提供控制起点，[xMimic 29 DoF 配置][xm-robot]提供新增末端关节控制参考。控制参数的引用不意味着替换冻结模型的机械结构。

**2. M0：分支、工作区与基线管理**

前置条件：尚未对项目进行迁移改造，现有 G1 数据和训练结果保留原位置。当前检查到本地分支为 `main`、HEAD 短哈希为 `751aa5c`；实际实施时重新核对，不把本文记录当成实时状态。

当前工作区已有 README、训练/渲染文件修改，以及环境、下载、检查、流水线脚本和调研报告等未跟踪文件。创建分支会保留这些修改，但不会自动把它们纳入提交。

以下是**实施时使用的现有 Git 命令**，从仓库根目录执行：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git diff --stat
git diff --cached --stat
git branch --list feat/tienkung3-29dof-acp
```

确认当前为 `main`、目标分支不存在且 HEAD 与记录一致后：

```bash
git switch -c feat/tienkung3-29dof-acp
git status --short
```

操作约束：

- 不执行 reset、clean、自动 stash、强制切分支或自动更新 main。
- 目标分支已存在时先检查其起点和内容；不删除或覆盖它。当前已在目标分支时继续核验即可。
- 当前不是 main 时，不擅自把其他分支的历史当作 main；先报告分支状态偏差。
- 记录现有文件列表、已暂存/未暂存 diff 和相关文件哈希。基线记录必须覆盖实际工作树，不能只记录 HEAD。
- 迁移提交使用明确文件路径或 `git add -p` 分块暂存，不使用 `git add .` 混入已有修改；保留用户原有暂存状态。
- 若已有未提交修改是迁移依赖，把对应基线差异单独列入复现说明，后续单独处理，不假装分支 HEAD 已包含这些依赖。

记录环境：Python 3.11、torch 2.9.0+cu128、mjlab 1.6.0、MuJoCo/MuJoCo Warp 3.11.0、Warp 1.16.0，并保存实际安装版本、GPU 型号与驱动信息。[环境配置](../../environment.yml)

在可用 GPU 和数据齐备时，执行现有 G1 检查，作为迁移前基线；这是**已有接口**，不接受 `--robot`：

```bash
python -m tools.check_acp_setup --num-envs 16 --steps 16
```

保存 G1 的短回放与检查结果。若缺少可用 checkpoint，只记录环境与数值基线，明确策略回归尚缺基线，不能把未运行评估写成通过。

交付物：分支起点、工作区差异记录、环境信息、G1 检查结果。验收：原有改动未丢失，新增文档与后续迁移提交边界明确。失败处理：先解决基线检查或工作区冲突，不进入资产替换。

**3. M1：冻结资产并生成 29 DoF 模型**

前置条件：M0 通过。资产来自冻结提交中的 `tiangong3_urdf/urdf/tiangong3.urdf`，以及其实际引用的网格；同时保留上游许可和来源记录。原始下载内容只读保存，转换写入派生目录。

规划目录如下；这些是未来资产/数据位置，本次不创建：

```text
data/tienkung3_29dof/
  upstream/                 # 原始 URDF、网格与来源
  generated/                # 固定头部的派生 URDF、专用 MJCF
  manifests/                # 资产清单、参数来源和哈希
  motions/bootstrap/        # 初期平地动作
  motions/php_retargeted/    # 分类别重定向动作
  contacts/bootstrap/       # 与动作和模型匹配的标签
  contacts/php_retargeted/
  splits/                   # 固定训练、验证、测试划分
logs/tienkung3_29dof/<run_id>/
results/tienkung3_29dof/<run_id>/
```

保持现有 `data/`、`logs/` 忽略规则。下载/转换代码、配置与小型复现 manifest 纳入版本控制；大网格、NPZ、checkpoint 和训练缓存不加入源码提交。每次实验保留模型、数据及配置的哈希清单。

转换要求：

1. 原始模型应为 31 个 revolute；仅将 `head_yaw_joint`、`head_pitch_joint` 改为 fixed，保持零位下的原始 joint origin。
2. 不删除头部或传感器 link，不损失其质量和惯量；固定 link 若合并，必须正确传递惯性和保留所需 frame。
3. 保留 pelvis 浮动基座，生成专用 MJCF。禁止直接使用 Lab 的 31 电机 `tiangong3_torq.xml` 代替。
4. 资产 MJCF 不内嵌驱动；由统一 RobotSpec/mjlab 构造 29 个位置执行器，避免重复添加。
5. 解析并验证网格路径、单位、关节限位、碰撞/视觉标记；仿真、FK、标注和渲染使用同一派生模型及变换规则。
6. 资产 manifest 保存上游 SHA、原始/派生 URDF 与网格哈希、转换器版本、MuJoCo 版本、锁定关节、canonical joint order 和模型哈希。

**3.1 完整 canonical joint order**

下表为策略、动作文件和配置采用的零基索引。模拟器内部顺序通过名称显式映射，不假设与表格天然相同。

| 索引 | 关节名 | 部位 |
|---:|---|---|
| 0 | hip_pitch_l_joint | 左腿 |
| 1 | hip_roll_l_joint | 左腿 |
| 2 | hip_yaw_l_joint | 左腿 |
| 3 | knee_pitch_l_joint | 左腿 |
| 4 | ankle_pitch_l_joint | 左腿 |
| 5 | ankle_roll_l_joint | 左腿 |
| 6 | hip_pitch_r_joint | 右腿 |
| 7 | hip_roll_r_joint | 右腿 |
| 8 | hip_yaw_r_joint | 右腿 |
| 9 | knee_pitch_r_joint | 右腿 |
| 10 | ankle_pitch_r_joint | 右腿 |
| 11 | ankle_roll_r_joint | 右腿 |
| 12 | waist_yaw_joint | 腰部 |
| 13 | waist_roll_joint | 腰部 |
| 14 | waist_pitch_joint | 腰部 |
| 15 | shoulder_pitch_l_joint | 左臂 |
| 16 | shoulder_roll_l_joint | 左臂 |
| 17 | shoulder_yaw_l_joint | 左臂 |
| 18 | elbow_pitch_l_joint | 左臂 |
| 19 | elbow_yaw_l_joint | 左臂 |
| 20 | wrist_pitch_l_joint | 左臂 |
| 21 | wrist_roll_l_joint | 左臂 |
| 22 | shoulder_pitch_r_joint | 右臂 |
| 23 | shoulder_roll_r_joint | 右臂 |
| 24 | shoulder_yaw_r_joint | 右臂 |
| 25 | elbow_pitch_r_joint | 右臂 |
| 26 | elbow_yaw_r_joint | 右臂 |
| 27 | wrist_pitch_r_joint | 右臂 |
| 28 | wrist_roll_r_joint | 右臂 |

最终裸模型应满足 `nq=36`、`nv=35`，包含 1 个 freejoint 和 29 个 hinge；组装训练实体后 `nu=29`。头部没有活动关节，不能用“不给动作”的方式让头部被动摆动。

**3.2 初始控制参数与冲突处理**

以下为待验证的仿真起点，不是硬件额定值。主体 kp/kd 参考冻结 Lab 配置，新增末端 kp/kd 参考冻结 xMimic 配置；effort 采用参考配置与目标 URDF 两者较小值。上下限、惯量和几何保持目标 URDF。

| 关节类别 | kp | kd | 初始 effort 上限 N·m | 来源/处理 |
|---|---:|---:|---:|---|
| hip_pitch / hip_roll | 300 | 10 | 200 | Lab 控制配置，低于 URDF 235 |
| hip_yaw | 150 | 5 | 142 | Lab 控制配置，低于 URDF 150 |
| knee_pitch | 330 | 10 | 330 | Lab 控制配置，低于 URDF 400 |
| ankle_pitch | 30 | 2.5 | 55 | Lab 为 100，按目标 URDF 55 收紧 |
| ankle_roll | 16.8 | 1.4 | 50 | Lab 控制配置，低于 URDF 55 |
| waist_yaw | 400 | 5 | 91 | Lab 控制配置，低于目标 URDF 200 |
| waist_roll / waist_pitch | 400 | 10 | 91 | Lab 控制配置，低于目标 URDF 150 |
| shoulder_pitch | 150 | 5 | 90 | Lab 与目标 URDF 一致 |
| shoulder_roll | 50 | 2.5 | 60 | Lab 控制配置，低于目标 URDF 90 |
| shoulder_yaw | 50 | 2.5 | 36 | Lab 控制配置，低于目标 URDF 50 |
| elbow_pitch | 150 | 5 | 50 | Lab 为 60，按目标 URDF 50 收紧 |
| elbow_yaw | 50 | 5 | 40 | xMimic 29 DoF 参考与目标 URDF |
| wrist_pitch / wrist_roll | 20 | 2 | 40 | xMimic 29 DoF 参考与目标 URDF |

动作约定为 `q_target = q_default + joint_offset + 0.25 * action`，不使用 G1 的 `0.25 * effort / kp` 公式。实际执行器和可微 reward 中的力矩估计必须使用相同参数及饱和规则；任何新增目标限幅也要同步。

速度参考上限同样取目标 URDF 与对应参考控制配置的较小值，逐关节记录。当前 MuJoCo 配置不提供等价的硬速度约束，速度表用于惩罚和验收，不宣称已经物理强制执行。

不得沿用 G1 电机 armature。源文件未提供可追溯值时，初始显式设为 0 并标记为未辨识的仿真假设；后续调整记录实验依据。区分被动关节阻尼和 PD 的 kd，避免重复计入。

默认姿态采用：双腿 hip_pitch=-0.15、knee_pitch=0.30、ankle_pitch=-0.15；双侧 shoulder_pitch=0.20、左 shoulder_roll=0.10、右 shoulder_roll=-0.10、双侧 elbow_pitch=-0.50，其余关节为 0，单位 rad。站立 root 高度通过该姿态的足底最低点解算，使脚底接近地面；不固定照抄 G1 的 0.76/0.78 m 或 Lab 的 0.97 m。

交付物：派生 URDF/MJCF、manifest、参数来源与冲突表、静态模型检查。验收：关节数量和上述顺序正确，头部质量保留，总质量相对原模型变化不超过 1e-4，惯量有效，默认站姿无明显穿透。失败处理：修正转换和几何，禁止以放宽动力学限制掩盖模型错误。

**4. M2：统一机器人接口与兼容性**

前置条件：M1 资产检查通过。新增 RobotSpec 与机器人注册入口，构造时传入配置，不通过修改进程级全局变量切换机器人。

RobotSpec 至少承载：robot_id、资产/模型版本、MJCF 路径、29 关节及内部排列映射、固定关节、root/torso、tracked/contact frames、默认姿态、执行器与动作缩放、控制周期、数据位置和任务尺度配置。

统一规则：

- `root_body="pelvis"`，`torso_body="waist_pitch_link"`；CoM 随机化使用配置指定躯干，去除 `torso_link` 的硬绑定。
- 训练、动作修正、标注、检查、评估、渲染及流水线均增加 `--robot`；兼容现有 G1 默认调用。
- 训练显式指定机器人；评估/渲染优先读取 checkpoint/trajectory 的机器人元信息，显式参数冲突时报错。旧文件无机器人信息时仅按兼容 G1 路径处理，不能自动推断成天工。
- 参数化 Policy、WorldModel、Critic、buffer 与 MDP 的维度；通过名称重排数据，不以两个机器人同为 29 维判定兼容。
- 初始化、resume、数据加载分别核对 robot_id、模型/执行器配置哈希、joint order、control_dt、数据版本；失败时明确指出不匹配项。
- Stage 1 天工禁止用 G1 `--init/--resume`。Stage 2/3 可以且必须使用通过验收的上一阶段天工 checkpoint；“从头训练”不意味着各阶段丢弃天工已有学习结果。
- 训练流水线读取数据集数量，去除固定 296 clips 和 PHP 路径假设；完成标记绑定输入数据/模型/命令，输入变化时不能沿用旧标记。

天工 29 DoF 初始维度：state=71、xi=33、tracker 单帧本体观测=99、planner 单帧本体观测=93、世界模型输出=70。维度仍从配置推导，不重新写成常量。

tracked bodies 固定为以下 14 个语义 frame，顺序用于参考跟踪：

```text
pelvis,
hip_roll_l_link, knee_pitch_l_link, ankle_roll_l_link,
hip_roll_r_link, knee_pitch_r_link, ankle_roll_r_link,
waist_pitch_link,
shoulder_roll_l_link, elbow_pitch_l_link, left_tcp_link,
shoulder_roll_r_link, elbow_pitch_r_link, right_tcp_link
```

交付物：通用机器人接口、天工注册配置、CLI 和元信息校验。验收：G1 与天工都能独立初始化，G1 旧调用和旧 checkpoint 回归通过，故意输入错模型/错顺序可被拒绝。失败处理：定位剩余硬编码，不能仅删除原来的 assert 来绕过校验。

**5. M3：可微运动学与接触一致性**

前置条件：M2 通过。修改可微 FK，正确处理全连杆树、固定链、joint anchor、局部旋转和速度传播；不能跳过固定 link 后仍假设活动父节点直接相连。

九个语义接触通道保持原顺序，初始映射如下：

| 顺序/语义 | 目标 frame | 初始碰撞几何归属 |
|---|---|---|
| 0 left_foot | ankle_roll_l_link | 该 link 的足部碰撞几何 |
| 1 right_foot | ankle_roll_r_link | 该 link 的足部碰撞几何 |
| 2 left_knee | knee_pitch_l_link | 该 link 的膝/小腿碰撞几何 |
| 3 right_knee | knee_pitch_r_link | 该 link 的膝/小腿碰撞几何 |
| 4 left_hand | left_tcp_link | TCP 上的末端碰撞几何 |
| 5 right_hand | right_tcp_link | TCP 上的末端碰撞几何 |
| 6 left_elbow | elbow_pitch_l_link | 该 link 的肘/前臂碰撞几何 |
| 7 right_elbow | elbow_pitch_r_link | 该 link 的肘/前臂碰撞几何 |
| 8 pelvis | pelvis | pelvis 碰撞几何 |

该映射针对选定的独立 URDF，不沿用 Lab 23 DoF 的碰撞缺省。TCP 原点不必位于掌面，接触点必须由其几何确定。若补充固定子链几何，累积整个固定变换；不得将跨活动 wrist/elbow_yaw 的几何当成静态偏移。

工作内容：

- 每个接触组显式验证几何非空，检查脚底和掌面实际覆盖范围；除九类以外的碰撞归入 other_contact。
- 将 mesh、sphere、capsule、cylinder、box 的碰撞近似和标注退化回退处理统一起来，避免新模型几何类型触发未支持分支。
- 初始沿用现有机器人与地形碰撞策略；离线重定向和最终评估额外检查非相邻 link 自碰撞，不能把“仿真关闭自碰撞”当作动作无自碰撞的证据。物理自碰撞开关和排除对均写入配置及实验记录。
- 仿真、离线标签与可微 reward 使用同一模型几何；修改几何后重新生成标签和模型哈希。
- 随机姿态对照 MuJoCo 的位置、旋转、线速度和角速度；对 FK/接触距离使用有限差分检查梯度，避开最近点切换等不可微边界。

建议验收门槛：1,000 个关节限位内随机姿态，位置最大误差 ≤1e-4 m，旋转角最大误差 ≤1e-4 rad，线/角速度最大误差分别 ≤1e-3 m/s、1e-3 rad/s；平滑区域梯度采用相对容差 1e-2、绝对容差 1e-3。接触近似在距离表面 ±5 cm 的样本中，距离误差 P95 ≤5 mm、最大 ≤15 mm，并单列脚底/掌面误差。

这些是工程目标；若 mesh 球近似达不到要求，先增加采样或显式支撑几何并重新比较，不通过直接扩大接触阈值宣布通过。

交付物：通用 FK/contact 实现、对照检查与可视化。失败处理：优先定位坐标/固定链错误，再处理几何近似精度；未通过前不启动学习。

**6. M4：动作转换、重定向与标签**

前置条件：M3 通过。所有动作和标签携带 robot_id、模型哈希、joint order、来源、fps 和转换版本；配套 manifest 保存文件哈希、重定向参数及划分信息。

**6.1 统一文件约定**

```text
<clip>_motion.npz
  fps           = [50]
  joint_names   = canonical 29 joint names
  joint_pos     = (T,36): root_xyz + root_quat_wxyz + q[29]
<clip>_terrain.npy
  shape         = (K,V,10)
  each box      = center_xyz + quaternion_wxyz + full_size_xyz
```

平地生成 `(0,1,10)` 的空 terrain 数组。保持 yaw-only 箱体约束；动作和地形使用同一世界坐标，单位为米、弧度、秒。已保存的 joint_vel/body pose 若保留，必须按目标模型与时间采样重新计算；不能留下混合 G1 的缓存字段。

加载时按名称显式重排，拒绝重复/缺失关节、未知模型、非法形状、NaN/Inf、非单位四元数及错误时间信息。四元数先连续化符号后 SLERP；速度由重采样后的轨迹计算，根部角速度按项目的世界坐标约定输出。

**6.2 官方平地动作启动集**

使用冻结 [Lab visualization walk 数据与回放定义][lab-env]。其每帧 52 维对应 root xyz、Euler XYZ、20 关节位置、root 线/角速度、20 关节速度。

- 20 关节为双腿 12 + 双臂 8；腰部 3 个关节补零，双臂新增的 6 个末端关节赋默认姿态。
- 源腿部顺序为 hip_roll、hip_pitch、hip_yaw、knee_pitch、ankle_pitch、ankle_roll，按名称转换至 canonical 顺序。
- 按 `FrameDuration` 解释时间并记录其精度，再重采样至 50 Hz；Euler 使用源代码的 `XYZ` 约定转 wxyz。
- 不复制显示代码的 root z 加 0.3 m；用天工足底几何校正地面高度，检查双脚支撑与脚滑。
- 不使用 AMP expert 文件替代 visualization 文件：其同为 52 维，但没有完整根部世界轨迹。
- 首批包含默认站立片段及 walk；必要的起止过渡须经过接触和动态检查，不能以插值拼接默认等同可执行。

该启动集用于链路验证和小数据过拟合，不用于声明独立动作泛化。

**6.3 PHP G1 → 天工重定向**

采用以下数据流，按动作类别逐批启用：

```text
原始 G1 动作与地形
→ G1 FK / 接触阶段提取
→ 天工 root + 29 joints 的整段轨迹优化
→ 接触与碰撞修正 / 动态可行性检查
→ 天工 motion / terrain / labels
→ 固定 train / validation / test 清单
```

轨迹优化同时考虑身体任务空间位置/姿态、足底和掌面接触、关节限位、时序平滑、支撑期滑动、地形穿透和非相邻 link 自碰撞。保持人体部位语义，不能按腕部关节名字直接复制角度；G1 与天工的臂部关节链不同。

初始保留原箱体位置与尺寸。不可行片段输出失败原因并隔离；需要改变地形时另建数据版本，联合优化动作、接触点和起终点，不能覆盖原 benchmark 后仍视作同一任务。

允许对高速度片段做时间伸缩，重新生成接触时序、速度和标签；记录缩放因子。现有 refine_motion 只承担重定向后的微调，不能代替跨机器人 IK。

每个地形变体独立校验。取消“所有变体失败则全部视为有效”的天工回退；无有效变体的片段明确拒绝。仅用 body origin 检查不足，必须检查实际接触/身体几何。

质量门槛：无关节硬限位超出 1e-4 rad；四元数范数误差 ≤1e-4；地形/非相邻自碰撞穿透最大 ≤5 mm，持续超过 2 mm 达 3 帧的片段拒绝；已标记足/手支撑段的表面距离 P95 ≤10 mm，足底切向速度 P95 ≤0.05 m/s。形状近似导致的边界情况保存可视化和诊断，不静默接受。

几何检查通过后做短时 PD/动力学跟踪，报告力矩饱和、速度超限、漂移与跌倒；几何合格不等于动态可行。

数据划分按源动作家族分组，包含同一源动作的变体、时间伸缩版本和派生片段都在同一组；固定种子 0，按类别分层为 80%/10%/10%。类内不足三个独立家族时明确标记“泛化评估数据不足”，不能随机切帧制造独立测试集。

交付物：转换器、重定向器、质量报告、天工动作与接触标签、固定划分清单。失败处理：把失败动作按 IK/碰撞/速度/动力学分类，先缩小训练集到合格子集，不以奖励调整掩盖错误标签。

**7. M5：Stage 1 平地闭环与跟踪能力扩展**

前置条件：M4 启动集与标签通过；扩展训练使用对应已合格的重定向批次。

执行顺序：

1. 1、16、64 环境分别运行至少 500 控制步的 reset/step 检查，检查 obs、state、reward、command、力矩与梯度为有限值。允许尚未学会的策略跌倒，但不允许数值异常、越界访问或接触缓冲溢出。
2. 在 64 环境的小数据集上训练 tracker/world model/critic；先确认能过拟合站立与步行，再按显存和吞吐提升至 512、4096 环境。
3. 同时记录世界模型 1、8、32 步预测误差；短时多步误差与“状态保持不变”的基线比较，避免只看单步训练 loss。
4. 每次 checkpoint 同时做无动作噪声评估和训练动作噪声评估。通过门槛以无动作噪声为准，噪声评估用于诊断。
5. 平地通过后逐类引入台阶、手支撑和攀爬，重新检查接触覆盖、奖励量级及终止阈值。

初期保留现有网络宽度、历史长度、训练超参数和扫描点数作为可复现起点；控制参数、关节/接触语义及站姿高度必须先完成适配。物理不稳定时先排查模型和 PD，再做 200/400 Hz 物理频率对照，策略仍为 50 Hz。

设置天工 stand_height 为默认站姿的足底对齐结果；goal fall_height 根据通过质量检查的参考数据最低 pelvis 离地高度减去 0.10 m、下限 0.15 m 生成并冻结。最大允许倾斜取有效参考范围加余量后人工核查；不直接照搬 G1 参数或把明显跌倒姿态纳入正常范围。生成值写入运行配置，扩展动作集后重新审计。

验收目标：默认站立 15 s，在 20 次固定评估复位中的无跌倒比例 ≥95%；启动集每个完整片段 20 次复位成功率 ≥90%，无动作噪声；足底支撑期切向速度 P95 ≤0.10 m/s；世界模型 8 步的 root/joint position RMSE 均优于状态保持基线，32 步误差必须报告。扩展集测试按类别报告，不用总体均值掩盖整类失败。

交付物：首个 29 DoF 可运行闭环、tracker/world model checkpoint、完整评估和视频。失败处理：优先区分数据质量、动作不可行、执行器、模型误差及噪声依赖；未通过平地门槛不推进 planner。

**8. M6：Stage 2 接触规划与联合训练**

前置条件：对应任务类别的 Stage 1 tracker 通过验收；使用同一模型、动作和接触定义。

以天工 tracker 的成功轨迹生成 hindsight 标签，训练新的离线 planner，再进入 planner/tracker 联合优化；不复用 G1 planner labels、权重或归一化统计。联合训练从通过评估的天工 tracker 与离线 planner 初始化。

初始保留 planner 17×11、tracker 9×7 的扫描点数及 1.5 s 接触窗口。统计未来有效接触点的窗口内覆盖率，目标 ≥95%；不足时按 0.1 m 网格步长对称扩大对应扫描边界并增加点数，保持分辨率。扫描配置变更后重新训练相应输入层和标签，不与旧配置混用。

验收在固定的独立动作划分上进行：脚部接触 F1 ≥0.85；手、膝有足够正例时 F1 ≥0.75；有效落点误差中位数 ≤0.10 m、P95 ≤0.20 m。每类至少统计 50 次真实接触 onset，缺少正例标记 N/A；攀爬类缺少手接触数据时不得宣称攀爬规划通过。肘/骨盆无正例时可为 N/A，但对应几何通道仍必须正确。

无动作噪声的测试片段完成率要求每类 ≥80%，且较对应 Stage 1 下降不超过 10 个百分点。固定并同时报告参考时间窗口；不能通过不断放宽终止窗口制造成功率提升。

交付物：hindsight 数据版本、离线 planner、联合 checkpoint、接触指标及成功率。失败处理：先检查标签覆盖、扫描可达性和 tracker 能力，再调整联合训练交接进度；不跳过失败类别直接进入最终泛化结论。

**9. M7：Stage 3 程序地形与目标到达**

前置条件：M6 对纳入任务的类别通过；保持九类接触语义和资产版本。

从 Stage 2 天工 checkpoint 初始化，训练中保留参考跟踪与目标到达两组，初始比例 1:1。程序地形沿用现有九类、十级难度作为第一版 benchmark；几何不可行的场景明确记录，不在评估后静默删去。

训练种子固定为 0、1、2。训练地形种子与模型训练 seed 绑定；最终评估使用未参与训练/选 checkpoint 的地形种子 1001、1002，各生成每类每级 20 个实例。修正当前评估器将 episodes 截断到 bank 数量的隐式行为：生成足够实例并报告实际 episode 数；不足时明确失败，不以更小样本假装完成。

保留现有目标到达定义：15 s 内到达水平距离 ≤0.5 m、垂直误差 ≤0.1 m 的目标，且 episode 最终没有跌倒；天工目标 pelvis 高度与已冻结站姿一致。额外报告到达时间和最终稳定性，不修改成功定义来适配结果。

完成门槛：三个训练种子分别在每类低难度 0–2 级的未见地形上，合并两个测试地形种子的成功率 ≥80%、跌倒率 ≤10%；3–9 级完整报告逐级结果，不保证最高难度通过。若有不可行类别，该类别仍单列，整体状态标记为未完成或明确缩小后的版本，不能报告为完整九类通过。

同时回归 Stage 2 的独立参考动作集：无噪声完成率下降不超过 5 个百分点。报告训练 seed 间差异、每类成功率/跌倒率/耗时、接触落点误差、脚滑、关节越限、力矩饱和与世界模型误差。最终 rollout 还检查自碰撞与地形穿透；达到目标但存在严重无效穿透的 episode 不算有效成功。

交付物：三 seed 训练记录、固定测试地形和 split manifest、完整评估 JSON/表格/视频、最终配置和 checkpoint 哈希。失败处理：按地形和难度定位数据覆盖或动力学不足，保留失败结果，回到 M4–M6 补充能力后重新跑固定评估。

**10. 目标命令、提交顺序与验证清单**

**10.1 尚未实现的目标接口**

以下命令只规定迁移后的接口形态，当前仓库不能直接按这些参数运行。原有模块参数风格保持兼容：工具层使用 `--motion-dir`，向训练模块转发为 `--motion_dir`；新增 `--robot` 在各入口统一。

```bash
# 目标接口：机器人检查，增加 robot/data 参数并去除固定 clip 数量
python -m tools.check_acp_setup \
  --robot tienkung3_29dof \
  --motion-dir data/tienkung3_29dof/motions/bootstrap \
  --contacts-dir data/tienkung3_29dof/contacts/bootstrap \
  --num-envs 16 --steps 500

# 目标接口：生成天工标签
python -m supermimic.acp.make_labels \
  --robot tienkung3_29dof \
  --motion_dir data/tienkung3_29dof/motions/bootstrap \
  --out data/tienkung3_29dof/contacts/bootstrap

# 目标接口：Stage 1 启动闭环；无 G1 --init/--resume
python -m supermimic.acp.train_stage1 \
  --robot tienkung3_29dof \
  --motion_dir data/tienkung3_29dof/motions/bootstrap \
  --contacts_dir data/tienkung3_29dof/contacts/bootstrap \
  --num_envs 64 --seed 0 \
  --log_dir logs/tienkung3_29dof/bootstrap_seed0/stage1

# 目标接口：扩展现有流水线至 Stage 3，数据必须已通过 M4
python -m tools.train_acp_pipeline \
  --robot tienkung3_29dof \
  --motion-dir data/tienkung3_29dof/motions/php_retargeted \
  --contacts-dir data/tienkung3_29dof/contacts/php_retargeted \
  --run-dir logs/tienkung3_29dof/full_seed0 \
  --seed 0 --through-stage stage3
```

对 make_labels/refine_motion/evaluate/render 的调用由流水线统一传递 robot 和 manifest。渲染使用记录中的模型信息；不依赖硬编码 G1 XML 字符串或 qpos 索引。

**10.2 分阶段提交**

| 顺序 | 建议提交主题 | 提交内容 |
|---:|---|---|
| 1 | docs: plan tienkung3 29dof migration | 本计划与来源说明 |
| 2 | feat: prepare pinned tienkung3 assets | 下载/转换工具、manifest、头部固定规则 |
| 3 | refactor: parameterize acp robot configuration | RobotSpec、维度、CLI 和 G1 兼容 |
| 4 | feat: support tienkung3 kinematics and contacts | FK、碰撞几何、数值验证 |
| 5 | feat: convert and retarget tienkung3 motions | 动作转换、重定向、标签和质量检查 |
| 6 | feat: train tienkung3 acp stages | 训练、resume、流水线和完成标记 |
| 7 | feat: evaluate tienkung3 simulation migration | 泛化评估、回归与复现说明 |

实际提交前逐项检查暂存 diff，特别是已有用户修改所在文件。当前文档创建不自动执行这些提交，也不自动创建分支。

**10.3 必须覆盖的验证场景**

| 分类 | 正常场景 | 必须拒绝或明确报告的场景 |
|---|---|---|
| 模型 | 29 hinge + freejoint、头部 fixed、29 位置执行器 | 31 活动关节但只输出 29 维、重复 actuator、网格缺失 |
| FK | 活动/固定多级链、TCP、非零 anchor、左右关节脉冲 | 丢失固定 frame、旋转或速度坐标不一致 |
| 接触 | 九组非空、足底/掌面、箱体顶面/侧面/边缘 | 空几何、未支持形状、近似距离显著偏离真值 |
| 动作 | 名称重排、wxyz、50 Hz、空平地 terrain | 重复/缺失名称、混入 G1 数据、fps/地形不匹配 |
| 缓存 | 模型和数据哈希一致时复用 | 几何/执行器/时间步变化后沿用旧标签或完成标记 |
| checkpoint | 天工同配置 resume、天工阶段间初始化 | G1→天工误加载、显式 robot 与记录冲突 |
| G1 回归 | 旧默认调用、初始化、短回放 | 参数化后 G1 名称、默认姿态、控制周期发生变化 |
| 训练 | 1/16/64 环境、compile/backward、正常 reset | 非有限数、接触约束溢出、非法索引 |
| 评估 | 无噪声主结果与噪声诊断分别保存 | 只报告训练噪声成功率、按帧泄漏划分、静默减少 episode 数 |

每个阶段证据包含代码版本、工作区依赖、robot/model/config/data 哈希、种子、命令、实际样本数、指标和失败原因。阶段状态使用未开始/进行中/通过/失败，不把文件存在等同验收通过。

**11. 已知风险与后续边界**

当前项目 README 中 Stage 3 结果仍为 TBD；Stage 2 还记录了无动作噪声性能显著变差的问题。这些是原项目基线风险，本计划不承诺仅完成模型替换即可解决。所有天工里程碑以实际仿真评估为依据。

官方资产之间存在机械参数差异；使用本计划冻结源并保留冲突表。xMimic/GMR 的动作格式和机器人注册不能仅凭同名字段直接复用，后续采用人体源动作时另写对应适配器。

实机部署不属于 M0–M7。后续需要确认实际天工硬件/末端版本与负载能力、SDK 关节映射、IMU 外参、真实高度图、planner/tracker 导出及推理时序。训练中的世界模型和 critic 通常无需部署到机器人。仿真中手/膝/肘接触成功不构成实机结构可承载的证明。

**12. 计划文档验收与状态**

- [x] 明确 29 DoF 模型来源、固定提交和头部处理。
- [x] 给出分支名、工作区保护原则和分阶段提交边界。
- [x] 给出完整关节顺序、frame/contact 映射、控制和数据约定。
- [x] 覆盖 M0–M7 的前置条件、交付、验收和失败处理。
- [x] 将目标命令与现有接口区分，标明尚未实现。
- [x] 将工程目标与已核实事实区分，实机部署单列为后续。
- [ ] M0–M7 实施、训练与仿真验收完成。

后续更新本文时，只有存在相应实验或检查证据才更新阶段状态。更改模型版本、动作集、控制周期或验收门槛应记录原因和影响，不覆盖历史结果。

[urdf]: https://github.com/Open-X-Humanoid/TienKung_URDF/blob/4c2c061d72dcdab60d74e45d3efb17246a3b6f62/tiangong3_urdf/urdf/tiangong3.urdf
[lab-robot]: https://github.com/Open-X-Humanoid/TienKung-Lab/blob/ffea8bde32fcb632b5ef2a700e452f99e907ea40/legged_lab/assets/tiangong3/tiangong3.py
[xm-robot]: https://github.com/Open-X-Humanoid/xMimic/blob/0385122a232279210e941c7bb28da6a0bbb42356/source/whole_body_tracking/whole_body_tracking/robots/tiangong3.py
[lab-env]: https://github.com/Open-X-Humanoid/TienKung-Lab/blob/ffea8bde32fcb632b5ef2a700e452f99e907ea40/legged_lab/envs/tienkung3/tienkung3_env.py
