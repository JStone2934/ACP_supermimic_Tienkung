**G1 → 天工 3.0：ACP / SuperMimic 迁移调研**

调研日期：2026-11-07。范围：当前工作区代码、实际 G1 模型与 PHP 动作样本，以及官方 GitHub 源码。本文区分“源码/数据核实的事实”和“建议实施方案”；没有进行天工训练或实机验证。本次仅新增此报告，未修改训练代码、资产、数据或已有用户改动。

**1. 建议与路线选择**

迁移可行，但不是替换 URDF 后继续训练。需要同时迁移机器人描述、执行器、可微运动学、接触几何、参考动作和部署观测接口。

建议保留当前 `supermimic/acp` 的 mjlab / MuJoCo Warp + SuperMimic 主线，将天工作为第二种机器人接入。TienKung-Lab 用于提供资产、PD/默认姿态参考和独立步行基线；不要为了换机器人把 ACP 整体改成 Isaac Lab + AMP/PPO。

| 目标 | 建议模型 | 工作重点 |
|---|---|---|
| 先验证天工模型能在本项目训练、站立、走路 | 23 DoF，头部及双臂末端 6 关节固定 | 最小化动作来源和执行器配置的额外变量 |
| 保留当前双手、膝、肘参与的全身攀爬 | 29 DoF，固定头部 2 关节，保留完整双臂 | 优先支持手掌朝向、末端接触和接触约束重定向 |
| 头部运动也进入任务 | 完整 31 DoF | 增加执行器、动作数据、观测与传感器外参变化处理；不是第一阶段必需 |

29 DoF 是面向全身接触任务的工程建议，不代表已经证明天工假手/腕部可以承受目标攀爬载荷。真实末端配置、负载能力和硬件版本需在部署前确认。23 DoF 可以验证整条软件链路，但其成功不能证明原 G1 全部攀爬动作可实现。

**2. 调研所用版本与资产差异**

| 仓库 | 分支 | 本次读取的提交 |
|---|---|---|
| TienKung-Lab | dev | `ffea8bde32fcb632b5ef2a700e452f99e907ea40` |
| TienKung_URDF | main | `4c2c061d72dcdab60d74e45d3efb17246a3b6f62` |
| xMimic | main | `0385122a232279210e941c7bb28da6a0bbb42356` |
| xGMR | main | `70f08d75f5e03aa06ed6d521451bd1e6888f5c0f` |
| Deploy_Tienkung | 3.0 | `9786cf1a7ed9e3bead1a8de47ed6a5c251cb1869` |

当前项目 HEAD 为 `751aa5c`，工作区另有用户修改，因此本报告依据工作区内容而非只依据该提交。

实际解析两个官方 URDF 后，得到以下区别。质量是文件中的质量求和，不是厂家整机实测重量；G1 一列来自本地编译后的 MuJoCo 模型。

| 项目 | 当前 G1 | Lab 的天工训练 URDF | 独立 URDF 仓库的天工 3.0 |
|---|---:|---:|---:|
| 可动关节数，不含浮动基座 | 29 | 23 | 31 |
| 天工 URDF link 数 | — | 39 | 39 |
| 天工固定关节数 | — | 15 | 7 |
| 模型质量合计 | 33.341 kg | 63.272 kg | 65.818 kg |
| 头部 yaw/pitch | 不在该 29 DoF 控制集 | fixed | revolute |
| 双侧 elbow_yaw / wrist_pitch / wrist_roll | G1 使用不同关节链 | fixed | revolute |
| URDF collision 形状统计 | 项目采用网格/足底球等 | 14 mesh + 14 cylinder | 39 mesh |

来源：[Lab 训练 URDF][lab-urdf]、[独立天工 3.0 URDF][tg-urdf]。

两套天工资产并不只是“锁住几个关节”的关系，还存在惯性、关节限位、局部坐标和碰撞形状变化。例如腰 yaw 的 URDF effort 从 Lab 文件中的 91 变成独立仓库中的 200；手臂末端 effort 从 25 变成 40。不能混用一套模型的质量、另一套模型的关节限制和第三套部署参数，随后把它们视为同一个机器人。

特别注意：Lab 同目录下的 `tiangong3_torq.xml` 实际有 31 个 hinge、1 个 freejoint、31 个 motor，而训练 URDF 是 23 DoF。该 XML 是力矩执行器模型，不能直接当作当前 `BuiltinPositionActuatorCfg` 的机器人模板重复添加驱动。应明确选定资产版本和 DoF，再生成专用 MJCF。[来源][lab-mjcf]

独立 URDF 仓库根 README 没有充分反映天工 3.0 子目录；本报告已检查实际目录和文件，没有仅依赖 README 推断型号。

**3. 当前项目中哪些部分可复用**

项目有两条代码路径：

- `supermimic/*.py`：较早的 Isaac Lab / USD 运动模仿实现，使用 G1 LAFAN1 数据；`dimensions.py` 有固定的 71/33 维常量。
- `supermimic/acp/*.py`：当前 README 主线，使用 mjlab / MuJoCo Warp、PHP parkour 动作与箱体地形，训练 tracker、planner 和世界模型。

本次推荐优先迁移 ACP，避免同时维护两套后端的迁移。当前主线已经有按 `NUM_J` 推导的大多数维度，因此算法主体可以保留；但配置仍通过 `from .g1 import ...` 分散进入各模块。

| 模块 | 复用判断 | 必要适配 |
|---|---|---|
| 世界模型学习、BPTT、policy/critic 优化 | 算法可复用 | 新机器人数据、归一化统计、网络输入输出与参数重新训练 |
| ContactPlanner attention 结构 | 架构可复用 | 本体观测、身体语义、可达范围、标签重新训练 |
| 箱体地形、height scan、课程采样 | 大部分可复用 | 起始姿态、地形尺度和扫描范围复核 |
| 三阶段训练流程 | 可复用 | 全链路传递 robot/model/data 标识 |
| G1 已训练权重、世界模型、replay buffer | 不可作为直接替换后的可靠策略 | 基线从头训练；迁移学习另做实验 |
| PHP G1 关节动作与接触标签 | 不能直接复用 | 重新重定向、修正和标注 |
| G1 电机常量、固定接触偏移 | 不能复用 | 使用目标机器人参数及几何 |

即使选择 29 DoF，张量形状恰好与 G1 相同，也不意味着策略、归一化或动力学可复用。当前 checkpoint 加载调用 `load_state_dict`，需要额外模型指纹校验，防止“尺寸相同但机器人不同”的静默误用。

**4. 文件级改造清单**

| 本地文件 | 已核实的绑定点 | 建议改动 |
|---|---|---|
| `supermimic/acp/g1.py` | 29 关节名、14 tracked bodies、9 contact bodies、PD、armature、动作缩放、MJCF/data 路径 | 提取 RobotSpec；保留 G1，新增天工 23/29 配置 |
| `supermimic/acp/kinematics.py` | 类名 G1Kinematics、`np.zeros(29)`、忽略固定 link、不支持固定父链 | 通用 FK，正确处理固定链和关节锚点，参数化 tracked frames |
| `supermimic/acp/env.py` | 严格断言 G1 joint 顺序、`torso_link`、G1 动作缩放 | 显式关节排列映射、配置 torso、default pose、执行器 |
| `supermimic/acp/motion.py` | NPZ 的 joint_names 必须等于 G1 列表；joint_pos 前 7 维为 root | 按名字重排，验证模型与坐标约定，新增数据转换入口 |
| `supermimic/acp/task.py` | BodySpheres、PD 力矩估计、G1 contact body 映射 | 通用碰撞近似、固定子树变换、目标机器人的 PD/limit |
| `supermimic/acp/make_labels.py` | 从 G1 spec 构建标注模型；退化距离回退采样仅支持 mesh/sphere | 参数化模型，扩展形状采样；重新生成全部标签 |
| `supermimic/acp/refine_motion.py` | 已重定向 G1 动作的轨迹微调、9 个接触体专用阈值 | 与跨机器人重定向分开；重新调整阈值并加入全身碰撞检查 |
| `supermimic/acp/mdp.py` | 大部分维度已按 NUM_J 推导 | RobotSpec 注入，按尺寸/控制速率调整奖励与终止 |
| `supermimic/acp/networks.py` / `buffer.py` | 导入 NUM_J 等模块常量 | 在初始化时注入维度；隔离缓存和归一化统计 |
| `supermimic/acp/goal.py` | `STAND_HEIGHT=0.78`、`torso_link`、fall_height=0.25 | 从目标站姿与参考动作统计确定 |
| `supermimic/acp/planner.py` / `terrain.py` | 扫描尺寸与时间窗口以 G1 任务配置为准 | 根据天工可达性与目标接触覆盖率决定是否扩大 |
| `train_stage1/2/3.py`、离线 planner、evaluate/render | 直接导入 G1；render 按 pelvis XML 字符串截取 | 统一机器人选择与 checkpoint 元信息；渲染使用明确模型 |
| `tools/train_acp_pipeline.py` | 默认 PHP 路径、296 clips、固定阶段流程与完成标记 | 加 robot/data 参数，数量从数据读取，完成标记包含模型指纹 |
| 下载/转换/检查脚本 | G1 专用路径与断言 | 资产 manifest、固定提交、模型转换及数据验收 |

建议 RobotSpec 至少包含：模型路径和哈希、受控关节名及顺序、所有物理关节与锁定关节、root/torso frame、tracked/contact frames、默认姿态、PD、effort/velocity、armature、action scale、控制周期、初始化高度和数据目录。区分“受控关节”和“物理可动关节”；若模型保留 31 个活动关节却只输出 29 维动作，头部必须有明确的控制/约束，不能自由摆动。

维度由当前 ACP 公式计算如下，不包含额外 contact command 和 height scan：

| ACP 维度 | 公式 | 23 DoF | 29 DoF | 31 DoF |
|---|---|---:|---:|---:|
| state | 13 + 2J | 59 | 71 | 75 |
| xi | J + 4 | 27 | 33 | 35 |
| tracker 单帧 proprioception | 3J + 12 | 81 | 99 | 105 |
| planner 单帧 proprioception | 3J + 6 | 75 | 93 | 99 |
| world model 输出 | 12 + 2J | 58 | 70 | 74 |

**5. 最大的代码风险：可微 FK 与接触几何**

`G1Kinematics` 跳过所有没有 joint 的 body，并且在活动 link 的父 link 未被记录时直接报错。这样无法直接把 `left_tcp_link` 等固定末端纳入 tracked bodies。即使不报错，固定链位姿或非零 joint anchor 若处理错误，也会使世界模型的解析 reward 与仿真状态不一致。

通用 FK 应递归处理整棵树；固定 link 继承父变换，活动 link 应按完整的 joint origin/axis/anchor 计算。线速度还需包含角速度与固定偏移的叉乘。需要以 MuJoCo FK 为参照检查随机姿态下的位置、旋转、线速度、角速度，并用有限差分检查 torch 梯度。这对通过 FK 反传训练的 SuperMimic 尤其重要。

当前 `BodySpheres` 还有两个直接阻塞点：

1. 只支持 sphere、capsule、mesh，不支持 cylinder；Lab 训练 URDF 含 cylinder，而 Isaac Lab 的 `replace_cylinders_with_capsules=True` 不会自动作用于本项目的 MuJoCo 导入。
2. 接触几何只能来自目标 frame 自身或它的直接固定子节点。天工锁定的 `elbow_yaw → wrist_pitch → wrist_roll → TCP` 是多级固定链，不能直接套用 G1 手腕/橡胶手结构。

Lab URDF 中 `elbow_pitch_l_link` 没有 collision，`left_tcp_link` 有 collision。若仅按名称把“肘部”映射到 elbow_pitch 而不核实几何，可能得到空的接触球集合。应为每个语义接触体检查几何覆盖并要求非空。

建议保留以下 9 个语义接触通道，但重新定义每个通道的 frame 与几何归属：

| 接触语义 | 天工候选 frame / 几何 |
|---|---|
| 左右脚 | ankle_roll_l/r_link，脚底支撑面 |
| 左右膝 | knee_pitch_l/r_link，膝部/小腿允许接触区域 |
| 左右手 | left/right_tcp_link 或专门 palm site；明确 TCP 到掌面的偏移 |
| 左右肘 | elbow_pitch_l/r_link 与实际前臂碰撞几何；必要时增加专用碰撞体 |
| pelvis | pelvis 允许接触的外壳区域 |

候选 frame 不等于已验证配置。若选择 29 DoF，不可将跨活动 wrist/elbow_yaw 的几何合并为固定偏移。

足底建议使用可校准的支撑几何，手部使用明确的掌面几何；高精度 mesh 保留用于视觉。当前 mesh 仅用 24 个半径 1 cm 的球近似，迁移后应比较近似距离与 `mj_geomDistance`，测量漏检、误检及接触点偏差。仿真、离线标签和可微 reward 必须使用一致的几何定义。

**6. 执行器、频率和动力学**

当前 G1 使用 BeyondMimic 电机 armature 推导 PD，动作缩放为 `0.25 * effort / kp`。天工 Lab 配置采用另一组 PD 和固定动作缩放 0.25。不能保留 G1 的 `_kp/_kd/motor_of` 再仅修改名字。[天工执行器配置][lab-cfg]

以下是 Lab dev 的训练配置参考值，单位为 kp=N·m/rad、kd=N·m·s/rad、effort=N·m，不能直接视为实际硬件安全限值：

| 关节类别 | kp | kd | effort_limit_sim |
|---|---:|---:|---:|
| hip_pitch / hip_roll | 300 | 10 | 200 |
| hip_yaw | 150 | 5 | 142 |
| knee_pitch | 330 | 10 | 330 |
| ankle_pitch | 30 | 2.5 | 100 |
| ankle_roll | 16.8 | 1.4 | 50 |
| waist_yaw | 400 | 5 | 91 |
| waist_roll / waist_pitch | 400 | 10 | 91 |
| shoulder_pitch / elbow_pitch | 150 | 5 | 90 / 60 |
| shoulder_roll / shoulder_yaw | 50 | 2.5 | 60 / 36 |

这套配置与同仓库 URDF 已有差异，例如 ankle_pitch 的 URDF effort 为 55，而 actuator 配置是 100；elbow_pitch 的 URDF effort 为 50，而 actuator 配置是 60。应保留来源，按目标硬件版本明确最终限制，不可自动取更大值。xMimic 的 29 DoF 配置还提供末端关节 PD 的候选起点，但也需要与选定模型核对。[xMimic 配置][xm-robot]

频率也不同：

- 当前 ACP：`CONTROL_DT=0.02`，4 个 0.005 s 子步，即策略 50 Hz / 物理 200 Hz。
- Lab dev 天工 walk：`dt=0.0025, decimation=4`，即策略 100 Hz / 物理 400 Hz。
- Lab `sim2sim.py`：`dt=0.005, decimation=4`，即 50 Hz，而且固定 23 动作、84 单帧观测。

来源：[walk_cfg.py][lab-walk]、[sim2sim.py][lab-sim2sim]。因此即使只跑官方策略，也需要核对频率和模型一致性。

建议先保留 ACP 的 50 Hz 接口；如刚度与接触稳定性要求，可测试把物理步长减为 0.0025 s、decimation 改为 8，控制周期仍为 0.02 s。此处是待验证方案。若改成 100 Hz 策略，历史长度、1.5 s contact horizon、reward 的 dt、速度差分、折扣/想象时域都要按实际时间重新审计。

较重模型不代表所有奖励权重或 PD 应按质量同比放大。当前 reward 对 torque 平方、关节误差平方做求和，机器人变化后需要检查各项量级，必要时按关节限值归一化。MuJoCo 中现有配置没有硬性关节速度限制，不能把 URDF velocity 误认为已被严格执行。

**7. 动作数据是迁移的主要工作量**

本地实际有 296 个 PHP 动作。抽查 NPZ 的 `joint_pos` 为 `(T,36)`，由 root xyz、root quaternion wxyz、29 个 G1 角度组成；fps=50。当前 loader 用 root 与关节重新计算 FK，没有直接使用文件内旧 body pose 作为新机器人真值。

对于天工，建议继续沿用当前 loader 的紧凑接口：

```text
<clip>_motion.npz
  fps: 50
  joint_names: (J,) 明确的天工关节名
  joint_pos: (T, 7+J) = root_xyz + root_quat_wxyz + joint_angles
<clip>_terrain.npy
  (K,V,10) = box center xyz + quaternion wxyz + full size xyz
```

对应 joint_pos 宽度分别是 30 / 36 / 38。平地也生成空 terrain 文件，因为 `load_terrain()` 仍然会加载它。

路径 A：用官方 walk 数据启动最小闭环。

- `motion_visualization/walk.txt` 实测为 736 帧，每帧 52 维；`FrameDuration=0.033`。
- 布局由 loader/env 代码核实为：root xyz(3)、Euler XYZ(3)、20 关节位置、root 线速度(3)、root 角速度(3)、20 关节速度。
- 20 关节为双腿 12 + 双臂 8，回放代码为腰部 3 关节补零，得到 23 DoF。
- 每条腿的源数据顺序从 hip_roll、hip_pitch、hip_yaw 开始，而 URDF 遍历顺序从 hip_pitch、hip_roll 开始，必须按名字重排。
- Euler 按源代码的 `XYZ` 约定转 quaternion，再输出 wxyz；统一轴向、地面高度与时间采样。不要盲目复制官方显示代码中 `root_pos[2] += 0.3` 的显示抬高。
- 目标是 29 DoF 时，可对额外 6 个末端关节先赋默认姿态，但这只适合平地启动，不是完整攀爬动作。
- 重新计算 qd 和根部速度；检查首尾导数、足底高度和脚滑。

`motion_amp_expert/walk.txt` 也是 52 维，但语义不同：20 joint pos + 20 joint vel + 12 end-effector pos，缺少完整 root 世界轨迹。不能仅依据维度相同就与 visualization 数据互换。[数据与解释代码][lab-data]、[环境回放实现][lab-env]、[AMP loader][lab-loader]

路径 B：使用人体源动作重定向到目标天工模型。这是全身动作的优先路线。可借鉴官方 xMimic 的 23/29 DoF 模型与转换器，但它输出的 NPZ 中 `joint_pos` 只存关节；根部姿态在 body 数据中。与本项目 PHP 的同名字段不是同一种格式，需要专门适配器。[xMimic 转换器][xm-convert]

进一步核查发现，xMimic README 给出了 xGMR 的 `--robot tiangong3` 示例，但本次读取的 xGMR main 的 `params.py` 没有注册 `tiangong3`；上游 GMR 的 `tienkung` 资产路径则指向 `tienkung2_lite.urdf`。因此不能将 `--robot tienkung` 视为已完成天工 3.0 重定向，应固定实际可用分支/提交，或补齐目标模型、关节定义和 IK 配置。[xMimic README][xm-readme]、[xGMR 注册表][xgmr-params]、[上游 GMR 资产][gmr-asset]

路径 C：只有现有 G1 PHP 动作时，做 G1 → 天工的任务空间重定向：

1. 用 G1 FK 提取脚、膝、手、肘、pelvis 轨迹和接触阶段。
2. 针对天工解 root + joint 的轨迹 IK，约束关节限位、足底/掌面位置和朝向、支撑期不滑动、地形穿透和自碰撞。
3. 保留或调整动作时序，检查关节速度和执行器负载；不满足时放慢或重新设计动作。
4. 地形若变化，动作目标、接触点、起终点也必须一起重新优化，不能只缩放机器人或直接平移 root 高度。
5. 输出天工 NPZ，针对每个地形变体重新验证并生成 labels。

G1 与天工肘部默认角度符号不同、腕部关节链不同；机械臂长度、躯干质量和足底尺寸也不同。角度重命名、29 维照抄、删除 6 维或全局比例缩放都不能替代重定向。

现有 `refine_motion.py` 明确是对已重定向动作做小幅清理，不能直接承担跨机器人重定向。即便 IK 没有穿透，也只证明几何可行，仍需 PD 跟踪和动态仿真检查受力可行性。

**8. 接触标签、奖励与地形应一起迁移**

所有 G1 contact labels、refined motions、离线 planner labels、replay buffer、normalizer 和地形有效性缓存都应按新机器人重新生成。建议模型、motion、contacts 各自有 metadata/hash，并在加载时比对。

注意 `MotionLib._variants_ok()` 目前主要用 body origin 检查地形是否切入；当所有变体不通过时还会回退为全部有效。新机器人验证阶段建议改成明确失败并输出诊断，不能把这个回退当成数据合格。

站立目标高度 0.78 m 和 goal fall_height 0.25 m 是 G1 参数。天工 Lab 默认 root z=0.97 m，只能作为初始参考；应从所选模型的 q0 足底位置、参考动作的 pelvis 高度分布确定实际高度。绝对高度阈值和相对参考的高度误差阈值需要分别调整。

扫描初期可保留 planner 17×11、tracker 9×7 的点数，测量新机器人下“未来接触点落在扫描窗口内”的比例，再决定增大覆盖范围或分辨率。若扫描范围变大但点数不变，空间分辨率会下降，窄踏点的选点误差需重新评估。

官方 walk 的 undesired-contact/termination 规则把膝和肘接触视为失败；本项目 ACP 则主动使用这些接触。可以借鉴官方动力学和站立控制参数，但不能整套复制走路任务 reward/termination 到攀爬任务。

**9. 建议实施顺序与验收**

| 阶段 | 产物 | 建议验收标准 |
|---|---|---|
| 0：冻结模型 | asset manifest、23/29 关节表、锁定关节策略、参数来源表 | URDF/MJCF/训练/数据/部署对同一版本和顺序达成一致 |
| 1：通用配置与 MJCF | RobotSpec、天工 MJCF、通用 FK/contact geometry | 单模型可加载；总质量/惯性/限位核对；无重复 actuator；关节单位脉冲与左右方向正确 |
| 2：数值一致性 | FK、速度、梯度及碰撞距离对照结果 | 随机姿态 FK 与 MuJoCo 一致；固定 TCP 链正确；9 类几何非空且距离误差低于选定标注容差 |
| 3：平地最小闭环 | 1–5 个天工站立/走路片段及 labels | 先可视化与 PD 跟踪；再 1/16/64 env reset/step 检查；奖励、状态、梯度有限 |
| 4：Stage 1 | 重新训练的 tracker + world model | 在真实仿真评价完整片段；统计跌倒、跟踪误差、脚滑、力矩/速度超限与模型多步误差 |
| 5：Stage 2 | 天工 tracker 生成的 hindsight 标签与新 planner | 在独立片段/地形上检查 contact F1、落点误差与任务成功率 |
| 6：Stage 3 | 程序地形目标到达策略 | 多 seed、未见地形、无参考情况下验证；保留 reference following 防退化 |
| 7：部署适配 | 导出模型、FSM、观测与关节映射配置、真实高度图接口 | CPU MuJoCo 与训练行为一致；推理延迟和控制时序达标，再逐步做实机试验 |

阶段 2 可先以位置误差 1e-4 m、旋转误差 1e-4 rad 为随机姿态 FK 的工程目标，并按 fp32/模型误差调整；这些是建议目标，未在天工模型上实测。接触球近似不能要求与解析 mesh 距离完全一致，应以任务接触阈值和落点精度需求制定误差预算。

不要在资产与数据还未通过小规模验证时，直接启动 4096 env、296 clips 的完整训练。完成平地最小闭环后，按“低台阶 → 高台阶 → 双手支撑 → 攀爬”逐类扩充数据，便于定位机器人适配与算法问题。

**10. 当前项目自身的基线风险**

当前 README 的 Stage 3 结果仍为 TBD，所以不能把“天工完整三阶段部署”估算为一个已成熟 G1 系统的机械移植。

README 还记录 Stage 2 的明显 action-noise 依赖：40k checkpoint 无动作噪声时失败 165/296，加训练动作噪声时为 8/296。这是仓库报告的结果，本次未重跑。对新机器人，应同时评价无噪声策略和训练噪声策略，跟踪世界模型长时误差，不能将“执行时必须加随机动作”直接当成部署方案。

训练吞吐与显存需要用天工实测。质量更大本身不决定计算量，碰撞 mesh 数量/复杂度、接触对数、求解迭代、物理步长和世界模型规模才直接影响吞吐。本地 G1 的训练时间不能作为天工的可靠工期承诺。

**11. 实机部署的实际缺口**

当前仓库未见完整 ONNX 导出、机器人通信和部署控制器。训练中的世界模型与 critic 通常不需要随策略上机；运行时重点是 planner、tracker、归一化、历史缓存、目标指令、地形观测和 PD 输出。

官方 `Deploy_Tienkung` 的 `3.0` 分支提供 Python FSM 和机器人接口，可作为宿主新增 ACP policy，但它不是现成 ACP 运行时。该分支 checked-in BeyondMimic YAML 是 `num_actions=19`、`motor_nums=29`、`num_obs=133`，且有 locked_joint_map；不能直接以“同为天工”加载 23/29 DoF ACP 权重。[部署分支][deploy-readme]、[示例配置][deploy-cfg]

新增 ACP 运行时应显式处理：

- 状态消息到 canonical joint order 的重排、零位/符号与单位；输出反向映射。
- IMU 安装 frame 到 pelvis frame 的旋转；头/躯干 IMU 与 pelvis 有活动关节时还需运动学补偿。
- 与训练相同的 gravity、angular velocity、heading-relative attitude、history、previous action 和 normalization。
- 正确的 planner/tracker 调用频率、动作 scale/default pose、PD、饱和和推理超时处理。
- 用相机/LiDAR或其他感知得到地形高度图，替换训练里的精确 box height scan；处理外参、延迟、遮挡和缺失值。
- 相对目标的定义与更新。局部 heading 表示降低对全局定位的依赖，但不等于不需要状态估计或局部地形配准。

实机验证应从悬挂/保护条件下的关节映射与小幅动作开始，再站立、平地、低台阶；手掌/膝/肘接触任务还需核实允许接触的结构与负载。此部分属于后续部署工作，不是本次已完成的结果。

**12. 最小交付建议**

第一份实施交付应限定为：保留 G1 可运行；新增天工 RobotSpec 和 MJCF；完成 FK/contact 数值对照；转换少量平地动作与 labels；以小规模环境验证 Stage 1；附上新机器人可复现的评估结果。

若最终目标明确为当前的全身攀爬，应从设计之初支持 29 DoF，固定头部，并采用包含完整末端关节的源动作。可先用 23 DoF/末端静止动作调通基础链路，但避免把 23 DoF 模型和训练结果误认为最终方案。

尚需在正式实施前确定的项目约束是：实际天工 3.0 硬件/末端版本、最终使用 23 还是 29 DoF、是否已有天工全身动作或原始人体动作、目标只到仿真还是需要实机，以及首批必须复现的接触任务。上述信息不影响本次源码层面的迁移结论，但会决定数据和部署工作量。

[lab-urdf]: https://github.com/Open-X-Humanoid/TienKung-Lab/blob/ffea8bde32fcb632b5ef2a700e452f99e907ea40/legged_lab/assets/tiangong3/urdf/tiangong3.urdf
[tg-urdf]: https://github.com/Open-X-Humanoid/TienKung_URDF/blob/4c2c061d72dcdab60d74e45d3efb17246a3b6f62/tiangong3_urdf/urdf/tiangong3.urdf
[lab-mjcf]: https://github.com/Open-X-Humanoid/TienKung-Lab/blob/ffea8bde32fcb632b5ef2a700e452f99e907ea40/legged_lab/assets/tiangong3/tiangong3_torq.xml
[lab-cfg]: https://github.com/Open-X-Humanoid/TienKung-Lab/blob/ffea8bde32fcb632b5ef2a700e452f99e907ea40/legged_lab/assets/tiangong3/tiangong3.py
[lab-walk]: https://github.com/Open-X-Humanoid/TienKung-Lab/blob/ffea8bde32fcb632b5ef2a700e452f99e907ea40/legged_lab/envs/tienkung3/walk_cfg.py
[lab-sim2sim]: https://github.com/Open-X-Humanoid/TienKung-Lab/blob/ffea8bde32fcb632b5ef2a700e452f99e907ea40/legged_lab/scripts/sim2sim.py
[lab-data]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/ffea8bde32fcb632b5ef2a700e452f99e907ea40/legged_lab/envs/tienkung3/datasets
[lab-env]: https://github.com/Open-X-Humanoid/TienKung-Lab/blob/ffea8bde32fcb632b5ef2a700e452f99e907ea40/legged_lab/envs/tienkung3/tienkung3_env.py
[lab-loader]: https://github.com/Open-X-Humanoid/TienKung-Lab/blob/ffea8bde32fcb632b5ef2a700e452f99e907ea40/rsl_rl/rsl_rl/utils/motion_loader.py
[xm-readme]: https://github.com/Open-X-Humanoid/xMimic/blob/0385122a232279210e941c7bb28da6a0bbb42356/README.md
[xm-robot]: https://github.com/Open-X-Humanoid/xMimic/blob/0385122a232279210e941c7bb28da6a0bbb42356/source/whole_body_tracking/whole_body_tracking/robots/tiangong3.py
[xm-convert]: https://github.com/Open-X-Humanoid/xMimic/blob/0385122a232279210e941c7bb28da6a0bbb42356/scripts/gmr_to_npz_inter.py
[xgmr-params]: https://github.com/Open-X-Humanoid/xGMR/blob/70f08d75f5e03aa06ed6d521451bd1e6888f5c0f/general_motion_retargeting/params.py
[gmr-asset]: https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/assets/tienkung/urdf/tienkung2_lite.urdf
[deploy-readme]: https://github.com/Open-X-Humanoid/Deploy_Tienkung/blob/9786cf1a7ed9e3bead1a8de47ed6a5c251cb1869/README.md
[deploy-cfg]: https://github.com/Open-X-Humanoid/Deploy_Tienkung/blob/9786cf1a7ed9e3bead1a8de47ed6a5c251cb1869/policy/beyond_mimic/config/BeyondMimic.yaml
