# Stage2 与当前 Stage1 的对齐检查

2026-11-10：原代码已经继承 Stage1 的机器人、接触几何和数据路径，但执行规模、完整检查点保存及规划器训练/验证划分存在缺口。已完成下述修正并通过 CPU 检查。**没有启动 Stage2，也没有停止或重启当前 Stage1。**

## 当前对齐状态

| 项目 | Stage1 当前配置 | Stage2 检查结论 |
|---|---|---|
| 机器人和模型 | tienkung3_29dof，29 关节 | 从 checkpoint 校验模型/config/joint order，实际 WM/policy 权重严格加载通过 |
| 接触奖励几何 | spheres224，224 球，0 个奖励凸包 | 离线 planner 与联合训练都从 ctc_cfg 继承；真实球坐标哈希完全相同 |
| 输入数据及标签 | 296 条原动作、2960 个原始地形组合 | 从 checkpoint 继承 motion_dir / contacts_dir，天工数据哈希校验；没有删减片段 |
| 环境 / WM / policy batch | 全局 512 / 256 / 256 | 新默认从输入 checkpoint 的全局 args 继承，再由 Runtime 按 GPU 数分配；显式参数优先 |
| 奖励执行 | reward_chunk_size=0，不重算 | 新默认继承该值；显式传 0 仍可覆盖旧断点的分块值 |
| horizon / WM 更新 / UTD / 学习率 | 32 / 10 / 0.4 / 2e-4 | 当前 Stage2 默认一致，预备配置也显式固定；未采用上轮淘汰的超参 |
| BF16 / critic 更新 | MLP BF16 / 5×4 | 当前默认一致 |
| 编号检查点 | 完整保存优化器等状态 | 新增 --save_full_checkpoints，预备配置启用；last.pt 本来即完整保存 |
| 接触几何迁移来源 | geometry_transition | Stage2 保存时继续携带该记录 |
| 双卡同步 | 共享 Runtime，全局计数和同步更新 | 同一路径；本次未额外启动双卡训练 |

## 已修正的问题

1. **执行规模未继承。** 原 Stage2 默认全局 4096 环境、2048 WM batch、4096 policy batch；只传 --init 不会沿用当前 512/256/256。现在在 Runtime 切分之前读取并继承 checkpoint 中的全局值，resume 使用 Stage2 自己保存的值。离线规划器同样继承环境数。
2. **编号检查点不可完整恢复。** 原 Stage2 固定 full=False，编号文件缺少 critic/优化器/课程等恢复字段。现在支持与 Stage1 相同的完整保存开关，预备配置明确开启。未开启时仍保留兼容的轻量编号文件；应使用 last.pt 或开启完整保存后的编号文件恢复。
3. **地形展开后的验证泄漏。** 原来每 10 个 clip 取一个。当前每个原动作恰好有 10 个地形变体，这会让同一原动作同时出现在 planner 训练与验证中。现在依据数据 manifest 的 source_key 分组，每类每 10 个原动作取一个，所有变体一起归组。实际为 planner 训练 266 个原动作/2660 个变体，验证 30 个原动作/300 个变体。两边原动作交集为零，跟踪器仍训练全部 2960 条；这只是 planner 的划分，不宣称整个系统的独立泛化评估。
4. **规划器来源校验不足。** 原 --planner_init 只检查机器人身份，现在同时核对天工数据哈希、生成该规划器的 tracker checkpoint 哈希、接触配置及验证划分。不同 tracker/数据/几何的规划器不能意外混入本次衔接。

验证划分写入 validation_split.json 和完整 checkpoint。新的展开数据任务恢复时必须保持划分一致；没有划分记录的旧展开数据 planner/Stage2 断点会明确拒绝静默改变分组，需要新开对应阶段。没有展开 manifest 的普通/G1 数据仍保留原按 clip 的划分。

## 应保留的阶段差异

- **缓存长度：Stage2 保留 1350，Stage1 是 1000。** 全量数据最长 1326 帧；planner 的 hindsight 标签需要保留完整 episode。若机械改为 1000，会有 250 条变体长度不满足要求，并触发原有完整轨迹断言。
- **学习目标和课程：** Stage2 增加规划器训练、参考/规划目标混合、eta_obs/eta_rew、成功轨迹筛选与 admission hold；起点使用均匀动作阶段。Stage1 的 backward curriculum 不能直接覆盖这些阶段设计。原成功过滤阈值和终止条件未放宽。
- **critic 输入：** 为规划器目标和两个混合系数新增 29 维。初始化将新权重置零，保留 Stage1 的输出行为；宽度变化后的 critic 优化器保持原 Stage2 的重新初始化设计。
- **奖励编译：** Stage1 编译奖励，Stage2 为规避代码中记录的 planned-term Triton backward 编译问题保留 eager 路径。整批计算/不重算已对齐，但这不意味着 Stage2 的速度与 Stage1 相同，不能套用 Stage1 的 ETA。Stage2 还多了每轮 horizon+1 次规划器预测和规划器更新。

## 验证依据

13 项 CPU 单元检查通过：真实入口参数解析、继承与显式覆盖、旧默认兼容、源动作分组/恢复、manifest 误配拒绝、规划器来源匹配、critic 扩展的输出与梯度、224 球身份和 G1 几何回归。见 evidence/stage2_alignment_tests.json。

另读取真实 Stage1 第 5000 轮完整断点做 CPU FP32 审查（它仅用于代码检查，不是已选定的 Stage2 初始化模型）：

- 224 球坐标/半径 SHA-256 与用户接受的候选相同：d97738195ac0160804dacee9306f68ba29e14ca05cbc8eed465feab4bc021a2f。
- WM/policy 实际权重严格加载通过；真实 critic 扩展后最大输出误差 1.53e-5，满足浮点容差。
- 五类动作、129 个带扰动姿态，直接提取并执行两个训练入口的奖励函数；eta_rew=0 时 Stage2 与 Stage1 的输出及状态/动作梯度一致。
- eta_rew=0、0.5、1 下整批与 17 行分块/重算的输出误差为 0，最大梯度误差约 9.09e-7，覆盖规划奖励和不能整除的块尾。

实物理 GPU rollout、双卡 Stage2 实训、规划器实际更新数和吞吐仍需正式启动前验证，本次不宣称已经通过。证据：evidence/stage2_alignment_cpu.json；可复现工具 tools/check_stage2_alignment.py。

## 后续衔接配置

配置草案在 evidence/stage2_prepared_config.json，状态为 prepared_not_launched。保留当前512/256/256和整批奖励，Stage2缓存1350，保存完整检查点；planner batch 512、capacity 300000 是待双卡 smoke 测试的初始值，尚未测速。

应先完成 Stage1 的策略评估，选择固定、不可变的完整 checkpoint，再完成离线规划器和联合训练的短训/恢复检查。Stage2 不能直接拿本次第5000轮审查用断点作为已验收模型。现有通用 tools/train_acp_pipeline.py 还包含重新训练/细化数据及其旧流程设置，不是当前后台任务的直接续跑入口。

当前 Stage1 的训练入口及其全部原有依赖未改。本次只修改 Stage2、离线 planner 入口并新增 stage2_contracts.py。为防止未来 Stage1 恢复被“未使用的 Stage2 文件哈希改变”挡住，已归档旧文件并仅更新这几项源码绑定，记录在 evidence/stage2_alignment_binding.json；当前源码快照为运行目录 source_snapshot_stage2_alignment.zip。
