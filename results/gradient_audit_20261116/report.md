# Take006 当前训练计算图审计

审计对象：实际运行目录 `results/lafan_jwp_aiming_20261115/frozen_take006_upright_v2`，固定检查点 `ckpt_001800.pt`。当前根目录中的 train_aiming_fullbody.py、aiming_fullbody.py、aiming_wide.py 与冻结版本一致。仅检查及 CPU 小批量求导，不改变训练。

## 结果

- 没有发现主链整体断图。imagination_step 保留 policy → compose_action → wm.step → reward；update_actor 冻结 WM/critic 参数但保留输入梯度。rewards/dones 的 detach 用于 critic 标签，日志 detach 不替换 actor objective。
- 当前是 level 0 balance、full 参数训练。实际 aim_weight 来自课程且为 0，覆盖 reward_weights.aim=4 的默认值。测试中 aim 原始项梯度非零、加权梯度为零。feet、heading、foot_boundary 权重同样为零，是配置停用。
- 确认动作硬裁剪产生局部零梯度：两个相同标称初始状态共 58 个输出中 16 个超过 soft limits，其 compose_action 对 raw 的导数均为 0（每个状态 8/29）。request_limit 权重为 0。预测状态上的 joint_limit 惩罚不能直接绕过 raw → clamp 的零导数。共享参数或其他时间步仍可能改变这些输出，不能说关节永远无法恢复。
- chest_upright 单步加权输出层梯度范数约 0.297，正常。chest_excess 在未超限时为 0，符合惩罚定义。
- compact soft termination 在 28–32 度附近提供梯度；安全区和明显失败区饱和。29/30/31 度均测得非零梯度。不是布尔化的硬终止，但失败区不能依赖它提供恢复方向。
- 32 步 nominal rollout 输出层梯度：窗口奖励 135.86、末端 critic 1661.32、合计 1700.06。末端价值梯度约 12.2 倍，提示应检查 critic 主导更新的风险；不能推广为整个 replay 的比例或证明实际不稳的根因。
- 瞄准训练使用连续方向/距离误差，不使用布尔成功作为直接奖励。实际验收使用射线与目标平面的交点误差、捕获时间和持续命中比例，两者不是同一目标。硬验收本身正常，但训练不会直接对“连续命中95%”求导。
- stage2 跟踪入口抽查：实际 actor loss 使用未 detach 的 Jr；相邻 r_sum/rews/dones 的 detach 用于日志和 critic，并非主链断图。未对所有历史入口做动态覆盖。

## 优先后续验证

1. 在真实 replay 分布记录逐关节裁剪率、请求越限量及各奖励的梯度；独立实验考虑现有 request_limit 连续惩罚，保留执行限位。
2. 对固定 replay 比较 reward-only 和完整 objective 的梯度方向/尺度，再做小规模 actor_value_gradient 对照，保持相同物理验收。
3. 启用瞄准阶段后测量连续代理误差与真实目标平面命中误差的一致性。

边界：probe.py 在 CPU eager 执行，两个相同标称初始状态；不是实际 replay 分布，不是 torch.compile 后端梯度一致性检查，也不证明世界模型导数符合真实物理。具体数值见 probe.json。没有运行优化器或改动正在训练的算法。
