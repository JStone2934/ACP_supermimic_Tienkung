# 最小实现与训练准备

更新：2026-11-11（Asia/Shanghai）。本次仅准备 C0，不启动正式课程长训练。监督初始化已通过，动力学结果见本页末尾。

## 已接入

- `AimingPolicy` 复用原共享 encoder/decoder，command 为原 93 维加瞄准 17 维，29 维输出中 25 维有效。旧归一化统计冻结，新字段使用显式尺度。
- `AimingTask` 统一实际 FK、动作合成、末端软限位／2 rad/s 限速和瞄准奖励；`AimingSim` 管理目标与上一末端目标，基础环境不重复限速。
- 单独的 `train_aiming` 复用既有网络、replay、仿真和 WM 算法；Stage1 的新任务入口转到此 runner。原七关节固定任务保持原语义。
- 静态、切换、正弦、Hermite 和混合课程，以及 64 次结算窗口、升级回退、身体保持退火和恢复。提前练习切换的可选支线首版不启用。
- replay 起点恢复目标过程与限速状态；奖励使用后继状态／目标，想象在 30 秒边界切断回报。actor 只读当前目标；critic 另读过程参数。
- 独立评估和导出只需要学生 actor，均不加载教师。导出权重仍需项目的状态／command 转换及 PD 环境，不是实机部署包。

配置见 [default.json](default.json)。模式和子档在 `aiming_curriculum.LEVELS` 中集中列出，不另建配置框架。默认参数是训练起点，代码接通不等于动态课程已通过。

## 运行

在仓库根目录激活现有环境：

```bash
conda activate acp_supermimic
```

生成首批实际动力学教师数据并监督初始化：

```bash
python -m tools.prepare_aiming \
  --out data/tienkung3_29dof/standing_aiming/ik_supervision/20261111_c0
```

`--reuse-data` 复用数据；`--smoke` 只用 8 条短序列验证流程。输出的 `best.pt`、`last.pt` 都带监督阶段标识。真实状态上的离线目标重标注由 `--augment-targets` 执行，标注不作为 WM 动力学数据；`--init-supervised` 可从已有监督 actor 继续拟合。

本次实际增加了 16,049 个有效目标重标注样本，验证集保持不变。方向误差仍较大时使用运行目录中的 `refinement_config.json`：方向损失权重 10、监督学习率 3e-5，其他模型结构和损失不变。该调整只属于监督阶段，正式课程奖励仍按设计权重执行。

中央目标独立闭环、录像和曲线：

```bash
python -m tools.evaluate_aiming \
  --ckpt data/tienkung3_29dof/standing_aiming/ik_supervision/20261111_c0/best.pt \
  --central --out results/standing_aiming_curriculum/20261111_c0/central_final
python -m tools.render_aiming \
  results/standing_aiming_curriculum/20261111_c0/central_final.npz \
  results/standing_aiming_curriculum/20261111_c0/central_final.mp4
```

本轮只执行中央闭环。去掉 `--central` 则运行固定七点、一段切换和一段移动；输入半径固定为 5 cm，同条轨迹另报 20／10／5 cm 命中比例。中央 C0 检查使用半径 20 cm、捕获 3 秒、保持 1 秒。

正式课程命令已经准备，**本次不执行**：

```bash
python -m supermimic.acp.train_stage1 \
  --task standing_aiming_curriculum \
  --aim_config docs/config/standing_aiming_curriculum/default.json \
  --finetune data/tienkung3_29dof/standing_aiming/ik_supervision/20261111_c0/best.pt \
  --log_dir logs/tienkung3_29dof/standing_aiming_curriculum/first/train
```

续训将 `--finetune` 换成 `--resume <last.pt>`。恢复重新采集 replay，执行 100 次仅 WM 更新，丢弃未结算回合／半窗口；不宣称逐轨迹精确续跑。默认总预算 10000 次，每 500 次评估；未达标保留失败指标。

导出单 actor：

```bash
python -m tools.export_aiming \
  --ckpt data/tienkung3_29dof/standing_aiming/ik_supervision/20261111_c0/best.pt \
  --out results/standing_aiming_curriculum/20261111_c0/actor.pt
```

## 本次必要验证

- 两项 CPU 接线测试：方向正反、固定覆盖、末端梯度和限速、位置恢复梯度、目标过程、replay 扩展字段及输入迁移。
- 小规模 GPU：4 次迭代完成 4 次 WM、2 次 actor／critic 更新，保存后恢复再完成 4 次迭代；所有模型参数有限。短测试的 WM 预热为 2 次，正式默认仍为 500／恢复 100 次。
- 512 环境、1000 步 replay 分配和 16 控制步采集通过，实际动作与历史一致；PyTorch 峰值 allocated 约 0.90 GB，不包含全部 Warp／驱动占用。
- 初始采集 256×200=51,200 帧，其中 39,657 帧进入正常监督集；几何 IK 无失败，11,543 帧因先前站立位移超限剔除。最后仍处于正常站立范围的序列为 184/256。
- 监督训练中修复了报告用 `acos` 在编译反向中引入非有限梯度的问题；报告张量现在 detach，实际损失仍为连续方向损失。

最终监督及闭环指标以本页后续结果和运行 JSON 为准。代码准备、监督拟合通过、动力学闭环通过分别记录；不以其中一项代替其他项。

- 静态、切换和移动各跑 10 个控制步，核对真实执行与共享函数的动作、限速状态、后继目标和奖励一致。修复了切换时刻浮点舍入可能造成的一帧延迟。
- 指向向量在 FK 后归一化，避免 TF32 累积误差把非严格单位向量的点积误报为约 2°；监督与运行共用此修正。独立重新加载后的监督方向 P95 为 0.9829°，各关节 RMSE 为 0.3484／0.3511／0.3701°，身体动作 RMSE 为 0.01694，监督门槛通过。

## 最终结果与下一步

C0 **监督初始化通过，独立动力学接管未通过**。最终独立加载的完整 30 秒结果如下；没有回合内重置，也没有 IK 或站立参考策略接管。

| 指标 | 实际结果 | 条件 |
|---|---:|---:|
| 跌倒 | 无 | 无 |
| 骨盆最大水平位移 | 2.553 cm | ≤10 cm |
| 左／右脚最大水平位移 | 6.781／5.202 cm | 每只 ≤5 cm |
| 最大根倾角 | 6.600° | ≤15° |
| 固定四关节 P95 最大误差 | 4.196° | 每个 ≤5° |
| 中央目标捕获起点 | 0.58 s | ≤3 s |
| 最长连续命中 | 2.84 s | C0 ≥1 s |
| `[3,30] s` 内 20 cm 命中比例 | 23.61% | 记录实际保持能力 |
| `[3,30] s` 靶面误差 P95 | 60.55 cm | 未达到稳定瞄准 |

身体脚位条件未通过，因此目标尝试也计失败。几何监督精度不等于 PD 和自由浮动动力学中的跟踪精度。后续应从 C1 中央目标适配开始；课程只有实际整回合结算达标才扩大目标范围。本次没有启动正式课程，也不将站立瞄准任务标记为完成。

- [监督最佳权重](../../../data/tienkung3_29dof/standing_aiming/ik_supervision/20261111_c0/best.pt)、[最终权重](../../../data/tienkung3_29dof/standing_aiming/ik_supervision/20261111_c0/last.pt)、[独立监督复验](../../../data/tienkung3_29dof/standing_aiming/ik_supervision/20261111_c0/best.validation.json)。最终停止时二者 actor 权重相同。
- [单 actor 导出](../../../results/standing_aiming_curriculum/20261111_c0/actor.pt)、[准备状态](../../../results/standing_aiming_curriculum/20261111_c0/preparation.json)。
- [30 秒实际录像](../../../results/standing_aiming_curriculum/20261111_c0/central_final.mp4)、[误差与位移曲线](../../../results/standing_aiming_curriculum/20261111_c0/central_final.png)、[完整结果](../../../results/standing_aiming_curriculum/20261111_c0/central_final.json)、[轨迹](../../../results/standing_aiming_curriculum/20261111_c0/central_final.npz)。

本轮保留原始采集、补充标签、阶段权重与实际配置，便于继续使用。中间未通过的权重不覆盖原先已验收的固定七关节站立模型。归一化修正后使用 20＋14 个 epoch 达到监督门槛；最终独立复验方向 P95 为 0.9829°。首轮存在数值错误的训练不作为成绩。

## 2026-11-11 正式持续训练启动

用户随后授权不限训练预算持续训练，并要求定时检查、故障续训及内存／SSH 保护。正式服务已启动：`aiming-training.service`。原“本轮不启动正式课程”描述仅指前面的 C0 准备阶段。

- 运行目录：`logs/tienkung3_29dof/standing_aiming_curriculum/continuous_20261111/`；实际配置 `iterations=-1`。无限模式不因固定迭代数或一次验收通过而自动结束。
- 用户 systemd 服务托管监督进程，`Linger=yes`，脱离 SSH／登录会话生命周期。没有修改 sshd 配置。
- 每 60 秒检查真实训练子进程退出状态并更新 `status.json`；只在子进程确认退出后重启，已有 `last.pt` 时恢复，避免因观察超时重复启动训练。
- 内存软限额 32 GiB、硬限额 40 GiB、交换区限额 2 GiB、CPU 配额 8 核、nice 10，编译线程 4。限额只作用于本训练服务，不改其他任务。
- `last.pt` 原子替换，保留最近三个编号检查点和最佳权重。每 500 次评估保存；首个保存前失败会从 C0 重新开始。
- 查看：`systemctl --user status aiming-training.service`；日志在运行目录的 `attempt_*.log`。明确停止时使用 `systemctl --user stop aiming-training.service`，以同时停止监督器及其训练子进程。

启动不代表训练效果已通过验收。持续观察模型更新、物理评估、主机内存和显存，故障修复与后续变化另行记录。

持续运行另有 `aiming-health.timer` 每分钟执行只读检查，更新运行目录 `health.json`：记录实际日志迭代、日志年龄、服务状态、内存用量和 OOM 事件。它不杀死或重启活进程；退出恢复仍由单一监督器负责。停止所有监控时，可另执行 `systemctl --user stop aiming-health.timer`。


2026-11-12 在线调参：continuous_20261111 在第 2500 次后使用 recovery_body_weight=10；第 3500 次后增加 foot_position_weight=6（原 2）。其余默认参数未改，实际值以运行目录 config.json 和检查点为准。两次调整前状态分别归档为 before_recovery_weight10.pt、before_foot_weight6.pt。

第 4500 次之后增加 foot_boundary_weight=4：水平脚位误差超过 4 cm 后以 1 cm 过渡宽度增加有界惩罚，5 cm 附近强化回拉梯度。既有 foot_position_weight=6、recovery_body_weight=10 保留；调整前检查点为 before_foot_boundary.pt。
