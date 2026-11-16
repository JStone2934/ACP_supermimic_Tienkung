# H8 双卡物理回报监督：从零训练

用户于 2026-11-14 明确要求固定 H=8，不比较 H=4，并在实现验证后启动正式训练。

入口为 `python -m supermimic.acp.train_aiming_fullbody --from-scratch`，配置为同目录的
`h8_physical_dual_scratch.json`。三个网络随机初始化，不加载旧策略、WM、critic 或优化器。
actor 输出层偏置设为物理 reset 姿态对应的动作，避免右肘从 −π/2 突然请求 −0.5 rad；
网络权重仍为随机初始化，动作合成、限位、奖励与正式验收没有改变。

## 训练行为

- GPU 0：512 个常规环境、每轮 12 步采样、10 次 WM 更新；WM batch=512，actor batch=256。
- GPU 1：128 个完整回报环境，其中 96 个训练、32 个验证；候选配对评估每个策略 64 环境。
- H=8、WM 窗口=8、历史=10、replay=1000、gamma=0.99。WM 诊断只检查 1/8 步。
- 从零训练默认 actor 学习率 3e-6；WM/critic 为 1e-4。身体探索为 0.005 rad，正式通过 A 后降至 0.0005 rad；右臂为 0.02 rad 并按课程调节。
- 初始采集 256 步建立真实输入统计；输入统计增加尺度下限后冻结。WM 初始预热 1000 轮。
- 物理采样使用 0—64 步的不同探索前缀；每四个环境有一个无前缀。前缀之后关闭探索，只使用确定性尾段的完整回报标签，每 4 步记录一个有效起点。历史保留真实探索前缀，未来奖励对应固定策略。
- 真正终止或 1500 步任务终点切断回报，保留终止当步奖励；所有环境已结束时提前结束采集，reset 后奖励不进入标签。训练/验证按完整环境分组，验证误差用于选择权重，不宣称它是独立测试误差。
- critic 首次最多 5000 次、后续最多 2000 次更新，每 100 次检查验证 MAE，连续 5 次无明显改善提前结束，恢复验证最佳权重及优化器。actor 从同一批训练环境的真实起点优化，不使用验证环境。
- 每批最多 5 次 actor 更新，WM/critic 在 actor 更新期间固定；随后完整物理配对检查才提交候选。当前实现直接使用完整检查，不以短检查替代它。
- GPU 1 采集期间 GPU 0 按固定轮数更新 WM；一批五次 actor 候选对应五轮常规采样和 WM 更新，不按等待时间额外增加更新。

## 接受、回退与课程

尚未通过站立时，优先保护存活时间，再比较完整有效回合的未折扣物理回报；延长存活可以伴随最大漂移增加。
正式通过 A 后，额外保护脚/根漂移、倾角、跌倒和站立通过率，并对比本阶段最佳记录。
配对差异的标准误仅作为波动启发式，不是统计保证；候选容差不修改正式验收阈值。

拒绝时逐张量恢复 actor 与 Adam，接受计数和探索不增加；尝试次数单独保留。
连续三批拒绝才减半 actor 学习率（最低 1e-7），并先做 100 轮真实采样/WM 更新。
初始策略做一次正式评估，之后每十批接受更新做一次正式评估；两次完整通过才升级。
每阶段重建物理 critic 并重新校准，刷新 replay 任务上下文。C3b 最终还要独立重载检查点复验。

正式标准：30 秒完整轨迹，根水平最大漂移 ≤10 cm，每脚 ≤5 cm，最大倾角 ≤15°；
瞄准阶段要求 3 秒内开始持续命中并保持至少 5 秒。失败样本和起始帧均保留。

## 运行与恢复

```bash
python -m supermimic.acp.train_aiming_fullbody --from-scratch \
  --config docs/config/standing_aiming_fullbody_v2/h8_physical_dual_scratch.json \
  --log-dir /absolute/path/to/new/train

python -m supermimic.acp.train_aiming_fullbody --physical-critic \
  --resume /absolute/path/to/train/last.pt --log-dir /absolute/path/to/train
```

新运行要求空目录；全用户锁避免不同 checkout/冻结 runner 同时启动本训练。
worker 通过有版本和摘要的 CPU 文件交换数据，一次仅一个任务在途，不共享 CUDA tensor。
worker 退出、超时、过期结果或数值异常都会停止策略推进并保存最近接受状态。
中断中的未决候选不会成为正式 checkpoint；恢复重建 replay 和物理回报数据。

正式运行使用冻结代码副本，记录代码哈希。训练服务与 PHP guard 分离，保留主机资源保护。
状态见 `status.json`，历史见 `metrics.jsonl`，完整评估和每批物理证据见 `worker/*/`；
大型 IPC 数据消费后删除，小型 JSON 报告保留。

## 验证边界

`--smoke` 是有限接线测试，`--smoke --capacity-check` 保留正式环境数与 batch 检查容量；
测试预热和 critic 预算缩短，产物标记 `smoke=true`，不能作为正式训练起点。
从零策略的候选被拒绝不代表回退失败，也不代表已经完成站立训练。
旧 `tools.test_aiming` 依赖本机缺失的旧 motion 资产；新 fullbody 训练不依赖该数据。

## 本次启动回执

2026-11-14 20:08（Asia/Shanghai）已从全新初始化启动 `aiming-fullbody-h8-physical.service`。
运行根目录为 `logs/tienkung3_29dof/standing_aiming_fullbody_v2/dual_h8_scratch_20261114/`，
`launch.json` 保存精确启动命令、源代码状态和服务回执，`runner/` 为冻结代码副本。
该运行没有加载 smoke/capacity/resume 测试权重。训练没有轮数或 actor 更新预算上限，
按原课程验收运行；数值或 worker 故障会显式停机，服务不自动重新初始化模型。

验证目录为同级 `dual_h8_validation_20261114/`：26 项 CPU 契约通过，双卡小规模、
512/128 环境容量、checkpoint 恢复和 H8 WM 验证已运行。
小规模与容量测试的候选全部被拒绝，已核对 actor 参数及 Adam 状态精确恢复；
容量监测的总显存峰值 GPU 0 为 2687 MiB、GPU 1 为 1436 MiB。
这些是工程和短程模型检查结果，不代表正式策略已通过站立或瞄准验收。

```bash
systemctl --user status aiming-fullbody-h8-physical.service
tail -f logs/tienkung3_29dof/standing_aiming_fullbody_v2/dual_h8_scratch_20261114/stdout.log
cat logs/tienkung3_29dof/standing_aiming_fullbody_v2/dual_h8_scratch_20261114/train/status.json
# 正常停止并保留最近接受状态：
systemctl --user stop aiming-fullbody-h8-physical.service
```

## TensorBoard

本次训练的界面：`http://localhost:6007`，run 名为 `H8_from_scratch`。
6006 端口原有的 Stage1 TensorBoard 保持运行。远程连接时，在 VS Code 的“端口”面板
转发服务器的 6007 端口，再打开本地地址。

`aiming-fullbody-h8-tb-writer.service` 每 5 秒读取新增的完整 `metrics.jsonl` 行，
补齐历史并写入运行目录下 `tensorboard/H8_from_scratch/`。
它不导入或修改训练器，未重启正式训练；持久化游标支持观察进程重启后继续读取。
`aiming-fullbody-h8-tensorboard.service` 提供页面，部署回执在运行目录的 `tensorboard.json`。

曲线横轴统一使用 learner iteration：

- `loss/world_model`：WM loss；预热期间主要看这一项。
- `loss/actor_batch_mean`、`loss/critic_last_training`、`critic/best_mae`：进入联合训练后自动出现。
- `physical/current_policy/*`：实际保留策略的存活时间、根/脚漂移、跌倒率和站立通过率。
- `physical/candidate/*`、`policy/candidate_accepted`：候选表现及是否接受；被拒绝的候选不冒充当前策略。
- `progress/*`、`curriculum/stage_index`、`wm_validation/*`：训练计数、课程和 H8 模型检查。

已核对运行配置 `iterations=0`、`actor_budget=0`，训练服务 `RuntimeMaxUSec=infinity`：
无总训练轮数、actor 更新次数或运行时长预算上限。1000 轮仅是初始 WM 预热长度；
课程验收完成和故障保护仍按原训练协议处理。

```bash
systemctl --user status aiming-fullbody-h8-tb-writer.service aiming-fullbody-h8-tensorboard.service
# 同样的观察器也可手动用于其他运行：
python -m tools.aiming_tensorboard --metrics /path/to/train/metrics.jsonl \
  --out /path/to/tensorboard/H8_from_scratch
```
