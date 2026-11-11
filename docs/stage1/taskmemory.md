# Stage1 任务记忆

## 2026-11-10 启动授权

用户要求准备好的数据进入 Stage1，并在 `docs/stage1` 新建计划、taskmemory、资产列表等文档。此前用户已授权服务器后台持续执行，Mac 合盖和 VS Code 断开不影响作业。

已读根目录 AGENTS.md 和 `/home/js/.local/share/php-guard/OPERATIONS.txt`。本会话 `PHP_OVERNIGHT_OWNER=1`，`php-codex.service` active，当前就是同一服务器后台执行者。guard 中所有转换作业均已结束；没有活动训练进程。原转换目标已经 complete，不重新创建或恢复它。

已确认双 RTX 5090 空闲，主机可用内存约 108 GiB，磁盘剩余约 555 GiB。采用已有双卡实测的 512 环境、256/256 全局 batch、reward chunk 64 起步。没有修改 guard、服务现有配置、内存保护或质量阈值。

最终转换索引不可直接用于 MotionLib。决定每个原始地形编号生成一个完整单变体片段，完整保留 2960 个编号，避免直接合并 accepted 目录时纳入不匹配地形或重复难例。打包工具会验证交付收据、输入哈希和输出逐数组相等。

当前步骤：打包数据 → 全量装载 → 双卡短训和恢复 → 正式 Stage1。具体当前状态见 progress.md，证据放 evidence/，大型数据与模型放 data/、logs/。

## 18:01 后台流程已启动

打包完成 2960 个完整单变体样本、1812360 个展开帧，约 781 MiB。逐输出比较切片标签和地形；MotionLib 原帧数据以文件复制保持原哈希。3 项映射单元测试通过。

`acp-stage1-php-all-20261110.service` 于北京时间 18:01:20 启动，初始主 PID 3856648。流程为全量 loader 检查 → 双卡 3 轮短训 → 恢复至 4 轮 → 从头正式训练 60000 轮。失败即停止，不绕过检查推进。正式配置为双卡 512 环境、256/256 batch、每 500 轮完整 checkpoint；原有 500 轮 WM-only、奖励和课程不变。

控制器持有运行目录锁，并检查启动时登记的代码和数据清单哈希。阶段日志与命令持久化，progress.md 每 60 秒刷新。服务独立于终端，属于原计算 slice，64/80 GiB 的训练单元限制小于共享 96/104 GiB。当前服务为 transient unit，终端断线不影响；服务器重启后需依 usage.md 恢复，不承诺主机重启自动恢复。

## 全量加载时的重复模型哈希修正

第一次 loader 检查约 4 分钟仍在 MotionLib 加载，进程累计读取约 300 GB 缓存内容。定位到 `RobotSpec.metadata()` 在每个动作及每份标签验证时重复读取整套机器人网格资源。主动停止的是本次训练服务的 preflight，尚未发生训练更新；没有停止转换 guard 或后台 Codex 服务。

`supermimic/acp/robots.py` 增加仅用于模型资产的进程内哈希复用，依据 device/inode/size/mtime_ns/ctime_ns 失效，校验前后均检查文件身份。最近一秒发生变更的文件不复用哈希并执行两次读取，覆盖文件系统极短写入间隔内时间戳不变化的情况。动作和标签文件的哈希仍逐文件验证；模型、配置身份及验收阈值不变。

初版测试发现快速原地改写可能具有相同时间戳，已通过最近写入窗口修复，保留 `startup_regression_tests_initial_failed.txt`。修正后 18 项测试通过，含资产改写/替换检测、机器人契约及数据映射。原配置和源码快照保留为 `launch_config_initial.json` / `source_snapshot.zip`；当前配置绑定新哈希，源码快照为运行目录 `source_snapshot_v2.zip`，重新开始同一 preflight。

## 18:09 全量装载通过

`dataset_preflight.json` 确认 2960/2960 个原始地形样本可采样，展开 1812360 帧全部保持，所有帧接触指令有限，command_dim=93。全量校验用时 61.08 秒，PyTorch 数据加载峰值 4259645952 字节（约 3.97 GiB）。随后自动进入双卡短训；此显存仅指数据检查阶段，不代表训练峰值。

## 18:15 三轮短训通过

`smoke.done.json`：WM/policy/critic 实际更新 30/3/60，skipped 为空，两卡共享状态 SHA-256 一致，checkpoint 的 data_hash 与 preflight 一致。每卡 PyTorch 分配峰值约 15.79 GiB，保存时设备占用约 18.86 / 17.96 GiB；checkpoint 约 205 MiB。恢复至第 4 轮正在执行。

初次编译以外，第 2、3 轮约 44–45 秒/轮，60000 轮粗估约 31 天，已向用户明确说明这是真实全地形数据的初步测量，不以历史常值姿态小数据速度估算。保留 `initial_throughput.json`，后续需正式稳定窗口校准；迭代预算不是能力验收标准。

## 18:19 正式训练启动

恢复至第 4 轮通过，WM/policy/critic 更新计数 40/4/80、无跳过，checkpoint 和全部短训日志数值有限，双卡状态一致。`resume_smoke.done.json` 保存证据。控制器随后于 18:19 自动启动正式训练，目标 60000 轮，使用全量已验收数据，从头初始化（无 --init / --resume），原 500 轮 WM-only 保留。

正式 torchrun 初始 PID 3868500，服务器 unit `acp-stage1-php-all-20261110.service`，当前控制器 PID 3860640。启动收据 `evidence/training_start_receipt.json` 绑定当前代码快照、配置、数据、短训及恢复证据。后续状态以运行目录 status.json 和本目录 progress.md 为准，不能把本段启动记录当成实时完成状态。

正式训练已写出第 25 轮及之后的指标，数值有限，服务保持 active。此时仍在前 500 轮 world-model 预热，日志 policy/critic 为 0 属预期；约 1 轮/秒的预热速度不能替代完整 actor 训练约 44–45 秒/轮的初步测量。首批健康证据见 `evidence/first_training_health.json`。

## 18:45 耗时重新测量与接触采样诊断

用户要求解释训练慢并重新估计。正式第 501–525 轮实测 1088.1792 秒，即 43.527168 秒/轮，剩余 59475 轮约 30 天；前 500 轮 WM-only 只需 444.7 秒。服务内存 high/max/oom 事件均为 0，转换队列全部结束，两个训练 rank 正常运行。未停止或重启正式训练。

查明天工 BodySpheres 使用约 8 mm 凸包三角面网格，共 353146 表面点、9 组凸包；G1 使用球/胶囊及每网格 24 球的近似，共 224 点。该差异属于实现选择。每卡每轮 4096 个预测状态、64 个奖励块及反向重算，使接触几何成为首要性能分析候选；尚无分阶段 GPU profiler 归因。

用户随后询问能否降为 224 点。执行 CPU-only 同姿态几何对照，不改主训练代码/参数。15 条原动作、120 个姿态、1080 次部位检查，共 223 次原生接触。224 表面点漏判 5 次；224 小球近似误判 4 次；864 点漏判 2 次；2304 点本样本没有判定错误，但近表面距离最大误差仍有 7.83 mm，尚未验证梯度、随机姿态或策略 rollout，不能宣称已可正式替换。详见 contact_sampling.md 及 evidence/contact_sampling_probe_20261110.json。

当前正式训练保持密集几何，继续同一后台作业。原转换目标仍保持 complete，没有新增目标。优化后的 ETA 必须通过实际 GPU 完整更新测量，不能按点数缩减比估计。

## G1 小球分布可视化

用户要求图片或视频说明 G1 小球具体分布。已直接读取当前 BodySpheres 的 224 球，按真实半径生成全身三栏对照、五类部位近景及 192 帧/24 fps 的 8 秒相机旋转视频，保存于 `results/stage1_g1_contact_spheres_20261110/`。额外导出全部局部/世界球心坐标与半径 CSV，复核逐数值一致和产物哈希。

脚底四球半径 5 mm；其他网格奖励球半径 1 cm。每手 48 球分为腕部 24（橙）和橡胶手 24（黄）。膝组包含小腿，肘组覆盖相应肘/前臂碰撞网格。展示将肩部展开以便看清，视频是相机旋转而非训练策略表现；正式训练继续运行。

## 19:20 接触几何方案检索与天工候选图

用户要求检索点和球以外的方案（特别是 box），并画天工 + 224 球。读取 AGENTS.md 和 OPERATIONS，确认本会话仍为 PHP_OVERNIGHT_OWNER=1；后台训练 active，转换 guard 全部 completed。检索 MuJoCo、Isaac Sim / Omni Physics、cuRobo 官方文档及 DCOL / DiffPills 原论文，区别模拟器碰撞表示、可微奖励代理和接触力奖励。cuRobo 的 box 在引用示例中属于环境，查询输入仍是球，未误当成 robot-box 例证。

`tools/render_tienkung_contact_candidate.py` 精确复现前次 `g1_style_224` 配方，脚各 28、其余七组各 24、全部半径 1 cm。生成天工三栏全身、五部位近景、与 G1 同相机同尺度对照三张 PNG，以及 CSV/哈希清单。逐图目视检查，球心复核到原筛查公式；该方案之前有 4 次足部假接触，仍未推广。G1 诊断渲染器仅增加可选相机参数，默认保持。

检索结论及拟议验证路径保存在 contact_geometry_options.md。工程建议优先比较脚部 box、四肢胶囊、复杂部位简化凸包的混合表示，并以 2304 表面点为小改动基线；不是批准替换正式奖励代码或模拟器模型。本次未修改训练核心源码/参数、未重启训练、未新建或恢复原转换 goal。

## 19:28 用户接受 224 球，启动切换验证

用户明确“224 球的方案可以接受，并且可以和原版 G1 保持一致”。已按该选择实施，不再以原 4 次近阈值误判阻止使用。先停止同一 Stage1 unit，保存旧配置/状态/全部日志与第 500 轮断点；新运行目录 stage1_php_all_spheres224_20261110，仍由原服务唯一执行，64/80 GiB 和共享保护保持。

ContactCfg/训练 CLI 新增 spheres224：球心半径精确匹配用户查看的图，脚各 28、其余 7 组各 24、全部 1 cm；奖励直接采用 G1 球到地形距离，不保留天工凸包覆盖，仿真碰撞和转换标准不变。22 项测试通过（含数值梯度、坐标固定哈希、G1 回归、断点几何保护）；重复 120 姿态仍 0 漏判/4 假接触。去掉凸包分支后的差异另存 spheres224_no_hull_probe.json。

新模式下全量装载通过：2960/2960、1812360 帧，实际 224 点、0 个奖励凸包，59.20 秒。后台现按 fresh 3→4 轮、旧 500→502→503 轮迁移/恢复、正式从 500 轮续到 60000 的顺序执行。检查点保存几何模式和源哈希；新目录 resume 日志表头已修正。短训模型不作为正式初始化；未保存旧策略更新不带入。后续进度见 progress.md。

## 19:41 切换完成与正式速度

全部启动检查通过：fresh 3→4 更新 30/3/60→40/4/80；旧 500→502→503 更新 5000/0/0→5020/2/40→5030/3/60。均无跳过、两卡状态哈希一致，几何迁移来源 SHA-256 与第 500 轮原文件一致。19:37 正式从旧 500 轮断点启动，后台服务/保护仍为同一配置；启动收据 spheres224_training_start_receipt.json。

正式 25 次 actor 预热之后，连续 50 轮 65.1036 秒，平均 1.30207 秒/轮，窗口实更新 500/50/1000。相比原密集几何 43.5272 秒/轮约快 33.4 倍。19:41 已到 600 轮，日志所有数值有限、无跳过日志、内存保护事件全 0；剩余 59400 轮初步约 21.48 小时，预计 11 月 11 日 17:10 左右完成迭代预算，评估另计。证据 spheres224_formal_throughput.json。不是策略已达标或收敛承诺；保留全量动作与后续完整仿真验收要求。

当前 canonical launch_config.json 指向 stage1_php_all_spheres224_20261110，旧密集目录完整保留并标记 superseded。新模式已经正式采用，旧图片/报告中“候选、未推广”是历史状态。后续恢复使用当前配置和同一后台服务，默认检查点模式匹配保护有效；评估读取检查点 ctc_cfg 保持 spheres224。不要另起第二个训练执行者。

## 20:00 GPU 利用率与整批奖励执行优化

用户观察 GPU 约 30% 利用率并要求进一步提速。实测 97 秒均值 GPU0/1 35.9%/32.2%，显存带宽约 1.98%/1.11%；发现 224 球仍使用密集时期 reward_chunk_size=64，每卡 4096 状态拆 64 块且反向重算。

19:49 保存完整第 1000 轮后停止正式服务，仅启动 acp-stage1-execution-probe-20261110.service 顺序对照，保护仍 64/80 GiB。两个候选从同源断点各跑 75 轮，前 25 预热后测 50；训练数据、环境、batch、horizon、更新次数一致。chunk64=1.30022 秒/轮，chunk0=0.49781 秒/轮，快 2.612 倍，GPU 峰值约增加 0.893 GiB/卡。两组检查点更新均 10750/575/11500，无跳过、双卡状态一致；额外实际 PaperReward+soft termination 的五类动作 129 姿态对照输出一致、状态/动作梯度最大差 1.49e-8。

分项仅在预热期五轮采样，正式 50 轮窗口没有分项计时：rank0 奖励正向约 275→7 ms，policy_update 内 backward（含 critic）约 700→83 ms；嵌套分项不能相加。诊断副本含计时，正式入口未改。

20:00:46 将控制器硬编码 reward chunk 改为读取配置（默认64），当前 launch_config 设置0，归档旧配置/benchmark/命令/源码，正式同一服务从原第1000轮恢复，不采用测试权重。主训练代码、数据、学习率及更新规模保持。新源码快照 source_snapshot_unchunked.zip；采用收据 execution_tuning_receipt.json。正在校准正式新速度；当前生产服务 active，测试服务已完成 inactive。

20:03 正式新窗口完成：预热25轮后连续50轮21.0721秒，平均0.42144秒/轮，较chunk64基线快3.085倍，窗口更新500/50/1000完整。到1100轮日志有限、内存保护事件全部0，剩余58900轮约6.9小时，预计11月11日凌晨03:00完成迭代预算，评估另计。随后77秒持续采样GPU0/1平均48.77%/41.30%，设备占用约9128/8213MiB；此前97秒约35.90%/32.22%。新证据execution_tuning_formal_throughput.json、execution_tuning_formal_benchmark.json、execution_tuning_gpu_utilization.json。原21.5小时与0.498秒诊断数据保留为历史，当前正式训练仍在同一服务和运行目录推进。


## 20:16 用户授权超参效率对照

用户观察显存和利用率仍有约一倍空间，要求继续调整超参。读取 AGENTS/OPERATIONS，确认 PHP_OVERNIGHT_OWNER=1、转换队列全部终结。正式训练于 20:16:15 在第 3000 轮完整断点保存后停止，同一后台所有者启动单独的有界测试服务 `acp-stage1-hyperparam-probe-20261110.service`，资源保护仍为 64/80 GiB，不与正式训练并跑。

三组从同一个第 3000 轮断点各训练 400 轮：baseline 512 环境/256 WM batch/256 policy batch/WM10/UTD0.4；scale2 1024/512/512/WM10/UTD0.4；balanced2 1024/512/256/WM5/UTD0.8。所有输入仍为完整 2960 条，学习率和 horizon 不变。协议与预先确定的采用筛查门槛见 hyperparameter_tuning.md。固定 60 条诊断样本不代替最终全量验收。

第 3000 轮源断点的诊断评估为归一化回报 0.2223、48/60 提前终止，策略仍处于早期训练阶段；不将训练奖励视为完整动作通过率。停止收据、测试配置与输出分别见 evidence/hyperparameter_stop.json、evidence/hyperparameter_probe_config.json、logs/tienkung3_29dof/stage1_hyperparam_probe_20261110/。


20:38 三组短训及仿真评估完成。baseline/scale2/balanced2 分别为 0.44186/0.41236/0.31842 秒/轮，诊断回报 0.2515/0.2496/0.2246，失败数 48/49/49（共60）。balanced2 的回报和根部误差未通过事先记录的退化门槛，未采用。另起同保护的有界 followup 单元，只测试 1024 环境、256/256 batch、WM10、UTD0.8，仍从原3000轮断点训练400轮；其他训练服务均 inactive。

归档断点检查发现 `last.pt` 与 `ckpt_003000.pt` 原始字节 SHA 不同，确认由 torch.save 内部归档文件名不同所致；完整递归比较所有张量、优化器、计数、课程和元数据完全相同，结果见 evidence/hyperparameter_resume_identity.json。没有断点损坏或替换。


20:46:34 超参筛查结束，正式服务从原第3000轮断点恢复原512/256/256、WM10、UTD0.4配置。追加 env2_keep_updates=0.37231秒/轮、回报0.2439、48/60失败，根部RMSE1.2358m，较baseline0.8702m增42%，未通过预设检查。scale2虽然样本吞吐2.14倍且根部误差较低，迭代耗时只降6.7%、短训回报和通过数无改善，未满足预设10%耗时下降门槛。因此未采用实验权重或新超参。

完整判定和恢复收据为 evidence/hyperparameter_tuning_receipt.json。两个测试服务均完成inactive；同一正式服务active。控制器/训练源码未改，launch_config只记录本次review并将ETA下限更新到3100，原/新配置均归档。基线恢复后继续校准正式速度。预设门槛是本轮实验的筛查规则，不等同于新增用户验收标准。


20:49 恢复后正式校准：预热25轮后连续50轮平均 0.43064 秒/轮；日志到第 3225 轮，剩余约 6.79 小时，预计北京时间 11月11日 03:36 完成60000轮预算，评估另计，不代表收敛保证。实际参数为原512/256/256、WM10、UTD0.4，源码绑定一致，日志有限、内存保护事件为零。证据 evidence/hyperparameter_review_formal_throughput.json 与 hyperparameter_review_formal_benchmark.json。


## Stage2 对齐审查（2026-11-10）

用户要求检查 Stage2 是否与 Stage1 对齐。确认 OWNER=1 和现有 Stage1 服务持续 active，仅进行代码修正和 CPU 检查，未启动 Stage2、未停止/重启 Stage1。发现并修复 Stage2 默认环境/batch未继承、编号检查点不支持完整保存、展开变体按clip验证导致源动作泄漏、planner_init来源校验不足。新 stage2_contracts.py 共享执行参数继承、分组划分和来源验证，并提取 critic 扩展初始化供实际数值测试。

保留1350步Stage2缓存（最长动作1326帧）、规划器/混合奖励/课程和eager planned-reward路径。13项CPU单测通过，真实Stage1第5000轮断点和五类129姿态检查通过；eta_rew=0与Stage1输出/梯度一致，eta=0/0.5/1的分块输出误差0、最大梯度误差9.09e-7。完整记录 stage2_alignment.md；CPU证据 stage2_alignment_cpu.json、stage2_alignment_tests.json。尚未验证双卡Stage2训练、规划器实际更新与速度。

Stage1训练代码及已用依赖未改，仅重新绑定未被Stage1执行的Stage2两入口及新增helper，避免日后恢复因哈希变化受阻；旧代码/配置归档stage2_alignment_before，快照source_snapshot_stage2_alignment.zip，绑定收据stage2_alignment_binding.json。Stage2预备配置stage2_prepared_config.json仅为草案，需Stage1评估选定不可变完整断点后做离线planner和联合训练短训，不自动启动。

## 2026-11-11 03:00 Stage1 完成、评估与 Stage2 启动

Stage1 于 03:00:47 跑满 60000 轮，service Result=success，ckpt_060000.pt / last.pt 完整保存。全量 2960 条评估（`stage1_final_eval.json`，3 分钟）norm_return 0.5271、失败 837/2960（28.3%）：locomotion 0.6697/0%、low-step 0.7255/2.0%、high-step 0.574/4.3%、low-climb-76 0.1886/100%、high-climb-76 0.161/100%。门禁（return>=0.1、失败<=80%）通过；两个攀爬类别 100% 失败，Stage2 的攀爬规划没有成功 hindsight 标签，相关指标不能作为效果结论。评审记录 `stage1_final_eval.review.json`。

离线 planner 20,000 步完成（0.395 h，`stage2_php_all_spheres224_20261110/offline/planner_020000.pt`），验证脚 F1 0.83/0.81、膝 0.32/0.25、手无正例；脚落点误差中位约 0.09–0.10 m，低于 M6 的脚 F1≥0.85/膝≥0.75 目标，作为联合训练初始化继续。

Stage2 联合训练第一段（24,000 轮）于 03:56 启动，单元 `acp-stage2-joint-20261111`，端口 29573，512/256/256、planner 512、容量 300k、64/80 GiB。稳定测速 1.62 轮/秒（50 轮 benchmark，30.9 s，2071 transitions/s），CPU 22 GiB、GPU 10.2/8.9 GiB。η_obs 于 2000 轮开始上升、η_rew 于 12000 轮开始；admission 8–12%，planner buffer 正常增长。预计 08:05 前后到 24k，随后用 `acp-stage2-joint-window-20261111` 续到 40k（term_window 2、eta_rew_max 0.58），全程约 10:50。流程见 `stage2_handoff.md`。

## 2026-11-11 10:36 Stage2 联合训练完成与评估

Stage2 两段跑满 40,000 轮（24k 严格 + 24k→40k 用 term_window 2、eta_rew_max 0.58），总计约 6.6 h，1.8 轮/秒；`joint/last.pt`、`ckpt_040000.pt` 完整保存，服务 Result=success。最终 Planner val：脚 F1 0.628/0.585、膝 0.281/0.314，较离线 20k 的 0.83/0.81、0.32/0.25 有退化。

补跑了同协议对照：Stage1 严格 0.5271/837 失败、Stage1 ±2s+噪声 0.5143/807、Stage2 ±2s+噪声 0.4290/801、Stage2 严格 0.3739/1545。同协议下失败数接近，但 Stage2 回报更低；严格协议下 Stage2 失败翻倍（主要是晚于参考）。接触落点脚 18.9/22.9 cm → 6.2/6.9 cm、≤0.1 m 占比 19.7%/13.0% → 67.4%/65.8%，是 Stage2 的真实收益。

攀爬仍未解决：low-climb 100% 失败；high-climb 在 ±2s 下表观 56.2% 通过，但存活片段 root RMSE 中位 1.95 m，属终止窗口放宽假象，严格窗口仍 100% 失败。结论：Stage2 不能宣称优于 Stage1；参考跟踪优先用 Stage1 断点。完整对照见 `stage2_results.md`，看护定时任务已暂停。后续建议先补攀爬专项、再排查 planner 在线 F1 退化与膝/手标签覆盖。

## 2026-11-11：停止后仅实施第一步课程改动

用户要求“commit一下，然后先完成第一步：改课程，让失败区得到足够训练。不要进行第二第三步”。先提交诊断 `014f344`；新增 opt-in 事件局部课程、从头攀爬与技能保留的固定环境配额。奖励、policy 接口和网络保持原样，不实施第二/第三步。训练保持停止，当前仅做 CPU 与无参数更新的双 GPU 仿真检查。细节、入口与验证见 transition_curriculum.md；未获得实际学习收益，不把局部课程门槛当作任务验收。
