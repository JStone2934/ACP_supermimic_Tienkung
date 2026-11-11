# Stage1 资产列表

| 资产 | 路径 / 用途 |
|---|---|
| 原交付收据 | `docs/php_conversion/evidence/delivery_summary_20261110.json` |
| 原最终索引 | `docs/php_conversion/evidence/all_original_variants_final_20261110.index.json` |
| 数据打包工具 | `tools/prepare_stage1_dataset.py` |
| 训练数据目标目录 | `data/tienkung3_29dof/stage1/php_all_variants_20261110/` |
| 数据身份与映射 | 上述目录 `manifest.json`，每项含源编号、父批次、槽位、文件哈希 |
| Stage1 入口 | `supermimic/acp/train_stage1.py` |
| 双卡资源依据 | `docs/dual_rtx5090/README.md` |
| 训练证据 | `docs/stage1/evidence/` |
| 当前运行根目录 | `logs/tienkung3_29dof/stage1_php_all_spheres224_20261110/` |
| 旧密集几何运行目录 | `logs/tienkung3_29dof/stage1_php_all_20261110/`，完整保留，提供第 500 轮迁移断点 |
| 224 球切换记录 | `docs/stage1/spheres224_migration.md` |
| 新旧配置 | `evidence/launch_config_spheres224_20261110.json` / `evidence/launch_config_dense_20261110.json` |
| 新几何回归 / 参考姿态筛查 | `evidence/spheres224_regression.txt`（22 项通过）/ `evidence/spheres224_no_hull_probe.json` |
| 224 球初步速度 | `evidence/spheres224_initial_throughput.json`，短训第 2/3 轮约 1.37/1.29 秒，仅初估 |
| 224 球正式速度与健康 | `evidence/spheres224_formal_throughput.json`，50 轮平均 1.302 秒，第 600 轮时剩余约 21.5 小时 |
| GPU 利用率与执行优化 | `execution_tuning.md`、`evidence/gpu_utilization_before_tuning.csv` |
| 奖励执行等价检查 | `tools/check_stage1_reward_execution.py` / `evidence/reward_execution_equivalence.json` |
| 执行对照与分阶段耗时 | `logs/tienkung3_29dof/stage1_execution_probe_20261110/`，同第 1000 轮源断点、同数据/参数 |
| 执行优化采用收据 | `evidence/execution_tuning_receipt.json`，整批计算比 64 行分块快 2.61 倍 |
| 当前正式速度与利用率 | `evidence/execution_tuning_formal_throughput.json`、`execution_tuning_formal_benchmark.json`、`execution_tuning_gpu_utilization.json`，0.421 秒/轮、GPU 约 49%/41% |
| 224 球接受与启动证据 | `evidence/spheres224_geometry_acceptance.json` / `evidence/spheres224_training_start_receipt.json`，5 个启动检查阶段通过 |
| 后台服务 | `acp-stage1-php-all-20261110.service`，属于 `php-compute.slice` |
| 启动配置 / 代码哈希 | `docs/stage1/evidence/launch_config.json` |
| 实际服务启动命令 | `docs/stage1/evidence/systemd_command.json`，参数数组 |
| 旧密集几何启动收据 | `docs/stage1/evidence/training_start_receipt.json`，绑定当时数据、代码、配置、短训及恢复 |
| 旧密集几何首批训练健康 | `docs/stage1/evidence/first_training_health.json` |
| 旧密集几何初步耗时 | `docs/stage1/evidence/initial_throughput.json`，短训第 2/3 轮约 44–45 秒/轮 |
| 旧密集几何正式耗时 | `docs/stage1/evidence/eta_reestimate_20261110_1845.json`，当时正式窗口约 43.53 秒/轮、剩余约 30 天 |
| 几何计算规模 | `docs/stage1/evidence/geometry_workload_20261110.json` |
| 降采样对照工具 / 结果 | `tools/probe_stage1_contact_sampling.py` / `docs/stage1/evidence/contact_sampling_probe_20261110.json` |
| 降采样解释 | `docs/stage1/contact_sampling.md`，224 点存在错误，2304 点仅通过本次参考姿态筛查 |
| G1 小球图像 / 视频 / 坐标 | `results/stage1_g1_contact_spheres_20261110/`，全身对照、局部近景、8 秒旋转视频与 224 球坐标 CSV |
| G1 小球渲染与复核 | `tools/render_g1_reward_spheres.py` / `docs/stage1/evidence/g1_spheres_visualization.json` |
| 几何替代方案检索 | `docs/stage1/contact_geometry_options.md`，box、胶囊、凸包、SDF、可微凸体与来源 |
| 天工 224 球候选图与坐标 | `results/stage1_tienkung_contact_options_20261110/`，3 张图、224 球 CSV、哈希清单；未推广 |
| 天工候选图渲染与复核 | `tools/render_tienkung_contact_candidate.py` / `docs/stage1/evidence/tienkung_224_visualization.json` |
| 数据打包记录 | `docs/stage1/evidence/dataset_build.log`、`dataset_summary.json` |
| 映射测试 | `docs/stage1/evidence/dataset_tests.txt`，3 项通过 |
| 启动回归 | `docs/stage1/evidence/startup_regression_tests.txt`，18 项通过；包括资产哈希复用失效检测 |
| 当前源码快照 | 当前运行根目录 `source_snapshot_unchunked.zip`；224 球初版 `source_snapshot_spheres224.zip` 及旧密集版本快照保留 |
| 当前全量装载证据 | 当前运行根目录 `dataset_preflight.json`，2960/2960、224 球、0 个奖励凸包 |
| 后台控制器 | `tools/run_stage1_training.py`，互斥锁阻止重复执行 |
| 实时状态 | 运行根目录 `status.json`，同时刷新本目录 `progress.md` |
| 各阶段命令和日志 | 运行根目录 `<phase>.command.json` / `<phase>.stdout.log` |
| 短训 / 恢复证据 | 运行根目录 `smoke.done.json` / `resume_smoke.done.json` |
| 旧断点迁移 / 恢复证据 | 当前运行根目录 `migration_smoke.done.json` / `migration_resume_smoke.done.json` |
| 正式指标 | 运行根目录 `train/train_log.csv`、`train/updates.json`、`train/benchmark.json` |
| 正式模型 | 运行根目录 `train/last.pt`、`train/ckpt_XXXXXX.pt`，每 500 轮完整保存 |

打包后的 motions、contacts 均为训练副本，占约 781 MiB。原转换数据和验收文件保持原位。尚未生成的模型、测速或证据文件不视为已完成。


## 超参效率实验（2026-11-10）

| 资产 | 位置 |
|---|---|
| 对照协议和质量门槛 | `hyperparameter_tuning.md` |
| 三组短训配置/停止收据 | `evidence/hyperparameter_probe_config.json` / `hyperparameter_stop.json` |
| 三组训练、评估及 GPU 采样 | `logs/tienkung3_29dof/stage1_hyperparam_probe_20261110/` |
| 追加仿真并行度实验 | `evidence/hyperparameter_followup_config.json`、`logs/tienkung3_29dof/stage1_hyperparam_followup_20261110/` |
| 断点逐字段等价验证 | `evidence/hyperparameter_resume_identity.json` |
| 有界顺序实验入口 | `tools/bench_stage1_hyperparams.py`；首轮冻结副本在实验目录 `benchmark_entry.py` |
| 超参前正式配置 | `evidence/launch_config_before_hyperparameters_20261110.json` |
| 超参前正式参数/命令/benchmark | 正式目录中的 `*_before_hyperparameters.json` |
| 四组最终判定与正式恢复 | `evidence/hyperparameter_tuning_receipt.json`，保留原配置 |
| 超参审查后当前固定配置 | `evidence/launch_config_post_hyperparameter_review_20261110.json` |
| 超参审查后正式速度 | `evidence/hyperparameter_review_formal_throughput.json` / `hyperparameter_review_formal_benchmark.json` |


## Stage2 对齐审查

| 资产 | 位置 |
|---|---|
| 对齐结论、修正与待验证项 | `stage2_alignment.md` |
| 真实断点、224球与奖励梯度 | `evidence/stage2_alignment_cpu.json` |
| 13项CPU回归 | `evidence/stage2_alignment_tests.json` |
| 未启动的Stage2配置草案 | `evidence/stage2_prepared_config.json` |
| 旧Stage2源文件与启动绑定 | `evidence/stage2_alignment_before/` |
| 源码绑定与当前完整快照 | `evidence/stage2_alignment_binding.json`；运行目录 `source_snapshot_stage2_alignment.zip` |
| 共享衔接契约 | `supermimic/acp/stage2_contracts.py` |
| 单测 / 真实资产检查工具 | `tools/test_stage2_alignment.py` / `tools/check_stage2_alignment.py` |
