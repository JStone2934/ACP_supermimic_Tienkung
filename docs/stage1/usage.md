# Stage1 操作

2026-11-10 224 球更新：当前运行目录是 `logs/tienkung3_29dof/stage1_php_all_spheres224_20261110/`，`evidence/launch_config.json` 已绑定新几何与源码；服务名称不变。旧目录保留为密集版本历史记录及第 500 轮迁移来源。启动控制器会先完成全量检查、短训和迁移恢复检查，具体见 [spheres224_migration.md](spheres224_migration.md)。

20:00 执行优化后，当前配置使用 `reward_chunk_size=0`，通过同源断点完整更新对照和实际奖励梯度检查。运行目录和服务不变；已通过的启动阶段保留，正式从本目录最新 `train/last.pt` 恢复。旧分块配置保存为 `evidence/launch_config_spheres224_chunk64.json`，当前固定副本为 `evidence/launch_config_spheres224_unchunked_20261110.json`。

使用 `/home/js/miniconda3/envs/acp_supermimic/bin/python`，工作目录为项目根目录。正式训练只启动一个双卡同步作业。

数据准备：

```bash
/home/js/miniconda3/envs/acp_supermimic/bin/python -m tools.prepare_stage1_dataset \
  --delivery docs/php_conversion/evidence/delivery_summary_20261110.json \
  --out data/tienkung3_29dof/stage1/php_all_variants_20261110
```

目标目录存在时命令会拒绝覆盖；复用前需核对 manifest 与输入哈希，不能重复打包或重复启动训练。

后台服务于 2026-11-10 18:01:20 北京时间启动。检查流程通过后自动进入正式训练。服务 `MemoryHigh=64 GiB`、`MemoryMax=80 GiB`，同时受原共享 slice 的 96/104 GiB 与用户级主机限制保护；没有修改现有保护。程序异常会明确失败并停止，不自动重复一个持续出错的任务。

```bash
systemctl --user status acp-stage1-php-all-20261110.service --no-pager
journalctl --user -u acp-stage1-php-all-20261110.service -n 30 --no-pager
cat logs/tienkung3_29dof/stage1_php_all_spheres224_20261110/status.json
tail -5 logs/tienkung3_29dof/stage1_php_all_spheres224_20261110/train/train_log.csv
```

完整控制器启动参数在 `evidence/systemd_command.json`。`preflight`、`smoke`、`resume_smoke`、`training` 的实际参数数组和 stdout 位于运行根目录，各阶段通过后写 `.done.json`。短训检查点不作为正式训练初始化。

需要停止时：

```bash
systemctl --user stop acp-stage1-php-all-20261110.service
```

恢复前先检查服务确已停止、原输入与代码绑定未变，再启动同一个控制器及配置。若 unit 仍存在，使用 `systemctl --user start acp-stage1-php-all-20261110.service`；若临时 unit 已回收，重放 `evidence/systemd_command.json` 中的命令。控制器复用已通过阶段，正式训练检测到 `train/last.pt` 时添加 `--resume`，不使用新的运行目录。没有 checkpoint 时只能从头开始该阶段。

`last.pt` 保存完整网络、优化器、课程、归一化及计数。恢复需保持原始输入和全局配置；replay 与仿真状态会重建。停止训练使用其 systemd unit，不直接杀某个 torchrun rank。不得停止 `php-codex.service` 来管理训练进程。


20:46 超参对照后仍使用原512环境/256+256 batch/WM10/UTD0.4。从原3000轮断点恢复，未采用实验权重。当前固定配置副本为 `evidence/launch_config_post_hyperparameter_review_20261110.json`，最新恢复决策见 `evidence/hyperparameter_tuning_receipt.json`。
