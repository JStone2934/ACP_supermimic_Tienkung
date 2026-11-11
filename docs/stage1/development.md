# Stage1 实现与验证

## 数据装配

`tools.prepare_stage1_dataset` 首先验证最终交付收据绑定的验收报告和资产索引哈希。每个 `(source_key, source_variant)` 只生成一个样本，拒绝重复或缺失。动作文件直接复制，地形选取原 `export_slot`；标签的 variant 轴字段使用相同槽位切片，逐帧公共字段保留。标签中的 motion/terrain hash 绑定输出文件，quality hash 保留父批次的验收来源。

写出后重新读入，逐数组比较标签和地形，检查完整帧区间及 50 Hz。数据清单保存输出 SHA-256、父批次、原编号及原导出槽。当前 2960 项、1812360 展开帧；原始独立动作帧总量仍为 181236，展开帧数不能解读成更多独立动作。

`tools.check_stage1_dataset` 再验证输出文件，实际构造 MotionLib、ContactTask，核对所有样本均可采样，并逐帧检查 93 维接触命令有限。现有 MotionLib 几何筛选规则没有更改。

## 启动控制器

`tools.run_stage1_training` 持有运行目录排他锁，以子进程依次执行数据检查、3 轮双卡短训、恢复至 4 轮、正式 60000 轮。短训使用完整正式数据和相同 batch/horizon，仅立即启用 actor 以覆盖更新路径。正式训练重新初始化，保留原 500 轮 WM-only。

短训要求 WM/policy/critic 更新数分别达到 30/3/60，恢复后达到 40/4/80；检查参数有限、跳过数为零、双卡状态摘要存在且 checkpoint 的 data_hash 匹配全量装载报告。控制器只在检查成功后写阶段完成标记。

训练算法入口没有改动。实际参数数组、完整源码哈希、配置及快照可从 evidence/ 和运行目录追溯。后台进程每分钟更新 progress.md 与 status.json，actor 进入稳定更新区间后计算近期速度与剩余时间。训练达到轮数时状态为 `training_finished_evaluation_pending`，不会自动把策略能力标记为通过。

## 模型资源哈希复用

全量数据装载会逐片段验证机器人身份。原实现每次重新读取全部模型资源。`supermimic/acp/robots.py` 现在仅对未发生身份变化且已稳定超过一秒的模型文件复用进程内摘要；device、inode、size、mtime_ns、ctime_ns 任一变化都失效。最近写入的文件完整双读，校验前后检查身份。动作、地形和标签的 SHA-256 保持原计算路径。

测试曾发现快速同大小写入可被文件系统赋予相同时间戳，增加最近写入窗口后通过，初次失败日志保留。最终 18 项测试通过：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/js/miniconda3/envs/acp_supermimic/bin/python -m unittest \
  tools.test_robot_asset_cache tools.test_robot_contracts tools.test_stage1_dataset
```

全量 preflight 实测 61.08 秒通过，4259645952 字节为该检查的 PyTorch 显存峰值。训练显存与速度须以实际训练结果为准。
