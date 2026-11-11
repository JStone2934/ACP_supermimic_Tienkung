# PHP 小批量重定向：停止时交付

**2026-11-07 按用户要求停止实验并收尾。** 已完成7个片段的重定向、独立几何质检、自由根部PD基线、标签和参考录像；短训练没有完成，不宣称学习策略跟踪成功。

## 结果

| 类别 | 片段数 | 重定向时长 | 初步几何筛选 | 严格自碰撞 | PD跟踪达标 |
|---|---|---|---|---|---|
| 站立 | 1 | 1.98 s | 1/1 | 0/1 | 0/1 |
| 直行 | 2 | 3.96 s | 2/2 | 0/2 | 0/2 |
| 左右转向 | 2 | 3.36 s | 2/2 | 0/2 | 0/2 |
| 低台阶 | 2 | 14.96 s | 2/2 | 0/2 | 0/2 |

共1220帧，逐片段有效时长合计24.26 s，50 Hz。初步筛选明确排除两处默认模型已有的小腿—脚部凸包重叠；原始自碰撞仍完整记录，**不能把7/7初筛通过写成M4严格验收通过**。最大地形穿透约1.85 mm，各片段支撑距离P95不超过1.32 mm、支撑脚切向速度P95不超过0.014 m/s，无关节硬限位或速度上限超出。

G1原始同片段在相同自由根部PD协议下也不能稳定跟踪；该基线不代替学习控制器。天工PD漂移/终止及所有误差见 [quality_final.json](evidence/quality_final.json)。其中 `source_motion_hash` 字段指本次被检查的目标motion文件哈希，原始G1来源哈希另见 [selection_and_retargeting.json](evidence/selection_and_retargeting.json)。

原始动作和台阶位置/尺寸保留，固定使用地形第0变体。首轮的摆臂碰髋、脚部穿地和速度尖峰已通过中立/外展手臂姿态、加强脚部穿透及速度惩罚修正。当前版本只针对腿部动作：平地固定中立手臂，台阶固定肩roll为左+0.5、右−0.5 rad；没有保留源摆臂风格，不能据此扩展手接触攀爬。

## 训练停止状态

- 单站立：日志记录至250轮，用户停止后已中断进程；最后持久化的完整checkpoint为第200轮，位于 `logs/tienkung3_29dof/php_pilot_stand_v2/ckpt_000200.pt`，`last.pt` 同为200轮。未完成原计划800轮，也未评估该checkpoint。
- 五片段平地联合训练：日志记录至100轮，随后在Warp CUDA图启动时显存不足退出；没有checkpoint，未完成原计划1000轮。错误日志保留于 [flat_training.log](evidence/flat_training.log)，未继续诊断或重试。
- [standing_prior_tiny_policy.json](evidence/standing_prior_tiny_policy.json) 是先前M0–M3仅更新4轮权重在新站立片段上的入口检查，结果失败；不是本轮200轮checkpoint的结果。
- 20项回归测试通过，见 [tests.txt](evidence/tests.txt)。停止时不再启动新实验或补做策略录像。

## 本地文件与录像

- 最终动作：`data/tienkung3_29dof/motions/php_pilot_final/`。
- 重新生成的标签：`data/tienkung3_29dof/motions/php_pilot_final_contacts/`。
- **参考录像**：`results/php_pilot_final/reference.mp4`，含7个片段，是运动学回放，不含策略控制或动力学跟踪成功含义。
- PD回放轨迹：`results/php_pilot_final/<类别>/<片段>_pd.npz`；保留用于后续渲染，本轮未生成其MP4。
- `results/php_pilot_v2/flat_reference.mp4` 是五片段参考预览。
- 本轮没有训练后策略的评估录像；之前 `results/tienkung3_tiny_policy.mp4` 属于M0–M3的4轮检查。

所有大文件按原规则留在本地，源码/设计/命令/小型报告及哈希提交Git。来源家族不足以形成独立80/10/10划分，本批全部标为开发集，不宣称泛化。

## 复现入口（仅供后续主动恢复）

```bash
conda activate acp_supermimic
# 使用新的输出目录，避免覆盖本轮证据。
python -m tools.retarget_php_pilot --out data/tienkung3_29dof/motions/php_pilot_repro \
  --only standing/stand walking/forward_a walking/forward_b turning/left45 turning/right45 \
  --arm-mode neutral --iterations 1000
python -m tools.retarget_php_pilot --out data/tienkung3_29dof/motions/php_pilot_steps_repro \
  --only low_step/forward_a low_step/forward_b --arm-mode clearance --iterations 1200
python -m tools.check_php_pilot --motion-dir data/tienkung3_29dof/motions/php_pilot_repro \
  --out results/php_pilot_repro_quality.json --record-dir results/php_pilot_repro
python -m supermimic.acp.make_labels --robot tienkung3_29dof \
  --motion_dir data/tienkung3_29dof/motions/php_pilot_repro --workers 1
python -m tools.record_motion_reference data/tienkung3_29dof/motions/php_pilot_repro results/php_pilot_repro_reference.npz
python -m supermimic.acp.render results/php_pilot_repro_reference.npz results/php_pilot_repro_reference.mp4 \
  --robot tienkung3_29dof --per_folder 2 --solid-label 'reference IK'
```

训练参数分别保存在 evidence/standing_args.json、flat_args.json，当前配置已出现显存不足，后续需先诊断并降低试验负载。没有在后台安排恢复、全量重定向或攀爬训练。
