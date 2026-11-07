# 天工接入任务记忆

更新日期：2026-11-07（Asia/Shanghai）。

## 授权与范围

用户确认交付 **M0–M3 接入与数值验证，加合成站立链路检查**。允许实施、建立 docs 子文档并管理 Git；目标远程为 https://github.com/JStone2934/ACP_supermimic_Tienkung 。真实动作转换、步行/攀爬训练、M4–M7 能力验收及实机部署不在本次范围。

## 工作区与 Git

- 起点 `main@751aa5c3ccb70ad8b9e74f8f1c53c9b61f35c459`，原暂存区为空。
- 原有 README、render、train_stage1 修改及环境/流水线工具按明确路径独立保存为基线提交；初始文件哈希见 [evidence/baseline.json](evidence/baseline.json)，原始 tracked diff 另存 `.git/tienkung3_initial.patch`。
- 开发分支 `feat/tienkung3-29dof-acp`。原远程保留为 `upstream`，用户指定仓库设为 `origin`。只做普通推送，不强推、不覆盖 main。
- 模型、网格、NPZ、checkpoint 留在被忽略的 data/logs；可复现工具、设计、manifest 与小型验证报告提交。

## 固定约定

- 机器人 `tienkung3_29dof`，G1 保持默认；配置按实例传递，不修改进程全局机器人状态。
- 模型固定 `Open-X-Humanoid/TienKung_URDF@4c2c061d72dcdab60d74e45d3efb17246a3b6f62`。头 yaw/pitch 固定在零位，质量和 frame 保留。
- 29 hinge + freejoint，nq=36、nv=35，组装后 nu=29；canonical 顺序见 migration_plan.md。
- 50 Hz 控制、200 Hz 物理。动作 `q_default + offset + 0.25*a`；控制参数引用与 URDF 冲突取保守上限，armature=0 为未辨识假设。
- 九类接触、14 个 tracked frame，FK 支持固定链与非零 joint anchor。
- 合成站立仅证明接口、数值和梯度链路；跌倒指标如实记录，不宣称训练成功。

## 环境与恢复

Python：`/home/js/miniforge3/envs/acp_supermimic/bin/python`。已探测 torch 2.9.0+cu128、MuJoCo 3.11.0、CUDA 可用，GPU 为 RTX PRO 6000 Blackwell。

恢复工作先读 [progress.md](progress.md)、[design.md](design.md)，核对 git status 与 evidence。不要将原 migration_plan 的未实施命令当成已实现接口。

## 已落实的关键修正

- STL 面数超限：在 generated/meshes 无损转换 OBJ，原始资产保留。
- 601 帧合成片段揭示末帧 reset bin 越界：bin 数按 T-1 个 transition 计算，terminal frame 映射至末 bin；eval 已结束环境 frame 不继续越界。
- 阶段证据见 progress/evidence；旧 G1 checkpoint 对比使用 `.cache/tienkung_g1_baseline` detached worktree，数据与 logs 为符号链接，不修改源数据。

- Git HTTPS 无终端凭据，但现有 SSH 凭据可用；origin fetch 保留 HTTPS，push URL 使用 git@github.com:JStone2934/ACP_supermimic_Tienkung.git，首批提交已普通推送。
- 接触近似 v3 加入凸包面/箱体/棱边分离方向；原生 CCD 偶发零距离且 witness 分离，离线查询使用有界微位移重查（libccd 交叉检查发现同样不可靠，未采用）。变更已更新几何身份，需使用 v3 重生的动作/标签和最终证据。

## 本轮交付完成状态

- 实现提交 `ca74a80`；M0–M3 和合成链路已验收，具体命令/源码哈希/报告哈希见 evidence/run_manifest.json。工作分支保持 `feat/tienkung3-29dof-acp`，后续文档提交在该实现提交之上。
- 最终 v3：1000 姿态 FK 通过，接触 900 样本 P95 0.0409 mm、最大 1.324 mm；15 项测试通过；1/16/64 env 各 500 步及 G1 回归通过。
- Tiny Stage 1 路径为 `logs/tienkung3_29dof/integration_stage1_v3`：3 次更新后恢复至第 4 次。旧的 integration_stage1 / integration_stage1_v2 是中间实验，不要用作最终模型身份的 checkpoint。
- 实际评估 1/1 片段失败，约 1.40 s；这不是稳定站立能力交付。录像在 results/tienkung3_tiny_policy.mp4，报告和哈希已提交，数据/权重/视频按既有规则留在本地。
- shell 使用绝对 Python 路径时不会自动加入环境 bin；渲染须 conda activate acp_supermimic 或显式把其 bin 加入 PATH，环境内已安装 ffmpeg。
- `--resume` 继续同一数据版本；`--init` 允许同模型的新数据微调，仍严格校验机器人身份。
- 本次授权范围没有剩余实施项。下一阶段等真实动作数据进入 M4；未启动正式训练或实机部署。

## 后续授权：PHP 小批量验证（2026-11-07）

用户已授权先选站立、平地行走、转向、低台阶各少量片段，验证重定向质量和跟踪效果，再扩展攀爬和296条。当前执行首批验证，设计/进度/证据在 php_pilot/；不会自动把未合格动作加入训练或直接开始全量。起点36219b6，保留现有数据和M0–M3证据。

### PHP 小批量停止记录

用户随后明确要求“停止任务吧，直接收尾”。已停止本轮训练，不安排自动恢复。详细状态见 php_pilot/README.md 和 php_pilot/progress.md。

7片段重定向/质检/标签/运动学参考录像完成。带两处默认模型自碰撞例外的初筛7/7；严格几何0/7、PD跟踪达标0/7。单站立日志至250轮、checkpoint仅保存到200轮；五片段训练100轮日志后显存不足，无checkpoint。本轮训练后评估与策略录像未完成。20项回归通过。恢复前先解决模型自碰撞和显存问题，不把参考录像当策略效果，不直接扩展296条。
