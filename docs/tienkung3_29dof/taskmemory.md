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
