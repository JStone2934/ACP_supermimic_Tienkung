# 接触几何方案检索与天工 224 球可视化

后续决定：用户已接受本页图示 224 球，并选择与 G1 一致的球距离路径。当前切换详情见 [spheres224_migration.md](spheres224_migration.md)。本页图片“候选/未采用”字样和正文保留生成时的历史状态。

检索日期：2026-11-10。用户要求比较点、球以外的常见表示，特别是 box，并画出天工 + 224 球。以下区分外部来源、项目现状和工程建议；本次没有推广奖励几何变更。

## 检索结果

| 表示 | 已有实现 / 研究依据 | 用于本项目的收益与限制 |
|---|---|---|
| 有向盒 OBB / 多个 box | MuJoCo 支持 box 与多种基本体碰撞；Isaac Sim 支持 Bounding Cube。盒子随身体旋转。 | 建议优先研究脚掌：连续平面覆盖，适合足底与台阶。单盒可能把圆角或足弓外的空间也算作实体，导致提前接触；须按原生碰撞几何拟合并验证。 |
| 胶囊 capsule / 圆柱 / 椭球 | MuJoCo 支持这些基本体；DiffPills 给出胶囊及加厚多边形的可微接近度和最近点。 | 胶囊适合长条四肢，可用少数形状代替密集表面。统一圆截面不能准确表达扁平外壳、关节凸起和足底；可以按部位混用。 |
| 简化凸包 / 多凸体组合 | MuJoCo 对一般 mesh 使用凸包；非凸形状可拆为多个凸体。Isaac Sim 也提供 Convex Decomposition。 | 连续表面比稀疏小球更完整；复杂度由凸体数、面/顶点数决定。原生碰撞查询不自动成为当前 PyTorch 奖励的可微算子。 |
| SDF / 体素距离场 | Isaac Sim 提供 SDF Mesh；cuRobo v0.7.6 文档列出 mesh、nvblox、欧氏距离体素网格的 GPU 查询。 | 可预计算每个刚体局部距离场，表达更复杂外形。体素精度、内存、插值梯度及物体对距离查询仍需设计；有 SDF 不等于不再需要采样或最近点求解。 |
| 加厚多边形 / 圆角凸体、可微凸体求解 | DiffPills（2022）和 DCOL（ICRA 2023）处理可微几何查询；DCOL 支持多面体、胶囊、圆柱、圆锥、椭球、加厚多边形。 | 为脚掌等部位提供可微接触近似的研究路线；求解器的 GPU 批量实现和速度需实测，不能把论文单次查询计时当作本项目训练速度。 |

另一条路线是直接用模拟器接触力/接触状态构造奖励：Isaac Lab ContactSensor 可返回刚体接触力并筛选接触对象。这适合访问模拟器状态的训练或评估；当前 Stage1 在世界模型预测状态上计算奖励并反传，因此不能直接用现有仿真接触力读取替代。接触力方案改变的是奖励信息来源，不是另一种几何基本体。

主要来源：

- [MuJoCo：碰撞表示与凸分解](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection)。支持 box、capsule、ellipsoid 等，说明 mesh 凸包和多凸体路线。
- [Isaac Sim：Physics Simulation Fundamentals](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamentals.html)。提供 Bounding Cube、Sphere Approximation、Convex Decomposition、SDF Mesh。
- [Omni Physics：Colliders](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/dev_guide/rigid_bodies_articulations/collision.html)。给出基本形状、网格近似及后端兼容限制。
- [cuRobo v0.7.6：Collision World Representation](https://nvlabs.github.io/curobo/v0.7.6/get_started/2c_world_collision.html)。注意此处 cuboid 是环境表示，查询输入仍为球；不是机器人采用 box 的实例。
- [DCOL 原论文](https://arxiv.org/abs/2207.00669)、[全文](https://arxiv.org/html/2207.00669v3)。核心碰撞量为达到相交所需的最小统一缩放系数，不可未经换算直接替代以米计的距离阈值。
- [DiffPills 原论文](https://arxiv.org/abs/2207.00202)。胶囊及加厚多边形的可微凸二次规划方法。
- [Isaac Lab：Contact Sensor](https://isaac-sim.github.io/IsaacLab/v2.0.1/source/overview/sensors/contact_sensor.html)。刚体接触力读取与对象过滤。

## 项目现状与建议

项目当前的地形已经是平面和 box。天工奖励使用 353146 个零半径表面点、9 组凸包穿透计算；G1 奖励使用 224 个球。减少点数不会自动消除 `hull_box_penetration` 的计算，也不会按点数缩减比例加速整轮训练。

建议对照三条路线：现有 2304 表面点候选作为改动较小的基线；脚部 1–3 个拟合 box、膝/肘组少量胶囊、手和骨盆简化凸包的混合表示；进一步研究批量可微凸体距离。上述数量只是待拟合的工程起点，不是已验收参数。均保持九个接触语义组与原生参考几何，并以原距离/接触标签检查误差。

足底 box 与平面的最小有符号高度可以用盒中心高度减去三个半轴在世界 z 方向投影的绝对值之和。面对 box 地形则须处理面、边之间的最近距离及穿透，不能仅测八个角点；一般 SAT 分离量也不能不经验证就当作欧氏距离。硬 min/abs 和接触特征切换处的梯度，需要用有限差分及扰动姿态验证。

下一步推广任何方案前需覆盖原始接触、台阶边缘、脚跟/脚尖、手肘膝支撑、随机姿态扰动，检查距离误差、接触误判/漏判、奖励及梯度，再做小批 GPU 完整更新和策略 rollout。优化后 ETA 必须来自完整训练迭代测量。

## 天工 224 球图的定义与限制

这张图复现 `tools/probe_stage1_contact_sampling.py` 中的 `g1_style_224` 候选：

- 左右脚各 28 球；左右膝、左右手、左右肘各 24 球；骨盆 24 球，总计 224。
- 从天工各部位原碰撞凸包顶点做最远点采样，球心朝顶点均值方向移入 10 mm，全部半径 10 mm。
- 属于相同总数预算的天工候选，未直接复制 G1 球心或各部位分配。G1 每脚 4 球、每手 48 球；脚球半径 5 mm，其余 10 mm。
- 球心、半径均按实际数值渲染，没有放大；姿态为看清部位而设置，并非策略演示。头、胸、大腿不属于当前九组接触奖励。
- 该近似不保证包覆完整表面，也不保证每个球均完全位于原凸包内部。现有 120 姿态 / 1080 次部位检查中有 4 次原生非接触被判为接触（左脚 3、右脚 1）；尚未验证梯度或策略 rollout。
- 候选筛查保留原凸包穿透分支；图只展示稀疏小球这一部分，不表示已删除凸包计算，也不是新增的物理碰撞模型。

图片、坐标与证据：

- [天工全身：外观 / 叠加 / 仅球](../../results/stage1_tienkung_contact_options_20261110/01_tienkung_224_overview.png)
- [脚、手、膝、肘、骨盆近景](../../results/stage1_tienkung_contact_options_20261110/02_tienkung_224_closeups.png)
- [G1 与天工同相机、同尺度对照](../../results/stage1_tienkung_contact_options_20261110/03_g1_tienkung_224_comparison.png)
- [224 球局部 / 世界坐标和半径 CSV](../../results/stage1_tienkung_contact_options_20261110/tienkung_224_candidate.csv)
- [产物与源码哈希](../../results/stage1_tienkung_contact_options_20261110/manifest.json)、[坐标复核记录](evidence/tienkung_224_visualization.json)、[原候选实测](evidence/contact_sampling_probe_20261110.json)

复现命令：

```bash
/home/js/miniconda3/envs/acp_supermimic/bin/python -m tools.render_tienkung_contact_candidate \
  --out results/stage1_tienkung_contact_options_20261110
```

本次只新增诊断渲染与文档，原 G1 渲染器增加可选相机配置以实现同尺度对照，默认值不变。正式训练仍由同一 `acp-stage1-php-all-20261110.service` 执行。
