"""Attention-based contact planner (ACP) for the G1 on box terrain: "Humanoid Mobility via Attention-based Contacts"
(He, Wen, Zhang et al., arXiv:2610.31608) trained with SuperMimic (actor gradients through imagined rollouts of one
learned world model), not PPO.

Two networks at 50 Hz. The contact planner reads a 17 x 11 height scan around the pelvis, 5 steps of proprioception and
a goal, and gives, for each of 9 bodies (feet, knees, hands, elbows, pelvis), the terrain point of its next contact
within 1.5 s, or "no contact". The contact tracker reads 10 steps of proprioception, the 9 targets and the goal, and
outputs joint position targets; it also sees the paper's 9 x 7 local height scan. Every quantity is in the pelvis
heading frame. The world model and the critic see an exact height scan of the episode's terrain in every stage.

Self-contained on mjlab (MuJoCo-Warp), next to the Isaac Lab code of :mod:`supermimic`; data: the PHP parkour G1
bundles (``data/php_parkour/motions-*``).

Modules:
    g1, kinematics          G1 entity, actuators and contact bodies; torch FK
    terrain, motion         clip boxes in all terrain variants, height scans; the motion library
    mdp, networks, buffer   observation, tracking reward, termination; world model, policy, critic; replay buffer
    env                     the box-terrain simulator
    task                    contact command, sphere geometry of the bodies, the paper's Stage 1 reward
    planner                 the attention planner, hindsight labels, its replay buffers
    goal                    Stage 3: procedural goal-reaching terrains, goal reward and termination, the curriculum

Scripts (``python -m supermimic.acp.<name>``), in pipeline order:
    make_labels             per-frame contact labels of every clip and terrain variant
    refine_motion           the clips cleaned against their terrain (no penetration, exact contacts); label them again
    train_stage1            Stage 1: the contact tracker
    train_planner_offline   Stage 2, offline: the planner from the frozen tracker
    train_stage2            Stage 2: planner and tracker co-training
    train_stage3            Stage 3: reference following and goal reaching on procedural terrains
    evaluate                every clip once, with reference or planner targets
    evaluate_goal           goal reaching on every procedural terrain type and level
    render, render_labels   videos of evaluate --record rollouts; of a motion set with its contact labels
"""
