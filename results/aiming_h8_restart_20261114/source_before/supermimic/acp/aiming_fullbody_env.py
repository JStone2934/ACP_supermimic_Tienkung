"""Nominal flat-ground simulator with no MotionLib construction or reference reads."""
import mujoco
import torch
import warp as wp
from mjlab.scene import Scene
from mjlab.sim import Simulation

from .env import TrackingSim, make_scene_cfg, make_sim
from .mdp import flatten_state
from ..rotations import mat_to_quat


class FullbodySim:
    # Reuse only physical state access and penetration measurement, never tracking lifecycle.
    state = TrackingSim.state
    _terrain_penetration = TrackingSim._terrain_penetration

    def __init__(self, task, num_envs, sampler, *, eval_mode=False):
        self.task, self.mdp, self.spec = task, task.mdp, task.robot
        self.device = str(task.alpha.device)
        self.N, self.num_joints = num_envs, self.spec.num_joints
        self.eval_mode, self.sampler = eval_mode, sampler
        self.scene = Scene(make_scene_cfg(num_envs, 0, self.spec), device=self.device)
        self.sim = Simulation(num_envs=num_envs, cfg=make_sim(robot=self.spec), model=self.scene.compile(), device=self.device)
        wp.set_mempool_release_threshold(self.device, torch.cuda.get_device_properties(self.device).total_memory)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        self.robot = self.scene['robot']
        internal = list(self.robot.joint_names)
        if set(internal) != set(self.spec.joint_names):
            raise ValueError('Simulator joint identity mismatch')
        self.to_canonical = torch.tensor([internal.index(n) for n in self.spec.joint_names], device=self.device)
        self.to_internal = torch.argsort(self.to_canonical)
        if not torch.allclose(self.robot.data.default_joint_pos[0, self.to_canonical], task.mdp.default_q):
            raise ValueError('Simulator default joint targets differ from RobotSpec')
        m = self.sim.mj_model
        self._terrain_geoms = torch.tensor([i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE], device=self.device)
        self.sim.expand_model_fields(('geom_friction',))
        self.sim.model.geom_friction[:, self._terrain_geoms, 0] = 1.
        self.sim.model.geom_friction[:, self.robot.indexing.geom_ids, 0] = 1.
        self.xi = torch.zeros(num_envs, self.spec.xi_dim, device=self.device); self.xi[:, 0] = 1.
        self.a_prev = torch.zeros(num_envs, 29, device=self.device)
        self.ep_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.context = {}
        self.reset(torch.arange(num_envs, device=self.device))

    @torch.no_grad()
    def reset(self, ids):
        if len(ids) == 0:
            return
        S = self.task.initial_state(len(ids))
        p, R, v, w, q, qd = S
        self.sim.reset(ids); self.scene.reset(ids)
        pose = torch.cat((p + self.scene.env_origins[ids], mat_to_quat(R)), -1)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=ids)
        self.robot.write_joint_position_to_sim(q[:, self.to_internal], env_ids=ids)
        self.robot.write_joint_velocity_to_sim(qd[:, self.to_internal], env_ids=ids)
        pose[:, 2] += self._terrain_penetration(ids)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=ids)
        self.robot.set_joint_position_target(q[:, self.to_internal], env_ids=ids)
        self.sim.forward()
        self.ep_step[ids] = 0
        self.a_prev[ids] = (q - self.mdp.default_q - self.xi[ids, 1:30]) / self.task.alpha
        parameters, weights = self.sampler(len(ids))
        ctx = self.task.context(tuple(x[ids] for x in self.state()), parameters, weights)
        if not self.context:
            self.context = {k: x.new_zeros((self.N, *x.shape[1:])) for k, x in ctx.items()}
        for k, x in ctx.items():
            self.context[k][ids] = x

    def observe(self):
        return self.mdp.obs(self.state(), self.a_prev, self.xi)

    @torch.no_grad()
    def step(self, raw):
        S = self.state()
        ctx = {k: x.clone() for k, x in self.context.items()}
        ap, t = self.a_prev.clone(), self.ep_step.clone()
        a, q_des = self.task.compose_action(raw, self.xi)
        o = self.mdp.obs(S, ap, self.xi)
        self.robot.set_joint_position_target(q_des[:, self.to_internal])
        for _ in range(self.spec.decimation):
            self.scene.write_data_to_sim(); self.sim.step()
            self.scene.update(self.spec.control_dt / self.spec.decimation)
        S1 = self.state()
        terms, contributions = self.task.reward_terms(S1, a, ap, ctx)
        terminated, timeout = self.task.termination(S1, t + 1, ctx)
        done = terminated | timeout
        out = dict(s=flatten_state(S), s1=flatten_state(S1), a=a.clone(), a_prev=ap, xi=self.xi.clone(), o=o,
                   r=sum(contributions.values()), ep_step=t, terminated=terminated, timeout=timeout, done=done,
                   f=torch.zeros_like(t), v=ctx['terrain_id'].clone(), goal_steps=torch.zeros_like(t),
                   goal=S[0].new_zeros(self.N, 4), reward_terms=terms, reward_contributions=contributions,
                   q_des=q_des, **ctx)
        # Terminal transition retains its old context; reset happens only after capture.
        self.ep_step = t + 1; self.a_prev = a.clone()
        if not self.eval_mode:
            self.reset(done.nonzero().flatten())
        return out
