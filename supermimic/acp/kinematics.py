"""Differentiable full-tree FK, including fixed frames and offset hinge anchors."""
from __future__ import annotations
import mujoco
import numpy as np
import torch
from .robots import get_robot
from ..rotations import axis_angle_to_mat, quat_to_mat


class RobotKinematics:
    def __init__(self, device: str, robot='g1', tracked=None, *, model=None, dtype=torch.float32):
        self.robot = get_robot(robot)
        m = model if model is not None else self.robot.get_spec().compile()
        self.model, self.device = m, device
        names = {n:i for i,n in enumerate(self.robot.joint_names)}
        root = m.body(self.robot.root_body).id
        self.body_ids = []
        for b in range(1,m.nbody):
            a = b
            while a and a != root: a = m.body_parentid[a]
            if a == root: self.body_ids.append(b)
        self.body_names = [m.body(b).name for b in self.body_ids]
        local = {b:i for i,b in enumerate(self.body_ids)}
        self.parent, self.joint_idx = [], []
        axes, anchors, positions, rotations = [], [], [], []
        for b in self.body_ids:
            nj = m.body_jntnum[b]
            jid = m.body_jntadr[b]
            if nj > 1: raise ValueError(f'{m.body(b).name}: multiple joints per body unsupported')
            if b == root:
                if nj != 1 or m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                    raise ValueError('root must have exactly one freejoint')
                par, j = -1, -1
            else:
                par, j = local[m.body_parentid[b]], -1
                if nj:
                    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE: raise ValueError('only hinge/fixed descendants supported')
                    j = names[m.joint(jid).name]
            self.parent.append(par); self.joint_idx.append(j)
            axes.append(m.jnt_axis[jid].copy() if nj else np.zeros(3))
            anchors.append(m.jnt_pos[jid].copy() if nj else np.zeros(3))
            positions.append(m.body_pos[b].copy()); rotations.append(m.body_quat[b].copy())
        tensor = lambda x: torch.tensor(np.asarray(x),dtype=dtype,device=device)
        self.axes, self.anchors, self.off_p = tensor(axes),tensor(anchors),tensor(positions)
        self.off_R = quat_to_mat(tensor(rotations))
        self.tracked_names = list(tracked if tracked is not None else self.robot.tracked_bodies)
        self.tracked_idx = [self.body_names.index(n) for n in self.tracked_names]
        self.num_bodies = len(self.body_ids)
        self.q_lo = tensor([m.jnt_range[m.joint(n).id,0] for n in self.robot.joint_names])
        self.q_hi = tensor([m.jnt_range[m.joint(n).id,1] for n in self.robot.joint_names])
        self.q_ref = tensor([m.qpos0[m.jnt_qposadr[m.joint(n).id]] for n in self.robot.joint_names])

    def forward(self,p,R,v,w,q,qd,all_bodies=False):
        P, Rs, V, W = [None]*self.num_bodies,[None]*self.num_bodies,[None]*self.num_bodies,[None]*self.num_bodies
        for i,par in enumerate(self.parent):
            if par < 0:
                P[i],Rs[i],V[i],W[i] = p,R,v,w
                continue
            j = self.joint_idx[i]
            base_R = Rs[par] @ self.off_R[i]
            offset = (Rs[par] @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            base_p = P[par] + offset
            if j < 0:
                P[i],Rs[i] = base_p,base_R
                W[i] = W[par]
                V[i] = V[par] + torch.cross(W[par],offset,dim=-1)
            else:
                Ri = base_R @ axis_angle_to_mat(self.axes[i],q[...,j]-self.q_ref[j])
                anchor0 = (base_R @ self.anchors[i].unsqueeze(-1)).squeeze(-1)
                anchor1 = (Ri @ self.anchors[i].unsqueeze(-1)).squeeze(-1)
                axis = (base_R @ self.axes[i].unsqueeze(-1)).squeeze(-1)
                wi = W[par] + axis * qd[...,j:j+1]
                P[i],Rs[i],W[i] = base_p + anchor0 - anchor1,Ri,wi
                va = V[par] + torch.cross(W[par],offset+anchor0,dim=-1)
                V[i] = va - torch.cross(wi,anchor1,dim=-1)
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return tuple(torch.stack([xs[i] for i in idx],dim) for xs,dim in ((P,-2),(Rs,-3),(V,-2),(W,-2)))


class G1Kinematics(RobotKinematics):
    """Compatibility entry point for existing G1 callers."""
    def __init__(self,device: str,tracked=None):
        super().__init__(device,'g1',tracked)
