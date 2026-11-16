"""Replay a captured failed physical transition on CPU at several timesteps."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from supermimic.acp.env import make_scene_cfg
from supermimic.acp.robots import get_robot
from supermimic.rotations import quat_to_mat


def main():
    ap=argparse.ArgumentParser();ap.add_argument('capture');ap.add_argument('--out',required=True)
    args=ap.parse_args();saved=torch.load(args.capture,weights_only=False,map_location='cpu')
    rows=(~saved['wm_target'].isfinite()).any(-1).nonzero().flatten().tolist()
    robot=get_robot('tienkung3_29dof');scene=Scene(make_scene_cfg(1,0,robot),device='cpu')
    model=scene.compile();model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON;model.opt.iterations=100;model.opt.ls_iterations=50;model.opt.tolerance=1e-6
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)]
    ids=[next(i for i,n in enumerate(names) if n.split('/')[-1]==name) for name in robot.joint_names]
    qa=model.jnt_qposadr[ids];va=model.jnt_dofadr[ids]
    free=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0];fq=model.jnt_qposadr[free];fv=model.jnt_dofadr[free]
    bids=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(model.nbody)]
    torso=next(i for i,n in enumerate(bids) if n and n.split('/')[-1]==robot.torso_body)
    base_com=model.body_ipos[torso].copy()
    report=[]
    for row in rows:
        hist=saved['history'];starts=[i for i,h in enumerate(hist) if h['ep_step'][row]==0]
        if not starts:raise ValueError('capture does not reach reset')
        hist=hist[starts[-1]:];s=hist[0]['s'][row].numpy();xi=hist[0]['xi'][row].numpy()
        model.body_ipos[torso]=base_com+xi[-3:]
        for g in range(model.ngeom):model.geom_friction[g,0]=.3 if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE else xi[0]
        for substeps in (4,8,16,32):
            model.opt.timestep=.02/substeps;d=mujoco.MjData(model)
            d.qpos[fq:fq+7]=s[:7];d.qpos[qa]=s[13:42]
            d.qvel[fv:fv+3]=s[7:10];d.qvel[fv+3:fv+6]=quat_to_mat(torch.tensor(s[3:7])).numpy().T@s[10:13];d.qvel[va]=s[42:]
            mujoco.mj_forward(model,d);trace=[]
            for h in hist:
                target=np.array(robot.default_q)+xi[1:30]+np.array(robot.action_scales)*h['a'][row].numpy()
                for act in range(model.nu):d.ctrl[act]=target[ids.index(model.actuator_trnid[act,0])]
                for _ in range(substeps):mujoco.mj_step(model,d)
                trace.append(dict(time=float(d.time),max_q=float(np.max(np.abs(d.qpos[qa]))),max_qd=float(np.max(np.abs(d.qvel[va]))),max_torque=float(np.max(np.abs(d.actuator_force))),root_z=float(d.qpos[fq+2]),warnings=int(sum(w.number for w in d.warning))))
            report.append(dict(row=row,substeps=substeps,trace=trace))
    Path(args.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
