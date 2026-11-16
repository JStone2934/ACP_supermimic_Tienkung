"""Offline full-grid reachability with one shared proximal-arm pose and fixed body."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256
from supermimic.acp.wide_curriculum import grid
from supermimic.rotations import quat_to_mat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:1')
    a = p.parse_args(); torch.set_num_threads(2); torch.manual_seed(319)
    torch.backends.cuda.matmul.allow_tf32 = False
    b = Path(__file__).resolve().parent
    out = b/'shared_proximal_geometry.json'
    if out.exists(): raise FileExistsError(out)
    source = b/'balance_pose_center_candidates/ckpt_000002.pt'
    ck = torch.load(source, map_location=a.device, weights_only=False)
    task = WideAimingTask(a.device, ck['config'])
    trajectory = b/'evaluation_stance_bias/inward_0.25/seed_17.npz'
    with np.load(trajectory) as z:
        def state(frame):
            root = torch.tensor(z['root'][frame, 0], device=a.device)[None].repeat(75, 1)
            quat = torch.tensor(z['root_quat'][frame, 0], device=a.device)[None].repeat(75, 1)
            q = torch.tensor(z['q'][frame, 0], device=a.device)[None].repeat(75, 1)
            return root, quat_to_mat(quat), torch.zeros_like(root), torch.zeros_like(root), q, torch.zeros_like(q)
        initial, S = state(0), state(500)
        actual_quat = z['root_quat'][500, 0].copy()
    params = grid(5, a.device); ctx = task.context(initial, params, 4.)
    nominal = S[4].clone()
    nominal[:, task.arm_ids] = torch.tensor(ck['calibration']['arm_pose_rad'], device=a.device)
    proximal, distal = task.arm_ids[:4], task.arm_ids[4:]
    base = nominal[0, proximal].clone()
    model = task.robot.get_spec().compile(); data = mujoco.MjData(model)
    addresses, _ = task.robot.joint_addresses(model); tcp = model.body('right_tcp_link').id
    cases = []
    for bound in (.2, .4, .8):
        shared = base.clone().requires_grad_()
        x = nominal[:, distal].clone().requires_grad_()
        op = torch.optim.Adam([dict(params=[x], lr=.025), dict(params=[shared], lr=.004)])
        best_score = float('inf'); best_q = None
        lo = torch.maximum(task.mdp.soft_lo[proximal], base-bound)
        hi = torch.minimum(task.mdp.soft_hi[proximal], base+bound)
        for iteration in range(1200):
            q = nominal.clone(); q[:, proximal] = shared; q[:, distal] = x
            h, d, _ = task.hand((*S[:4], q, S[5]))
            delta = ctx['aim_target_w'] - h
            angular = (d - torch.nn.functional.normalize(delta, dim=-1)).square().sum(-1) * delta.square().sum(-1)
            loss = angular.mean() + angular.max() + .001*(shared-base).square().sum()
            with torch.no_grad():
                den = (ctx['aim_normal_w']*d).sum(-1)
                distance = (ctx['aim_normal_w']*delta).sum(-1)/den.clamp_min(1e-6)
                error = (h+distance[:,None]*d-ctx['aim_target_w']).norm(dim=-1)
                error = torch.where((den>1e-4)&(distance>0), error, torch.inf)
                score = float(error.max())
                if score < best_score:
                    best_score, best_q = score, q.detach().clone()
            op.zero_grad(); loss.backward(); op.step()
            with torch.no_grad():
                shared.clamp_(lo, hi); x.clamp_(task.mdp.soft_lo[distal], task.mdp.soft_hi[distal])
        assert best_q is not None
        assert torch.equal(best_q[:,proximal],best_q[0,proximal].expand(75,4))
        body=[i for i in range(29) if i not in task.arm_ids.tolist()]
        assert torch.equal(best_q[:,body],nominal[:,body])
        predicted = task.metrics((*S[:4],best_q,S[5]),ctx)[0].cpu().numpy()
        errors=[];difference=0.
        for i in range(75):
            data.qpos[:3]=S[0][i].cpu().numpy();data.qpos[3:7]=actual_quat
            data.qpos[addresses]=best_q[i].cpu().numpy();mujoco.mj_forward(model,data)
            h=data.xpos[tcp];d=-data.xmat[tcp].reshape(3,3)[:,2]
            target=ctx['aim_target_w'][i].cpu().numpy();normal=ctx['aim_normal_w'][i].cpu().numpy()
            den=normal@d;distance=normal@(target-h)/den if den>1e-4 else -1
            err=float(np.linalg.norm(h+distance*d-target)) if distance>0 else float('inf')
            if np.isfinite(err) and np.isfinite(predicted[i]): difference=max(difference,abs(err-float(predicted[i])))
            else: assert np.isfinite(err)==np.isfinite(predicted[i])
            errors.append(err if np.isfinite(err) else None)
        assert difference<1e-4
        row=dict(proximal_max_change_bound_rad=bound,shared_proximal_q_rad=best_q[0,proximal].cpu().tolist(),
            actual_proximal_change_rad=(best_q[0,proximal]-base).cpu().tolist(),q_rad=best_q.cpu().tolist(),
            errors_m=errors,within5cm=sum(e is not None and e<=.05 for e in errors),
            torch_mujoco_max_abs_error_m=difference)
        cases.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('q_rad','errors_m')}),flush=True)
    out.write_text(json.dumps(dict(scope=__doc__,cases=cases,source_sha256=sha256(source),
        trajectory_sha256=sha256(trajectory),target_parameters=params.cpu().tolist(),
        physical_policy_acceptance=False,limitation='Kinematics only; excludes torque limits, collisions and closed-loop balance.'),indent=2,allow_nan=False))


if __name__=='__main__':main()
