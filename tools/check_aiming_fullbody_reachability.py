"""Offline seven-joint direction search; diagnostic only, never executed by the actor."""
import argparse
import torch
from supermimic.acp.aiming_fullbody import FullbodyTask
from supermimic.acp.aiming_fullbody_curriculum import fixed_parameters
from supermimic.acp.aiming_fullbody_io import json_save


def check(task, iterations=300):
    parameters = fixed_parameters(8, task.alpha.device)
    S = task.initial_state(len(parameters)); ctx = task.context(S, parameters)
    q = S[4][:, task.arm_ids].clone().requires_grad_()
    opt = torch.optim.Adam([q], lr=.03)
    best_error = torch.full((len(parameters),), float('inf'), device=q.device)
    best_q = q.detach().clone()
    for _ in range(iterations):
        full_q = S[4].scatter(1, task.arm_ids.expand_as(q), q)
        state = (*S[:4], full_q, S[5])
        h, d, _ = task.hand(state)
        delta = ctx['aim_target_w'] - h
        direction = torch.nn.functional.normalize(delta, dim=-1)
        loss = ((d - direction).square().sum(-1) * delta.square().sum(-1)).mean()
        with torch.no_grad():
            e = task.metrics(state, ctx)[0]
            improved = e < best_error
            best_q[improved] = q[improved]; best_error = torch.minimum(best_error, e)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): q.clamp_(task.mdp.soft_lo[task.arm_ids], task.mdp.soft_hi[task.arm_ids])
    return dict(kind='seven_joint_kinematic_direction_search', physical_acceptance=False,
                collision_checked=False, iterations=iterations, all_within_5cm=bool((best_error <= .05).all()),
                parameters=parameters.cpu().tolist(), error_m=best_error.cpu().tolist(),
                arm_names=[task.robot.joint_names[i] for i in task.arm_ids.tolist()],
                diagnostic_q=best_q.cpu().tolist(), robot_metadata=task.robot.metadata())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--iterations', type=int, default=300)
    args = p.parse_args(); torch.set_num_threads(4)
    report = check(FullbodyTask(args.device), args.iterations)
    json_save(args.out, report); print(dict(all_within_5cm=report['all_within_5cm'], worst_m=max(report['error_m'])))
