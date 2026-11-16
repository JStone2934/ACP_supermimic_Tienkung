"""Read-only arm geometry diagnostic for the exact final75 target grid."""
from pathlib import Path
import hashlib
import json
import torch
from supermimic.acp.aiming_targets import fixed_target_grid
from supermimic.acp.aiming_wide import WideAimingTask


def main():
    torch.set_num_threads(2)
    base = Path('/home/js/robot/locomotion/standing-aiming/results/lafan_jwp_aiming_20261115')
    source = base / 'observed_support_init.pt'
    ck = torch.load(source, map_location='cpu', weights_only=False)
    task = WideAimingTask('cpu', ck['config'])
    parameters = fixed_target_grid()
    state = task.initial_state(len(parameters))
    context = task.context(state, parameters)
    q = state[4][:, task.arm_ids].clone().requires_grad_()
    optimizer = torch.optim.Adam([q], lr=.03)
    best = torch.full((len(q),), float('inf'))
    best_q = q.detach().clone()
    for _ in range(600):
        full_q = state[4].scatter(1, task.arm_ids.expand_as(q), q)
        candidate = (*state[:4], full_q, state[5])
        hand, direction, _ = task.hand(candidate)
        delta = context['aim_target_w'] - hand
        loss = ((direction - torch.nn.functional.normalize(delta, dim=-1)).square().sum(-1)
                * delta.square().sum(-1)).mean()
        with torch.no_grad():
            error = task.metrics(candidate, context)[0]
            improved = error < best
            best_q[improved] = q[improved]
            best = torch.minimum(best, error)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            q.clamp_(task.mdp.soft_lo[task.arm_ids], task.mdp.soft_hi[task.arm_ids])
    report = dict(scope='Kinematic direction search from the exact tracker-derived pose; no policy, dynamics, collision, or stability acceptance.',
                  checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  axis=[0,0,-1], root_fixed=True, joints_optimized=[task.robot.joint_names[i] for i in task.arm_ids],
                  parameters=parameters.tolist(), target_positions_w=context['aim_target_w'].tolist(),
                  error_m=best.tolist(), diagnostic_arm_q=best_q.tolist(),
                  all_within_5cm=bool((best <= .05).all()), within_5cm_count=int((best <= .05).sum()),
                  scenarios=len(q), iterations=600, physical_acceptance=False,
                  failure_interpretation='A failed optimization does not prove geometric impossibility.')
    (base / 'wide_grid_geometry.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:report[k] for k in ('scenarios','within_5cm_count','all_within_5cm','physical_acceptance')}))


if __name__ == '__main__':
    main()
