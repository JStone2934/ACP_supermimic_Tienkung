"""Export the v2 actor with its observation, command and uniform action contract."""
import argparse
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import TASK_NAME, FullbodyTask, FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_io import json_save, save_checkpoint
from supermimic.acp.robots import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    args = p.parse_args(); torch.set_num_threads(4)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    task = FullbodyTask('cpu', ck['config']); validate_checkpoint(ck, task)
    pi = FullbodyPolicy(task.robot); pi.load_state_dict(ck['pi']); pi.eval()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    metadata = dict(task=TASK_NAME, task_config=task.metadata(), robot_metadata=task.robot.metadata(),
                    default_q=list(task.robot.default_q), action_scales=list(task.robot.action_scales),
                    soft_lo=task.mdp.soft_lo.tolist(), soft_hi=task.mdp.soft_hi.tolist(),
                    kp=list(task.robot.kp), kd=list(task.robot.kd), effort=list(task.robot.effort),
                    control_dt=task.dt, decimation=task.robot.decimation,
                    initial_q=task.initial_q.tolist(), source_sha256=sha256(args.ckpt),
                    source_stage=ck['stage'], qualified_for_deployment=False,
                    observation_order=['q-default_q-offset (29)', 'qd (29)', 'body angular velocity (3)',
                                       'gravity body (3)', 'previous executed action (29)', '6D(R.T @ heading(R)) (6)'],
                    history='10 frames oldest first; replicate reset observation',
                    normalization='source statistics for 99D observations and command[0:93]; remaining command fields explicitly scaled',
                    joint_offset='xi[1:30], subtract once when reconstructing executed action',
                    target='sample at reset in chest horizontal frame, then fixed in world',
                    acceptance='See independent physical evaluation; exporting is not certification')
    save_checkpoint(out / 'actor.pt', dict(pi=pi.state_dict(), **metadata))
    example = (torch.zeros(1, 10, 99), torch.zeros(1, 107))
    scripted = torch.jit.trace(pi, example)
    scripted.save(str(out / 'actor.ts'))
    loaded = torch.jit.load(str(out / 'actor.ts'))
    with torch.no_grad():
        history = torch.randn(4, 10, 99); command = torch.randn(4, 107)
        torch.testing.assert_close(loaded(history, command), pi(history, command))
    json_save(out / 'runtime.json', metadata)
    print(str(out))


if __name__ == '__main__': main()
