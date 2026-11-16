"""Collect exact policy inputs/outputs from an accepted balance-stage teacher."""
import argparse
import json
from pathlib import Path

import torch

from supermimic.acp.aiming_fullbody import FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_fullbody_io import save_checkpoint, json_save
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256
from supermimic.acp.train_wide_aiming import check_promotion
from supermimic.acp.wide_curriculum import sample


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--balance-reports', type=Path, nargs=2, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=2057)
    p.add_argument('--envs', type=int, default=32)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--stride', type=int, default=2)
    a = p.parse_args()
    if a.out.exists() or a.out.with_suffix('.json').exists():
        raise FileExistsError(a.out)
    assert a.envs >= 8 and a.steps >= 1500 and a.stride > 0
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    digest = check_promotion(a.checkpoint, [json.loads(x.read_text()) for x in a.balance_reports], 0)
    ck = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    assert not ck.get('smoke') and ck['config']['curriculum_level'] == 0
    task = WideAimingTask(a.device, ck['config'])
    validate_checkpoint(ck, task)
    policy = FullbodyPolicy(task.robot).to(a.device)
    policy.load_state_dict(ck['pi'])
    policy.eval().requires_grad_(False)
    env = FullbodySim(task, a.envs, lambda n: sample(0, n, a.device))
    history = env.observe()[:, None].repeat(1, policy.hist, 1)
    rows = []
    terminations = torch.zeros(a.envs, dtype=torch.long, device=a.device)
    for step in range(a.steps):
        observation = env.observe()
        fresh = env.ep_step == 0
        history = torch.where(fresh[:, None, None], observation[:, None].expand_as(history), history)
        command = task.command(env.state(), env.context)
        raw = policy(history, command)
        if step % a.stride == 0:
            rows.append(dict(history=history.cpu().clone(), command=command.cpu().clone(),
                             raw=raw.cpu().clone(), ep_step=env.ep_step.cpu().clone()))
        transition = env.step(raw)
        terminations += transition['terminated'].long()
        history = torch.cat((history[:, 1:], env.observe()[:, None]), 1)
    data = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
    assert all(value.isfinite().all() for value in data.values())
    # Recompute sampled teacher outputs, catching stale histories or shifted actions.
    maximum = 0.
    for t in (0, len(rows)//3, len(rows)//2, len(rows)-1):
        predicted = policy(data['history'][t].to(a.device), data['command'][t].to(a.device))
        maximum = max(maximum, float((predicted.cpu()-data['raw'][t]).abs().max()))
    assert maximum < 1e-6
    assert sha256(a.checkpoint) == digest
    validation_worlds = list(range(3, a.envs, 4))
    training_worlds = [i for i in range(a.envs) if i not in validation_worlds]
    metadata = dict(scope=__doc__, teacher=str(a.checkpoint.resolve()), teacher_sha256=digest,
                    robot_metadata=task.robot.metadata(), seed=a.seed, envs=a.envs, steps=a.steps,
                    stride=a.stride, samples=len(rows)*a.envs, training_worlds=training_worlds,
                    validation_worlds=validation_worlds, no_action_noise=True,
                    terminations_by_world=terminations.cpu().tolist(),
                    teacher_output_recompute_max_abs_error=maximum,
                    collection_can_reset=True, physical_policy_acceptance=False,
                    runtime_data_dependency=False)
    save_checkpoint(a.out, dict(data=data, metadata=metadata))
    json_save(a.out.with_suffix('.json'), dict(metadata, dataset_sha256=sha256(a.out)))
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
