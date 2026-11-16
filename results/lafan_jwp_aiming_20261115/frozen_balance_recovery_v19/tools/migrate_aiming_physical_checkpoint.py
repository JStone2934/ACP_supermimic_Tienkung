"""Explicitly migrate a physical-training checkpoint to a reviewed configuration.

Writes a new checkpoint, preserves actor/WM weights, and invalidates reward-specific
critic state and comparisons when the reward changes. Ordinary resume stays strict.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path

import torch

from supermimic.acp.aiming_fullbody import FullbodyTask, validate_checkpoint
from supermimic.acp.aiming_fullbody_io import save_checkpoint, json_save
from supermimic.acp.aiming_physical import BACKEND, policy_digest
from supermimic.acp.networks import Critic
from supermimic.acp.train_aiming_physical import validate_config


ALLOWED_CHANGES = frozenset(('horizon', 'critic_envs', 'critic_train_envs',
                             'validation_horizons', 'reward_weights', 'feet_position_scale_m'))


def reconfigure_checkpoint(original, requested):
    if original.get('training_backend') != BACKEND or original.get('stage') != 'training':
        raise ValueError('Expected a complete physical-MC training checkpoint')
    if original.get('smoke') or original['physical_state'].get('completed'):
        raise ValueError('Cannot migrate smoke or completed training')
    old_cfg = original['config']
    changes = {k: {'old': old_cfg.get(k), 'new': v} for k, v in requested.items() if old_cfg.get(k) != v}
    unsupported = set(changes) - ALLOWED_CHANGES
    if unsupported:
        raise ValueError(f'Unsupported configuration changes: {sorted(unsupported)}')
    cfg = copy.deepcopy(old_cfg)
    cfg.update(copy.deepcopy(requested))
    validate_config(cfg)
    old_task = FullbodyTask('cpu', old_cfg)
    validate_checkpoint(original, old_task)
    task = FullbodyTask('cpu', cfg)
    reward_changed = old_task.metadata() != task.metadata()
    cfg['max_step_reward'] = task.max_step_reward
    checkpoint = copy.deepcopy(original)
    checkpoint.update(config=cfg, task_config=task.metadata())
    state = checkpoint['physical_state']
    state.update(anchor=None, best_key=None, last_decision=None, last_calibration=None,
                 wm_validation=None, initial_evaluated=False, official_pending=True,
                 consecutive_rejections=0, calibration_failures=0, completed=False)
    checkpoint['curriculum']['consecutive'] = 0
    if checkpoint['curriculum']['level'] == 0:
        state['standing_protected'] = False
    checkpoint['exploration'].update(best_score=None, stale_evaluations=0)
    if reward_changed:
        # Keep the current actor learning rate and WM Adam state. Old actor moments
        # and critic values describe a different objective and must not drive it.
        checkpoint['optimizers'][0]['state'] = {}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['seed'] + int(checkpoint['iteration']) + 16000)
            critic = Critic(task.critic_dim)
        optimizer = torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr'],
                                     fused=cfg.get('fused_adam', False))
        checkpoint['critic'] = critic.state_dict()
        checkpoint['optimizers'][2] = optimizer.state_dict()
        state['critic_ready'] = False
    record = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  from_iteration=checkpoint['iteration'], changes=changes,
                  reward_changed=reward_changed, critic_reinitialized=reward_changed,
                  actor_moments_reset=reward_changed,
                  actor_lr_preserved=checkpoint['optimizers'][0]['param_groups'][0]['lr'],
                  actor_weights_preserved=True, world_model_and_optimizer_preserved=True,
                  cumulative_counters_preserved=True, full_evaluation_required=True)
    checkpoint.setdefault('configuration_history', []).append(record)
    validate_checkpoint(checkpoint, task)
    assert policy_digest(original['pi']) == policy_digest(checkpoint['pi'])
    assert policy_digest(original['wm']) == policy_digest(checkpoint['wm'])
    return checkpoint, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.source.resolve() == args.out.resolve():
        raise FileExistsError('Migration requires a new checkpoint path')
    torch.set_num_threads(4)
    original = torch.load(args.source, map_location='cpu', weights_only=False)
    checkpoint, record = reconfigure_checkpoint(original, json.loads(args.config.read_text()))
    record.update(source=str(args.source.resolve()),
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  destination=str(args.out.resolve()),
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    save_checkpoint(args.out, checkpoint)
    json_save(args.out.with_suffix('.migration.json'), record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
