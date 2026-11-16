"""Checkpoint and dataset contracts shared by Stage 2 and offline planner training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def inherit_execution_sizes(args, checkpoint):
    """Resolve unspecified GLOBAL sizes before Runtime partitions them per rank.

    Stage 2's episode buffer and schedules deliberately keep their own defaults.
    Explicit CLI overrides always win, including reward_chunk_size=0.
    """
    defaults = dict(num_envs=4096, wm_batch=2048, pi_batch=4096, reward_chunk_size=0)
    previous = checkpoint.get('args', {})
    for name, default in defaults.items():
        if hasattr(args, name) and getattr(args, name) is None:
            value = previous.get(name)
            setattr(args, name, default if value is None else value)


def validation_split(names, motion_dir, *, saved=None, continuing=False):
    """Hold out every tenth source motion, keeping all materialized variants together.

    Ordinary datasets retain their existing per-clip split. Materialized Stage 1
    datasets use the authoritative source_key mapping rather than filename guesses.
    A resumed grouped run must carry the same split identity.
    """
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError('duplicate motion names in planner split')
    groups = {name: name for name in names}
    manifest_path = Path(motion_dir).parent/'manifest.json'
    grouping = 'clip'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('kind') == 'stage1-original-pairs-v1':
            entries = manifest['entries']
            groups = {e['key']: e['source_key'] for e in entries}
            if len(groups) != len(entries) or set(groups) != set(names):
                raise ValueError('planner split manifest does not match loaded motions')
            if any(name.split('/')[0] != groups[name].split('/')[0] for name in names):
                raise ValueError('planner source group crosses motion categories')
            grouping = 'manifest_source_key'
    selected = set()
    for folder in sorted({name.split('/')[0] for name in names}):
        ordered = list(dict.fromkeys(groups[name] for name in names if name.split('/')[0] == folder))
        selected.update(ordered[::10])
    mask = [groups[name] in selected for name in names]
    identity = sorted((name, groups[name], val) for name, val in zip(names, mask))
    record = dict(version=1, grouping=grouping,
                  sha256=hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest(),
                  clips=len(names), sources=len(set(groups.values())), validation_clips=sum(mask),
                  validation_sources=sorted(selected))
    if saved is not None and saved != record:
        raise ValueError('planner validation split changed since checkpoint')
    if continuing and saved is None and grouping == 'manifest_source_key':
        raise ValueError('legacy expanded-data checkpoint has no source-group validation split; start a new planner/Stage 2 run')
    return mask, record


def validate_planner_init(planner_checkpoint, tracker_checkpoint, tracker_hash, *, robot_id):
    """Reject a planner trained on a different tracker, dataset or reward geometry."""
    strict = robot_id != 'g1'
    for key, expected in (('data_hash', tracker_checkpoint.get('data_hash')), ('tracker_hash', tracker_hash)):
        actual = planner_checkpoint.get(key)
        if (strict or actual is not None) and (actual is None or actual != expected):
            raise ValueError(f'planner initialization {key} mismatch')
    saved_contact = planner_checkpoint.get('ctc_cfg')
    if saved_contact is not None:
        from dataclasses import asdict
        from .task import ContactCfg
        if asdict(ContactCfg(**saved_contact)) != asdict(ContactCfg(**tracker_checkpoint['ctc_cfg'])):
            raise ValueError('planner initialization contact configuration mismatch')


def initialize_critic(critic, source_state):
    """Widen the Stage 1 critic with zero-weight planner inputs, preserving its value."""
    target = critic.state_dict()
    n_in = source_state['net.0.weight'].shape[1]
    if target['net.0.weight'].shape[1] < n_in:
        raise ValueError('Stage 2 critic input cannot be narrower than Stage 1')
    for key, value in source_state.items():
        if key == 'net.0.weight':
            target[key].zero_()[:, :n_in] = value
        elif key in ('norm.mean', 'norm.var'):
            target[key][:n_in] = value
        else:
            target[key] = value
    critic.load_state_dict(target)
