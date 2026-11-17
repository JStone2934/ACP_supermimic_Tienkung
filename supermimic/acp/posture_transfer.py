"""Optional gradual posture transfer with a frozen same-input policy anchor."""
import math
import torch


def reference_reward(q, reference, scale, joint_ids=None):
    # Preserve historical experiments' 15-joint default. New contracts supply
    # explicit native joint IDs, including the left arm but excluding the right.
    selected = slice(0, 15) if joint_ids is None else joint_ids
    return 1 / (1 + ((q[..., selected] - reference[..., selected]) / scale).square().mean(-1))


def retention_cost(raw, teacher_raw, alpha, weight):
    # Teacher is a target, never an alternative differentiable controller.
    return weight * ((raw - teacher_raw.detach()) * alpha).square().mean(-1)


def attach_teacher(task, cfg):
    spec = cfg.get('policy_retention')
    if spec is None:
        return
    from .aiming_fullbody import FullbodyPolicy
    from .robots import sha256
    from .standing import freeze_normalizers
    if not math.isfinite(spec['weight']) or spec['weight'] <= 0:
        raise ValueError('Positive finite policy retention weight required')
    if sha256(spec['checkpoint']) != spec['sha256']:
        raise ValueError('Retention teacher identity changed')
    ck = torch.load(spec['checkpoint'], map_location=task.alpha.device, weights_only=False)
    task.robot.validate_metadata(ck['robot_metadata'])
    teacher = FullbodyPolicy(task.robot).to(task.alpha.device)
    teacher.load_state_dict(ck['pi'])
    teacher.eval().requires_grad_(False)
    freeze_normalizers(teacher)
    task.retention_teacher = teacher
