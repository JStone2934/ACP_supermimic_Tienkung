"""Preserve explicit dataset proportions independently of clip/bin counts."""
import math
import torch


def group_probabilities(names, weights, device, base=None):
    groups = [name.split('/')[0] for name in names]
    if set(groups) != set(weights):
        raise ValueError('Sampling weights must cover exactly the loaded dataset groups')
    if any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError('Dataset sampling weights must be finite and positive')
    total = sum(weights.values())
    p = torch.ones(len(names), device=device) if base is None else base.clone()
    if p.ndim != 1 or len(p) != len(names) or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid within-group sampling probabilities')
    for group, weight in weights.items():
        ids = torch.tensor([g == group for g in groups], device=device)
        mass = p[ids].sum()
        if mass <= 0:
            raise ValueError('A dataset group has no sampling mass')
        p[ids] *= (weight / total) / mass
    return p
