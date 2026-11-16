"""Simultaneous collision guidance with duplicate historical planes collapsed."""
import torch

def pair_collision_cost(gaps, groups, group_count, frames):
    """Sum frame/body-pair peak squared gaps, normalized by trajectory length.

    Groups are fixed outside the CUDA graph. Acceptance remains native geometry.
    """
    violations = torch.relu(.0002-gaps).square()
    peaks = torch.zeros(group_count, device=gaps.device, dtype=gaps.dtype).scatter_reduce(
        0, groups, violations, reduce='amax')
    return peaks.sum()/frames
