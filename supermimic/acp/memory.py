"""Bound analytic-reward activation memory without changing training batch sizes."""
import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_reward(function):
    def wrapped(*args):
        return checkpoint(function, *args, use_reentrant=False)
    return wrapped


def compile_reward(function, *, recompute=False):
    """Compile checkpoint and reward together to preserve saved tensor layouts.

    An eager checkpoint around an already compiled reward can recompute saved
    tensors with different padded strides in PyTorch 2.9's backward graph.
    """
    wrapped = checkpoint_reward(function) if recompute else function
    return torch.compile(wrapped, options={'max_fusion_size': 32})


def reward_batch(function, *args, chunk_size=0):
    """Split independent reward rows, preserving the complete reward graph.

    The outer imagination batch/horizon, losses and optimizer steps are unchanged.
    All non-scalar tensors in args are batched on axis 0 (state tuples included).
    Wrap the function with checkpoint_reward or compile_reward(recompute=True)
    to bound saved contact-geometry workspace.
    """
    if chunk_size == 0:
        return function(*args)
    if chunk_size < 0:
        raise ValueError('reward_chunk_size must be nonnegative')
    size = args[0][0].shape[0]  # flattened state tuple

    def sliced(value, start, end):
        if isinstance(value, torch.Tensor) and value.ndim:
            if value.shape[0] != size:
                raise ValueError('reward inputs must share the state batch dimension')
            return value[start:end]
        if isinstance(value, tuple):
            return tuple(sliced(item, start, end) for item in value)
        return value

    parts = [function(*(sliced(arg, start, start+chunk_size) for arg in args))
             for start in range(0, size, chunk_size)]
    if isinstance(parts[0], tuple):
        return tuple(torch.cat(items, 0) for items in zip(*parts))
    return torch.cat(parts, 0)
