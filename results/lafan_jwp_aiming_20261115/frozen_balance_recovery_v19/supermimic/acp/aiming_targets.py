"""World-fixed spherical target placement for the wide-angle aiming curriculum."""
import torch


def spherical_targets(chest_w, initial_yaw, parameters):
    """Return target and plane normal for [range_m, azimuth_rad, elevation_rad, radius_m].

    The caller stores the returned world coordinates across simulation steps.
    Recomputing from the current torso would incorrectly move the target.
    """
    if parameters.shape != (len(chest_w), 4) or chest_w.shape != (len(parameters), 3):
        raise ValueError('Expected batched chest positions and four target parameters')
    if not torch.isfinite(parameters).all() or not torch.isfinite(chest_w).all() or not torch.isfinite(initial_yaw).all():
        raise ValueError('Nonfinite target inputs')
    distance, azimuth, elevation, radius = parameters.unbind(-1)
    if (distance <= 0).any() or (radius <= 0).any() or (elevation.abs() > torch.pi/2).any():
        raise ValueError('Invalid range, radius or elevation')
    yaw = initial_yaw.reshape(-1) + azimuth
    direction = torch.stack((elevation.cos()*yaw.cos(), elevation.cos()*yaw.sin(), elevation.sin()), -1)
    return chest_w + distance[:, None]*direction, direction


def fixed_target_grid(device='cpu'):
    import itertools
    import math
    return torch.tensor([[r, math.radians(a), math.radians(e), .05]
                         for r, a, e in itertools.product((2., 3., 4.), (-60., -30., 0., 30., 60.),
                                                         (-30., -15., 0., 15., 30.))], device=device)
