"""Static target stages. Promotion requires two complete physical evaluations."""
import itertools
import math
import torch

STAGES = [
    dict(name='A', depth=[3., 3.], alpha=0., beta=0., rho=.20, aim_weight=0.),
    dict(name='B1', depth=[3., 3.], alpha=0., beta=0., rho=.20, aim_weight=2.),
    dict(name='B2', depth=[3., 3.], alpha=5., beta=3., rho=.20, aim_weight=2.),
    dict(name='C1a', depth=[3., 3.], alpha=10., beta=6., rho=.20, aim_weight=2.),
    dict(name='C1b', depth=[3., 3.], alpha=15., beta=10., rho=.20, aim_weight=2.),
    dict(name='C2a', depth=[2.5, 3.5], alpha=15., beta=10., rho=.20, aim_weight=2.),
    dict(name='C2b', depth=[2., 4.], alpha=15., beta=10., rho=.20, aim_weight=2.),
    dict(name='C3a', depth=[2., 4.], alpha=15., beta=10., rho=.10, aim_weight=2.),
    dict(name='C3b', depth=[2., 4.], alpha=15., beta=10., rho=.05, aim_weight=2.),
]


def fixed_parameters(level, device):
    s = STAGES[level]
    lo, hi = s['depth']
    depth = sorted(set([lo, (lo + hi) / 2, hi]))
    angles = lambda a: sorted(set([-math.radians(a), 0., math.radians(a)]))
    return torch.tensor([[d, a, b, s['rho']] for d, a, b in itertools.product(depth, angles(s['alpha']), angles(s['beta']))], device=device)


class Curriculum:
    def __init__(self, device, state=None):
        self.device = device
        self.level, self.consecutive, self.mastered = 0, 0, []
        if state:
            self.level, self.consecutive, self.mastered = state['level'], state['consecutive'], list(state['mastered'])

    def state_dict(self):
        return dict(level=self.level, consecutive=self.consecutive, mastered=self.mastered)

    def sample(self, n, aim_weight=2.):
        levels = torch.full((n,), self.level, device=self.device, dtype=torch.long)
        if self.level:
            levels[torch.rand(n, device=self.device) < .2] -= 1
        parameters = torch.empty(n, 4, device=self.device)
        weights = torch.empty(n, device=self.device)
        for i in levels.unique().tolist():
            s = STAGES[i]; ids = levels == i
            u = torch.rand(int(ids.sum()), 4, device=self.device)
            parameters[ids] = torch.stack((s['depth'][0] + u[:, 0] * (s['depth'][1] - s['depth'][0]),
                                          (2 * u[:, 1] - 1) * math.radians(s['alpha']),
                                          (2 * u[:, 2] - 1) * math.radians(s['beta']),
                                          torch.full_like(u[:, 3], s['rho'])), -1)
            # A samples remain standing-only even after promotion, explicitly recorded in replay.
            weights[ids] = 0. if i == 0 else aim_weight
        return parameters, weights

    def settle(self, report):
        if report.get('level') != self.level or report.get('stage') != STAGES[self.level]['name']:
            raise ValueError('Evaluation stage differs from curriculum')
        good = bool(report['passed']) and all(x['standing_pass'] for x in report['scenarios'])
        self.consecutive = self.consecutive + 1 if good else 0
        if self.consecutive >= 2:
            if self.level not in self.mastered:
                self.mastered.append(self.level)
            if self.level < len(STAGES) - 1:
                self.level += 1
            self.consecutive = 0
