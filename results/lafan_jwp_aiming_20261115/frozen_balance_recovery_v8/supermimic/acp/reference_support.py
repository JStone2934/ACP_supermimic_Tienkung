"""Optional world-frame root and labeled support-foot tracking rewards."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceSupportCfg:
    root_weight: float = 0.
    position_weight: float = 0.
    velocity_weight: float = 0.
    root_scale: float = .1
    position_scale: float = .05
    velocity_scale: float = .1


class ReferenceSupportReward:
    def __init__(self, mdp, contacts, cfg):
        self.mdp, self.contacts, self.cfg = mdp, contacts, cfg
        names = ('left_foot', 'right_foot')
        self.contact_columns = [list(mdp.robot.contact_bodies).index(name) for name in names]
        self.body_columns = [mdp.kin.tracked_names.index(mdp.robot.contact_bodies[name][0]) for name in names]
        self.max_step_reward = mdp.robot.control_dt*(cfg.root_weight+cfg.position_weight+cfg.velocity_weight)

    def terms(self, state, frames, variants):
        c, m = self.cfg, self.mdp.motion
        positions, _, velocities, _ = self.mdp.kin.forward(*state)
        mask = self.contacts.labels[frames, variants][:, self.contact_columns].to(state[0].dtype)
        count = mask.sum(-1).clamp_min(1)
        kernel = lambda error,scale:1/(1+error.square().sum(-1)/scale**2)
        root = kernel(state[0]-m.root_p[frames], c.root_scale)
        position = kernel(positions[:, self.body_columns]-m.body_p[frames][:, self.body_columns], c.position_scale)
        velocity = kernel(velocities[:, self.body_columns]-m.body_v[frames][:, self.body_columns], c.velocity_scale)
        return dict(root=root*c.root_weight,
                    support_position=(position*mask).sum(-1)/count*c.position_weight,
                    support_velocity=(velocity*mask).sum(-1)/count*c.velocity_weight)

    def __call__(self, state, frames, variants):
        return self.mdp.robot.control_dt*sum(self.terms(state, frames, variants).values())
