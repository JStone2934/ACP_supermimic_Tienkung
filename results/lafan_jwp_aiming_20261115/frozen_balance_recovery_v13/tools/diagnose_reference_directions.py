"""Check learned H8 action derivatives against physical finite differences.

Read-only checkpoint diagnostic on training-source states. No optimizer updates,
policy exports, or acceptance decisions are made here.
"""
import argparse
import json
from pathlib import Path
import torch
from supermimic.acp.env import TrackingSim, make_sim
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.motion import MotionLib
from supermimic.acp.mdp import MDP, RewardCfg, normalize_xi, flatten_state
from supermimic.acp.networks import Policy, WorldModel, Critic
from supermimic.acp.reference_guidance import cap_reference_value
from supermimic.acp.robots import get_robot, sha256
from supermimic.acp.task import ContactCfg, ContactTask
from supermimic.rotations import mat_to_quat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--epsilon', type=float, default=.005)
    p.add_argument('--full-tail', action='store_true', help='Continue the original actor to physical clip termination')
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    ck = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    if ck['args']['reward'] != 'supermimic':
        raise ValueError('This diagnostic implements the SuperMimic reward only')
    robot = get_robot('tienkung3_29dof')
    robot.validate_metadata(ck['robot_metadata'])
    kin = RobotKinematics(a.device, robot)
    motion = MotionLib.from_dir(str(a.dataset/'motions'), kin, a.device)
    mdp = MDP(kin, motion, torch.tensor(robot.default_q, device=a.device),
              RewardCfg(term_root_distance=ck['args'].get('tracking_root_radius')))
    ctc = ContactTask(ContactCfg(**(ck['ctc_cfg'] | dict(contacts_dir=str(a.dataset/'contacts')))),
                      mdp, str(a.dataset/'motions'), a.device)
    pi = Policy(robot.obs_dim, ctc.cmd_dim, hist=10, robot=robot,
                pose_action_skip=ck['args'].get('pose_action_skip', False)).to(a.device)
    wm = WorldModel(mdp.scan_dim, robot=robot).to(a.device)
    critic = Critic(robot.inv_dim+mdp.critic_extra+ctc.cmd_dim).to(a.device)
    for module, key in [(pi, 'pi'), (wm, 'wm'), (critic, 'critic')]:
        module.load_state_dict(ck[key]); module.eval(); module.requires_grad_(False)
    env = TrackingSim(a.count, motion, mdp, a.device, eval_mode=True,
                      sim_cfg=make_sim(robot=robot, physics_substeps=ck['args']['physics_substeps']))
    ids = torch.arange(a.count, device=a.device)
    clip_ids = torch.linspace(0, len(motion.names)-1, a.count, device=a.device).long()
    frames = motion.clip_start[clip_ids]+((motion.clip_len[clip_ids]-40)*.25).long()
    env.reset(ids, frames)
    gs = torch.full_like(frames, 50)
    with torch.no_grad():
        hist = env.observe()[:, None].repeat(1, 10, 1)
        for _ in range(16):
            action = pi(hist, ctc.command(env.state(), env.f, env.v, gs, noise=False))
            env.step(action)
            hist = torch.cat((hist[:, 1:], env.observe()[:, None]), 1)
        state = tuple(x.clone() for x in env.state())
        f0, v0, ap0, xi = env.f.clone(), env.v.clone(), env.a_prev.clone(), env.xi.clone()
        eligible = ~env.done_eval.clone()
    gamma = ck['args']['gamma']
    boxes = motion.boxes(f0, v0)

    def reward(S, f, action, previous):
        return mdp.reward(S, f, action, previous)+ctc.cfg.w_label*.02*ctc.label_reward(S, f, v0, boxes)

    def tail(S, f):
        value = critic(ctc.critic_input(S, f, v0, gs, normalize_xi(xi)))
        if ck['args'].get('bound_reference_bootstrap'):
            value = cap_reference_value(value, f, motion.clip_end, motion.clip_id,
                                        .1+ctc.cfg.w_label*.02, gamma)
        return value

    @torch.no_grad()
    def restore():
        env.sim.reset(ids); env.scene.reset(ids)
        pos, rot, vel, ang, q, qd = state
        env.robot.write_root_link_pose_to_sim(torch.cat((pos+env.scene.env_origins, mat_to_quat(rot)), -1), env_ids=ids)
        env.robot.write_root_link_velocity_to_sim(torch.cat((vel, ang), -1), env_ids=ids)
        env.robot.write_joint_position_to_sim(q[:, env.to_internal], env_ids=ids)
        env.robot.write_joint_velocity_to_sim(qd[:, env.to_internal], env_ids=ids)
        env.robot.set_joint_position_target(q[:, env.to_internal])
        env.f = f0.clone(); env.v = v0.clone(); env.a_prev = ap0.clone(); env.ep_step.zero_()
        env.done_eval.zero_(); env.failed_eval.zero_(); env.refresh()
        error = float((flatten_state(env.state())-flatten_state(state)).abs().max())
        if error > 1e-4:
            raise ValueError(f'Physical restore mismatch: {error}')
        return error

    @torch.no_grad()
    def physical(actions=None, include_tail=False):
        restore_error = restore()
        history, ap = hist.clone(), ap0.clone()
        total = torch.zeros(a.count, device=a.device)
        alive = torch.ones_like(total)
        sequence = []
        for t in range(8):
            action = actions[t] if actions is not None else pi(history, ctc.command(env.state(), env.f, env.v, gs, noise=False))
            tr = env.step(action)
            total += gamma**t*alive*reward(env.state(), f0+t+1, action, ap)
            alive *= (~tr['term']).float()
            history = torch.cat((history[:, 1:], env.observe()[:, None]), 1)
            sequence.append(action.clone()); ap = action
        result = dict(reward=total, tail=gamma**8*alive*tail(env.state(), f0+8),
                      actions=torch.stack(sequence), restore_error=restore_error)
        if include_tail:
            measured_tail = torch.zeros_like(total)
            alive *= eligible.float()
            t = 8
            while bool(alive.any()):
                action = pi(history, ctc.command(env.state(), env.f, env.v, gs, noise=False))
                tr = env.step(action)
                f1 = torch.minimum(tr['f']+1, motion.clip_end[motion.clip_id[tr['f']]])
                measured_tail += gamma**t*alive*reward(env.state(), f1, action, ap)
                alive *= (~tr['done']).float()
                history = torch.cat((history[:, 1:], env.observe()[:, None]), 1)
                ap = action
                env.f = torch.minimum(env.f, motion.clip_end[motion.clip_id[env.f]]-1)
                t += 1
            result['measured_tail'] = measured_tail
            result['total_return'] = total+measured_tail
        return result

    def imagined(actions):
        S, previous = state, ap0
        total = torch.zeros(a.count, device=a.device)
        alive = torch.ones_like(total)
        for t in range(8):
            action = actions[t]
            S = wm.step(S, action, normalize_xi(xi), mdp.scan(S, boxes))
            total = total+gamma**t*alive*reward(S, f0+t+1, action, previous)
            alive = alive*(1-mdp.soft_termination(S, f0+t+1))
            previous = action
        return dict(reward=total, tail=gamma**8*alive*tail(S, f0+8))

    baseline = physical()
    actions = baseline['actions'].detach().requires_grad_()
    predicted = imagined(actions)
    gradients = {}
    directions = {}
    for kind in ('reward', 'tail'):
        g = torch.autograd.grad(predicted[kind][eligible].mean(), actions, retain_graph=True)[0]
        gradients[kind] = g.detach()
        gq = g/env.alpha
        directions[kind] = (gq/gq.abs().amax((0, 2), keepdim=True).clamp_min(1e-12)/env.alpha).detach()
    gradients['total'] = gradients['reward']+gradients['tail']
    gq = gradients['total']/env.alpha
    directions['total'] = (gq/gq.abs().amax((0, 2), keepdim=True).clamp_min(1e-12)/env.alpha).detach()
    actions = actions.detach()
    report = dict(scope=__doc__, checkpoint_sha256=sha256(a.checkpoint), data_hash=motion.data_hash,
                  eligible_states=int(eligible.sum()), states=a.count, epsilon_joint_rad=a.epsilon,
                  physical_tail_complete=a.full_tail,
                  gradient_norms={k:float(v.norm()) for k,v in gradients.items()},
                  reward_tail_cosine=float((gradients['reward']*gradients['tail']).sum()/
                    (gradients['reward'].norm()*gradients['tail'].norm()).clamp_min(1e-12)), cases={})
    cases = [('baseline', torch.zeros_like(actions))]
    cases += [(kind+'_'+sign, sign_value*a.epsilon*d) for kind,d in directions.items()
              for sign,sign_value in [('plus', 1), ('minus', -1)]]
    cases += [('baseline_repeat', torch.zeros_like(actions))]
    evaluated = {}
    original_eligible = eligible.clone()
    original_count = a.count
    if a.full_tail:
        # Every perturbation is an independent simulation world. Batch all cases
        # so the slowest complete clip is replayed only once in wall-clock time.
        with torch.no_grad():
            predictions = {name:imagined(actions+adjustment) for name,adjustment in cases}
        combined = torch.cat([actions+adjustment for _,adjustment in cases], 1)
        repeat = lambda x:x.repeat((len(cases),)+(1,)*(x.ndim-1))
        state = tuple(repeat(x) for x in state)
        hist, f0, v0, ap0, xi, gs, boxes, eligible = map(repeat, (hist,f0,v0,ap0,xi,gs,boxes,eligible))
        a.count *= len(cases)
        ids = torch.arange(a.count, device=a.device)
        env = TrackingSim(a.count, motion, mdp, a.device, eval_mode=True,
                          sim_cfg=make_sim(robot=robot, physics_substeps=ck['args']['physics_substeps']))
        env.reset(ids, f0, v0)
        print(f'Physical full tails: {len(cases)} cases x {original_count} states = {a.count} GPU worlds', flush=True)
        real_all = physical(combined, include_tail=True)
        for index,(name,_) in enumerate(cases):
            sl = slice(index*original_count,(index+1)*original_count)
            real = {key:value[sl] for key,value in real_all.items()
                    if key not in ('actions','restore_error')}
            real['restore_error'] = real_all['restore_error']
            evaluated[name] = (predictions[name],real)
        eligible = original_eligible
        report['physical_cases_batched'] = True
        report['physical_worlds'] = a.count
    else:
        for name, adjustment in cases:
            with torch.no_grad():
                evaluated[name] = (imagined(actions+adjustment), physical(actions+adjustment))
    for name, _ in cases:
        pred,real = evaluated[name]
        row = {f'{prefix}_{kind}':value[kind][eligible].cpu().tolist()
               for prefix,value in [('predicted',pred),('physical',real)] for kind in ('reward','tail')}
        row['restore_error'] = real['restore_error']
        if a.full_tail:
            for key in ('measured_tail', 'total_return'):
                row['physical_'+key] = real[key][eligible].cpu().tolist()
        report['cases'][name] = row
        args_mean = {k:sum(v)/len(v) for k,v in row.items() if isinstance(v,list)}
        print(name, args_mean, flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
    comparisons = {}
    for kind in directions:
        plus, minus = report['cases'][kind+'_plus'], report['cases'][kind+'_minus']
        comparisons[kind] = {}
        keys = ['predicted_reward','predicted_tail','physical_reward','physical_tail']
        if a.full_tail:
            keys += ['physical_measured_tail', 'physical_total_return']
        for key in keys:
            delta = torch.tensor(plus[key])-torch.tensor(minus[key])
            comparisons[kind][key] = dict(mean=float(delta.mean()), sem=float(delta.std()/len(delta)**.5),
                                          positive_fraction=float((delta>0).float().mean()))
    report['plus_minus'] = comparisons
    report['baseline_repeat_max_abs'] = max(abs(x-y) for key in ('physical_reward','physical_tail')
        for x,y in zip(report['cases']['baseline'][key], report['cases']['baseline_repeat'][key]))
    a.out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
