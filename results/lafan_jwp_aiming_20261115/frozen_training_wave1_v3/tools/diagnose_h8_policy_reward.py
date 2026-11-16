"""Finite diagnostics on an immutable checkpoint; never alters the active trainer."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from supermimic.acp.aiming_fullbody import FullbodyTask, FullbodyPolicy, flat_boxes
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_physical import (execution_setup, rollout, take, to_device,
    actor_step, cpu_tree, policy_digest)
from supermimic.acp.mdp import unflatten_state, normalize_xi
from supermimic.acp.networks import WorldModel, Critic
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.train_aiming_fullbody import bound_bootstrap
from tools.diagnose_aiming_fullbody_update import gradient_report


def objective(pi, wm, critic, task, cfg, b):
    S, hist, ap, t = unflatten_state(b['s']), b['history'], b['ap'], b['t']
    alive = torch.ones_like(t, dtype=torch.float32)
    sums = {k: torch.zeros_like(alive) for k in task.weights}
    for j in range(cfg['horizon']):
        raw = pi(hist, task.command(S, b['ctx']))
        a, _ = task.compose_action(raw, b['xi'])
        S = wm.step(S, a, normalize_xi(b['xi']), task.mdp.scan(S, flat_boxes(S)))
        _, terms = task.reward_terms(S, a, ap, b['ctx'], raw=raw, xi=b['xi'])
        for k, value in terms.items():
            sums[k] = sums[k] + cfg['gamma']**j * alive * value
        alive = alive * (1 - task.soft_termination(S, t + 1, b['ctx']))
        hist = torch.cat((hist[:, 1:], task.mdp.obs(S, a, b['xi'])[:, None]), 1)
        ap, t = a, t + 1
    value = bound_bootstrap(critic(task.critic_features(S, hist, ap, b['xi'], t, b['ctx'])), t, task, cfg)
    sums['bootstrap'] = cfg['gamma']**cfg['horizon'] * alive * value
    return sums


class Trace:
    def __init__(self, env):
        self.env = env
        self.original = env.step
        env.step = self.step
        self.clear()

    def clear(self):
        self.alive = torch.ones(self.env.N, dtype=torch.bool, device=self.env.device)
        self.terms = []
        self.raws = []
        self.states = []
        self.alives = []

    def step(self, raw):
        tr = self.original(raw)
        self.terms.append({k: torch.where(self.alive, v, 0.) for k, v in tr['reward_contributions'].items()})
        self.raws.append(raw.detach().clone())
        self.states.append(tr['s1'].clone())
        self.alives.append(self.alive.clone())
        self.alive &= ~tr['done']
        return tr

    def summary(self, gamma):
        values = {k: torch.stack([r[k] for r in self.terms]) for k in self.terms[0]}
        discount = gamma ** torch.arange(len(self.terms), device=self.env.device)
        return dict(rewards={k: float(v.sum(0).mean()) for k, v in values.items()},
                    discounted_rewards={k: float((v * discount[:, None]).sum(0).mean()) for k, v in values.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.cuda.set_device(args.device); torch.manual_seed(20261114)
    import warp as wp
    wp.config.quiet = True
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = copy.deepcopy(ck['config']); cfg.update(compile_training=False, actor_recompute=False, amp='none', allow_tf32=False)
    execution_setup(cfg)
    task = FullbodyTask(args.device, cfg)
    pi = FullbodyPolicy(task.robot).to(args.device); pi.load_state_dict(ck['pi'])
    wm = WorldModel(task.mdp.scan_dim, robot=task.robot).to(args.device); wm.load_state_dict(ck['wm'])
    critic = Critic(task.critic_dim).to(args.device); critic.load_state_dict(ck['critic'])
    freeze_normalizers(pi, wm, critic); wm.requires_grad_(False); critic.requires_grad_(False)
    seed = cfg['seed'] + 50000
    env = FullbodySim(task, 64, lambda n: (task.alpha.new_tensor([[3., 0., 0., .2]]).repeat(n, 1), 0.), eval_mode=True)
    trace = Trace(env)
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_iteration=ck['iteration'],
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  original_policy_digest=policy_digest(ck['pi']), saved_actor_lr=ck['optimizers'][0]['param_groups'][0]['lr'],
                  limitations=['Finite diagnostic on copies; production service untouched.',
                      'FP32 eager execution; 64 collection environments instead of production 1024.',
                      'Checkpoint critic retained without additional fitting; sampled starts from this policy.',
                      'Counterfactual optimizer changes are diagnostics, not approved training changes.'], cases={}, started=time.time())

    def save():
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')

    def evaluate(label, model):
        trace.clear()
        _, result = rollout(model, task, env, cfg, seed=seed)
        result.update(trace.summary(cfg['gamma']))
        report['cases'][label] = result
        save()
        print(label, json.dumps({k:result[k] for k in ['return_','undiscounted_return','survival_s','feet_max_m']}), flush=True)
        return result

    for i in range(3):
        evaluate(f'baseline_repeat_{i}', pi)
    trace.clear()
    data, collected = rollout(pi, task, env, cfg, seed=cfg['seed']+10000+ck['physical_state']['collection_round'], keep_data=True, prefixes=True)
    data = to_device(data, args.device)
    training = take(data, (data['env_id'] < 48).nonzero().flatten())
    report['collection'] = {k:v for k,v in collected.items() if k!='per_environment'}
    report['collection'].update(valid_rows=len(data['s']), training_rows=len(training['s']),
        start_step_min=int(training['t'].min()), start_step_max=int(training['t'].max()))
    torch.manual_seed(714)
    diagnostic = take(training, torch.randint(len(training['s']), (128,), device=args.device))
    b = dict(diagnostic, S=unflatten_state(diagnostic['s']))
    report['gradients'] = gradient_report(pi, wm, critic, task, cfg, b, ck['optimizers'][0])
    with torch.no_grad():
        report['initial_objective'] = {k:float(v.mean()) for k,v in objective(pi,wm,critic,task,cfg,diagnostic).items()}
    save(); print('gradient_decomposition_done', flush=True)
    torch.manual_seed(715)
    batches = [take(training, torch.randint(len(training['s']), (cfg['pi_batch'],), device=args.device)) for _ in range(5)]
    candidates = {}
    for mode in ['full_saved', 'full_fresh', 'immediate_fresh']:
        model = FullbodyPolicy(task.robot).to(args.device); model.load_state_dict(ck['pi'])
        opt = torch.optim.Adam(model.parameters(), lr=report['saved_actor_lr'], betas=tuple(cfg['actor_betas']), fused=cfg.get('fused_adam',False))
        if mode == 'full_saved': opt.load_state_dict(copy.deepcopy(ck['optimizers'][0]))
        logs = []
        for batch in batches:
            if mode != 'immediate_fresh':
                logs.append(actor_step(model, wm, critic, task, batch, opt, cfg, prebatched=True))
                wm.requires_grad_(False); critic.requires_grad_(False)
            else:
                opt.zero_grad(set_to_none=True)
                terms = objective(model, wm, critic, task, cfg, batch)
                loss = -sum(v for k,v in terms.items() if k!='bootstrap').mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                torch.nn.utils.clip_grad_value_(model.parameters(),cfg['actor_grad_clip_value']); opt.step()
                logs.append(dict(loss=float(loss.detach()),gradient_norm=float(norm)))
        with torch.no_grad():
            before = pi(diagnostic['history'],task.command(unflatten_state(diagnostic['s']),diagnostic['ctx']))
            after = model(diagnostic['history'],task.command(unflatten_state(diagnostic['s']),diagnostic['ctx']))
            action_delta = (after-before)*task.alpha
            stats = dict(steps=logs, joint_request_delta_max_rad=float(action_delta.abs().max()),
                         joint_request_delta_rms_rad=float(action_delta.square().mean().sqrt()),
                         objective={k:float(v.mean()) for k,v in objective(model,wm,critic,task,cfg,diagnostic).items()})
        candidates[mode] = cpu_tree(model.state_dict())
        report.setdefault('updates',{})[mode] = stats
        evaluate(mode,model);evaluate(mode+'_repeat',model)
        del model,opt
    mirror = FullbodyPolicy(task.robot).to(args.device); mirror.load_state_dict(ck['pi'])
    with torch.no_grad():
        for name,param in mirror.named_parameters():
            param.add_((ck['pi'][name]-candidates['full_saved'][name]).to(args.device))
    evaluate('saved_update_reverse',mirror)
    evaluate('saved_update_reverse_repeat',mirror)
    evaluate('baseline_final',pi)
    assert policy_digest(pi.state_dict())==report['original_policy_digest']
    report['finished']=time.time(); save()
    torch.save(dict(batches=cpu_tree(batches),diagnostic=cpu_tree(diagnostic),candidates=candidates),out/'counterfactuals.pt')
    print('diagnostic_complete',flush=True)


if __name__=='__main__':
    main()
