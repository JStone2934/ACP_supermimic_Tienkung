"""CPU audit of real Stage 1 assets and the Stage 2 reward/initialization path."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from supermimic.acp.robots import resolve_robot, sha256
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.motion import MotionLib
from supermimic.acp.mdp import MDP
from supermimic.acp.networks import Critic, Policy, WorldModel
from supermimic.acp.task import ContactCfg, ContactTask, PaperReward, PaperRewardCfg
from supermimic.acp.memory import reward_batch, checkpoint_reward
from supermimic.acp.stage2_contracts import inherit_execution_sizes, initialize_critic, validation_split


def reward_entry(path, namespace):
    node = next(n for n in ast.parse(Path(path).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_reward_term')
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    return namespace['_reward_term']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(224)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    robot = resolve_robot(None, ck)
    motion_dir = Path(ck['args']['motion_dir'])
    manifest_path = motion_dir.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    entries = manifest['entries']
    by_name = {e['key']: e for e in entries}
    names = sorted(by_name)
    mask, split = validation_split(names, motion_dir)
    train = {by_name[n]['source_key'] for n, val in zip(names, mask) if not val}
    val = {by_name[n]['source_key'] for n, selected in zip(names, mask) if selected}
    assert not train & val
    inherited = SimpleNamespace(num_envs=None, wm_batch=None, pi_batch=None, reward_chunk_size=None)
    inherit_execution_sizes(inherited, ck)
    chosen = {}
    for entry in entries:
        chosen.setdefault(entry['source_key'].split('/')[0], entry)
    for entry in chosen.values():
        for key in ('motion', 'terrain', 'labels'):
            assert sha256(entry[key]) == entry[key+'_sha256']
    kin = RobotKinematics('cpu', robot)
    motion = MotionLib([e['motion'] for e in chosen.values()], kin, 'cpu', str(motion_dir))
    mdp = MDP(kin, motion, torch.tensor(robot.default_q))
    ctc = ContactTask(ContactCfg(**ck['ctc_cfg']), mdp, str(motion_dir), 'cpu')
    reward = PaperReward(PaperRewardCfg(**ck['reward_cfg']), ctc)
    packed = torch.cat((ctc.sph_pts, ctc.sph_rad[:, None]), -1).numpy().astype('<f4')
    proxy_hash = hashlib.sha256(packed.tobytes()).hexdigest()
    assert len(ctc.sph_pts) == 224 and not ctc.hulls
    assert proxy_hash == 'd97738195ac0160804dacee9306f68ba29e14ca05cbc8eed465feab4bc021a2f'
    pi = Policy(robot.obs_dim, ctc.cmd_dim, hist=ck['args']['hist'], robot=robot)
    wm = WorldModel(mdp.scan_dim, robot=robot)
    pi.load_state_dict(ck['pi']);wm.load_state_dict(ck['wm'])
    old_dim = ck['critic']['net.0.weight'].shape[1]
    old, new = Critic(old_dim), Critic(old_dim+3*len(robot.contact_bodies)+2)
    old.load_state_dict(ck['critic']);initialize_critic(new, ck['critic'])
    x = torch.randn(17, old_dim)
    extra = torch.randn(17, 3*len(robot.contact_bodies)+2)
    with torch.no_grad():
        before, after = old(x), new(torch.cat((x, extra), -1))
    torch.testing.assert_close(before, after, rtol=2e-5, atol=2e-6)
    f = torch.linspace(1, motion.num_frames-2, 129).long()
    v = torch.zeros_like(f)
    ref = motion.ref_state(f)
    state = tuple((t+.001*torch.randn_like(t)).requires_grad_() for t in ref)
    previous = tuple(t.detach().clone() for t in ref)
    action = (torch.randn(129, robot.num_joints)*.01).requires_grad_()
    old_action = torch.zeros_like(action)
    xi = torch.zeros(129, robot.xi_dim)
    inputs = (state, previous, f, v, action, old_action, old_action, xi)
    pw, pv = ctc.next_pos[f, v], ctc.next_valid[f, v].clone()
    pv[::3] = False
    stage1 = reward_entry('supermimic/acp/train_stage1.py', dict(paper_reward=reward, mdp=mdp))
    stage2 = reward_entry('supermimic/acp/train_stage2.py', dict(paper_reward=reward))
    def second(S, Sp, f, v, a, ap, app, xi, pw, pv, eta):
        return stage2(S, Sp, f, v, a, ap, app, xi, pw, pv, eta), mdp.soft_termination(S, f)
    base = stage1(*inputs)
    base_grad = torch.autograd.grad(sum(t.sum() for t in base), (*state, action))
    results = {}
    for eta in (0., .5, 1.):
        full_inputs = (*inputs, pw, pv, torch.tensor(eta))
        value = second(*full_inputs)
        grad = torch.autograd.grad(sum(t.sum() for t in value), (*state, action))
        if eta == 0:
            for a, b in zip(value, base):torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
            for a, b in zip(grad, base_grad):torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
        chunked = reward_batch(checkpoint_reward(second), *full_inputs, chunk_size=17)
        chunk_grad = torch.autograd.grad(sum(t.sum() for t in chunked), (*state, action))
        for a, b in zip(value, chunked):torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        for a, b in zip(grad, chunk_grad):torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
        results[str(eta)] = dict(max_chunk_output_error=max(float((a-b).abs().max().detach()) for a,b in zip(value,chunked)),
                                max_chunk_gradient_error=max(float((a-b).abs().max()) for a,b in zip(grad,chunk_grad)))
    result = dict(passed=True, scope='CPU FP32 contract and actual reward/gradient checks, not a GPU Stage2 training benchmark',
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                  stage1_iteration=ck['iteration'], robot=robot.robot_id, checkpoint_data_hash=ck['data_hash'],
                  dataset_manifest_sha256=sha256(manifest_path), inherited_execution=vars(inherited),
                  contact_geometry=ctc.cfg.contact_geometry, sphere_count=224, reward_hulls=0, sphere_sha256=proxy_hash,
                  split=split, planner_training_sources=len(train), planner_training_clips=len(mask)-sum(mask),
                  tracker_training_clips=len(mask), longest_clip_frames=max(e['frames'] for e in entries), stage2_buffer_steps=1350,
                  world_model_and_policy_weights_load_strict=True, critic_widening_max_error=float((before-after).abs().max()),
                  stage1_equals_stage2_eta_zero=True, reward_poses=129, reward_categories=list(chosen), reward_comparisons=results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':main()
