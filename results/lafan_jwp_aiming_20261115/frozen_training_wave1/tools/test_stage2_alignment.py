"""CPU regressions for Stage 1 -> offline planner -> Stage 2 contracts."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from supermimic.acp.stage2_contracts import (
    inherit_execution_sizes, initialize_critic, validate_planner_init, validation_split,
)


def entry_parser(name):
    """Use the real CLI definitions without entering training or creating a GPU runtime."""
    path = Path('supermimic/acp')/(name+'.py')
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets):
            break
        nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['parser']


class Stage2AlignmentTests(unittest.TestCase):
    def test_actual_stage2_cli_inherits_global_sizes_and_execution(self):
        args = entry_parser('train_stage2').parse_args(['--init', 'saved.pt', '--log_dir', 'unused', '--save_full_checkpoints'])
        ck = {'args': dict(num_envs=512, wm_batch=256, pi_batch=256, reward_chunk_size=0, buffer_steps=1000)}
        inherit_execution_sizes(args, ck)
        self.assertEqual((args.num_envs, args.wm_batch, args.pi_batch, args.reward_chunk_size), (512, 256, 256, 0))
        self.assertEqual(args.buffer_steps, 1350)
        self.assertTrue(args.save_full_checkpoints)

    def test_actual_cli_override_zero_and_resume_sizes(self):
        parser = entry_parser('train_stage2')
        ck = {'args': dict(num_envs=1024, wm_batch=512, pi_batch=256, reward_chunk_size=64)}
        args = parser.parse_args(['--resume', 'saved.pt', '--log_dir', 'unused', '--reward_chunk_size', '0', '--num_envs', '128'])
        inherit_execution_sizes(args, ck)
        self.assertEqual((args.num_envs, args.wm_batch, args.pi_batch, args.reward_chunk_size), (128, 512, 256, 0))
        args = parser.parse_args(['--resume', 'saved.pt', '--log_dir', 'unused'])
        inherit_execution_sizes(args, ck)
        self.assertEqual((args.num_envs, args.reward_chunk_size), (1024, 64))

    def test_offline_cli_inherits_environment_count(self):
        args = entry_parser('train_planner_offline').parse_args(['--tracker', 'saved.pt', '--log_dir', 'unused'])
        inherit_execution_sizes(args, {'args': {'num_envs': 512}})
        self.assertEqual(args.num_envs, 512)
        self.assertEqual(args.buffer_steps, 1350)

    def test_legacy_defaults(self):
        args = SimpleNamespace(num_envs=None, wm_batch=None, pi_batch=None, reward_chunk_size=None)
        inherit_execution_sizes(args, {})
        self.assertEqual(vars(args), dict(num_envs=4096, wm_batch=2048, pi_batch=4096, reward_chunk_size=0))

    def fixture(self, root):
        entries = [dict(key=f'{cat}/clip_{i:02}__variant_{v:02}', source_key=f'{cat}/clip_{i:02}')
                   for cat in ('climb', 'walk') for i in range(12) for v in range(3)]
        (root/'manifest.json').write_text(json.dumps({'kind': 'stage1-original-pairs-v1', 'entries': entries}))
        return entries

    def test_all_sibling_variants_stay_together_and_resume_is_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);entries = self.fixture(root);names = [e['key'] for e in entries]
            mask, record = validation_split(names, root/'motions')
            train = {e['source_key'] for e, val in zip(entries, mask) if not val}
            val = {e['source_key'] for e, selected in zip(entries, mask) if selected}
            self.assertFalse(train & val)
            self.assertEqual(len(mask), 72)
            self.assertEqual(sum(mask), 12)
            self.assertEqual(record['sources'], 24)
            self.assertEqual(validation_split(names, root/'motions', saved=record, continuing=True), (mask, record))
            with self.assertRaisesRegex(ValueError, 'no source-group'):
                validation_split(names, root/'motions', continuing=True)
            bad = dict(record, sha256='changed')
            with self.assertRaisesRegex(ValueError, 'split changed'):
                validation_split(names, root/'motions', saved=bad, continuing=True)

    def test_wrong_manifest_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);entries = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                validation_split([e['key'] for e in entries][1:], root/'motions')

    def test_ordinary_dataset_retains_legacy_split(self):
        with tempfile.TemporaryDirectory() as directory:
            names = [f'walk/{i:02}' for i in range(21)]
            mask, record = validation_split(names, Path(directory)/'motions', continuing=True)
            self.assertEqual([i for i, val in enumerate(mask) if val], [0, 10, 20])
            self.assertEqual(record['grouping'], 'clip')

    def test_planner_must_match_tracker_data_and_geometry(self):
        tracker = dict(data_hash='dataset', ctc_cfg={'contact_geometry': 'spheres224'})
        planner = dict(data_hash='dataset', tracker_hash='tracker', ctc_cfg=tracker['ctc_cfg'])
        validate_planner_init(planner, tracker, 'tracker', robot_id='tienkung3_29dof')
        for key in ('data_hash', 'tracker_hash'):
            for missing in (True, False):
                bad = copy.deepcopy(planner)
                if missing:del bad[key]
                else:bad[key] = 'different'
                with self.subTest(key=key, missing=missing), self.assertRaisesRegex(ValueError, key):
                    validate_planner_init(bad, tracker, 'tracker', robot_id='tienkung3_29dof')
        bad = dict(planner, ctc_cfg={'contact_geometry': 'legacy'})
        with self.assertRaisesRegex(ValueError, 'contact configuration'):
            validate_planner_init(bad, tracker, 'tracker', robot_id='tienkung3_29dof')
        validate_planner_init({}, {}, 'legacy', robot_id='g1')

    def test_critic_value_and_prefix_gradient_survive_widening(self):
        from supermimic.acp.networks import Critic
        torch.manual_seed(224)
        before, after = Critic(11), Critic(40)
        before.norm.update(torch.randn(100, 11)+2)
        initialize_critic(after, before.state_dict())
        x = torch.randn(7, 11, requires_grad=True)
        extra = torch.randn(7, 29, requires_grad=True)
        original, widened = before(x), after(torch.cat((x, extra), -1))
        torch.testing.assert_close(original, widened, rtol=1e-5, atol=1e-6)
        gx, ge = torch.autograd.grad(widened.sum(), (x, extra), retain_graph=True)
        torch.testing.assert_close(gx, torch.autograd.grad(original.sum(), x)[0], rtol=1e-5, atol=1e-6)
        self.assertEqual(int(torch.count_nonzero(ge)), 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
