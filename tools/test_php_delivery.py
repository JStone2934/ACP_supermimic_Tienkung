"""Delivery provenance guards: no successful-task replacement or source substitution."""
import tempfile
import unittest
from pathlib import Path

from supermimic.acp.php_conversion import atomic_json
from tools.deliver_php_conversion import assemble


class DeliveryTests(unittest.TestCase):
    def parent(self, root, status, source_hash='original'):
        task = dict(id='task', record={'key': 'category/clip', 'source_hash': source_hash}, solve_variant=0)
        plan = dict(tasks=[task], robot_metadata={}, collision_policy={}, code={}, source_robot_files={}, version='test')
        atomic_json(root/'plan.json', plan)
        atomic_json(root/'manifest.json', {'all_terminal': True})
        atomic_json(root/'tasks/task.json', {'status': status})

    def test_successful_result_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            base, retry, out = [Path(folder)/name for name in ('base', 'retry', 'out')]
            self.parent(base, 'accepted')
            self.parent(retry, 'rejected')
            with self.assertRaisesRegex(ValueError, 'only failed'):
                assemble(base, [retry], out, None)
            self.assertFalse(out.exists())

    def test_retry_cannot_substitute_another_source(self):
        with tempfile.TemporaryDirectory() as folder:
            base, retry, out = [Path(folder)/name for name in ('base', 'retry', 'out')]
            self.parent(base, 'failed')
            self.parent(retry, 'accepted', source_hash='different')
            with self.assertRaisesRegex(ValueError, 'changes input identity'):
                assemble(base, [retry], out, None)
            self.assertFalse(out.exists())

    def test_unresolved_failure_does_not_create_delivery(self):
        with tempfile.TemporaryDirectory() as folder:
            base, out = Path(folder)/'base', Path(folder)/'out'
            self.parent(base, 'failed')
            with self.assertRaisesRegex(ValueError, 'unresolved task'):
                assemble(base, [], out, None)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
