import tempfile
from pathlib import Path
import unittest
from supermimic.acp.robots import sha256
from tools.prepare_wide_aiming_init import verify_gates,verify_recovery_evidence


class InitializationGateTests(unittest.TestCase):
    def test_recovery_retains_failure_and_requires_actual_matching_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';path.write_bytes(b'failed checkpoint')
            reference=dict(checkpoint_sha256=sha256(path),passed=False,clips=59,successful_fraction=.05)
            standing=dict(checkpoint_sha256=sha256(path),passed=False,success_rate=0.,
                elapsed_sim_seconds=6.,continuous_without_reset=True,scenario_count=100)
            self.assertEqual(verify_recovery_evidence(path,reference,standing),sha256(path))
            self.assertFalse(reference['passed']);self.assertFalse(standing['passed'])
            with self.assertRaises(ValueError):verify_gates(path,reference,standing)
            for changes in ({'continuous_without_reset':False},{'scenario_count':99},{'elapsed_sim_seconds':0},
                            {'checkpoint_sha256':'wrong'}):
                with self.assertRaises(ValueError):verify_recovery_evidence(path,reference,standing|changes)

    def test_both_stage_gates_must_match_exact_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';path.write_bytes(b'checkpoint')
            reference=dict(checkpoint_sha256=sha256(path),passed=True,successful_fraction=.9)
            standing=dict(checkpoint_sha256=sha256(path),passed=True,success_rate=.95,
                elapsed_sim_seconds=30.,continuous_without_reset=True,scenario_count=100)
            self.assertEqual(verify_gates(path,reference,standing),sha256(path))
            for changes in ({'passed':False},{'elapsed_sim_seconds':29.9},{'success_rate':.94},
                            {'continuous_without_reset':False},{'scenario_count':99}):
                with self.assertRaises(ValueError):verify_gates(path,reference,standing|changes)
            path.write_bytes(b'new checkpoint')
            with self.assertRaises(ValueError):verify_gates(path,reference,standing)


if __name__=='__main__':unittest.main()
