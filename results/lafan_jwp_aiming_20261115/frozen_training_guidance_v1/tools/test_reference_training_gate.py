import tempfile
import unittest
from pathlib import Path
from tools.check_reference_training_gate import check
from tools.prepare_lafan_jwp_training import sha


class GateTests(unittest.TestCase):
    def test_short_survival_cannot_pass_low_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp)/'ck'; ck.write_bytes(b'checkpoint')
            report = dict(checkpoint_sha256=sha(ck), per_clip={'lafan/a': dict(
                expected_steps=500, alive_steps=20, failed=False, finite=True, root_rmse_m=.01, joint_rmse_rad=.01)})
            manifest = dict(entries=[dict(key='lafan/a', frames=501)])
            self.assertFalse(check(report, manifest, ck)['passed'])
            report['per_clip']['lafan/a']['alive_steps'] = 500
            self.assertTrue(check(report, manifest, ck)['passed'])
            report['per_clip']['lafan/a']['finite'] = False
            self.assertFalse(check(report, manifest, ck)['passed'])
            report['per_clip'] = {}
            with self.assertRaises(ValueError):check(report, manifest, ck)

    def test_replaced_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp)/'ck'; ck.write_bytes(b'new')
            with self.assertRaises(ValueError):
                check(dict(checkpoint_sha256='old'), {}, ck)


if __name__ == '__main__':unittest.main()
