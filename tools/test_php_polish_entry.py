import unittest
from unittest.mock import patch
from tools.convert_php import main

class ProtectedPolishEntryTests(unittest.TestCase):
    def test_requested_cpu_count_and_resume_are_preserved(self):
        with patch('tools.polish_php_conversion.main') as run:
            main(['--out','output','--polish-seeds','seeds.json','--workers-per-gpu','0','--cpu-workers','4','--resume'])
            run.assert_called_once_with(['--seeds','seeds.json','--out','output','--workers','4','--resume'])

    def test_worker_reservation_mismatch_is_rejected(self):
        with patch('tools.polish_php_conversion.main') as run:
            with self.assertRaises(SystemExit):
                main(['--out','output','--polish-seeds','seeds.json','--cpu-workers','4'])
            run.assert_not_called()

if __name__=='__main__':unittest.main()
