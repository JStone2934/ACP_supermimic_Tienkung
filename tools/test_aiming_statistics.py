import unittest
import numpy as np
from supermimic.acp.aiming_statistics import error_summary


class ErrorStatisticsTests(unittest.TestCase):
    def test_invalid_tail_is_not_float_max(self):
        x=error_summary([.01]*90+[np.inf]*10)
        self.assertIsNone(x['p95_m']);self.assertEqual(x['median_m'],.01)
        self.assertEqual(x['invalid_fraction'],.1)
        self.assertEqual(x['finite_only_p95_m'],.01)

    def test_nan_is_invalid(self):
        x=error_summary([np.nan,np.inf])
        self.assertEqual(x['invalid_count'],2);self.assertIsNone(x['finite_only_p95_m'])

    def test_finite_sample(self):
        self.assertEqual(error_summary(np.arange(1,101))['p95_m'],95.)


if __name__=='__main__':unittest.main()
