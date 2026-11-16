"""Dataset proportions must survive unequal lengths and hard-bin weighting."""
import unittest
import torch
from supermimic.acp.dataset_sampling import group_probabilities


class SamplingTests(unittest.TestCase):
    def test_unequal_lengths_and_difficulty(self):
        names = ['jwp/a', 'jwp/b'] + ['lafan/c'] * 100
        base = torch.arange(1., 103.)
        p = group_probabilities(names, {'jwp': .8, 'lafan': .2}, 'cpu', base)
        self.assertAlmostEqual(p[:2].sum().item(), .8, places=6)
        self.assertAlmostEqual(p[2:].sum().item(), .2, places=6)
        self.assertAlmostEqual((p[1]/p[0]).item(), 2.)
        torch.manual_seed(0)
        draws = torch.multinomial(p, 100000, replacement=True)
        self.assertLess(abs((draws < 2).float().mean().item() - .8), .005)

    def test_missing_group_rejected(self):
        with self.assertRaises(ValueError):
            group_probabilities(['jwp/a'], {'jwp': .8, 'lafan': .2}, 'cpu')

    def test_invalid_weights(self):
        for value in [0., -1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                group_probabilities(['jwp/a'], {'jwp': value}, 'cpu')

    def test_zero_group_mass_rejected(self):
        with self.assertRaises(ValueError):
            group_probabilities(['jwp/a', 'lafan/b'], {'jwp': .8, 'lafan': .2}, 'cpu', torch.tensor([0., 1.]))


if __name__ == '__main__':
    unittest.main()
