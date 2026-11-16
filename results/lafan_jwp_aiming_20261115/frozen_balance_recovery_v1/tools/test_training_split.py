import unittest
from tools.prepare_lafan_jwp_training import source_key, split_sources


class SplitTests(unittest.TestCase):
    def test_intervals_follow_original_source(self):
        rows = [dict(key=f'lafan/source{i}__f{j:06d}_{j+100:06d}')
                for i in range(10) for j in (0, 100, 200)]
        held = split_sources(rows)
        self.assertEqual(len(held), 1)
        self.assertEqual(sum(source_key(r) in held for r in rows), 3)
        self.assertEqual(split_sources(rows), split_sources(list(reversed(rows))))

    def test_single_source_is_not_independent(self):
        with self.assertRaises(ValueError):
            split_sources([dict(key='lafan/a__f000000_000100'), dict(key='lafan/a__f000100_000200')])


if __name__ == '__main__':
    unittest.main()
