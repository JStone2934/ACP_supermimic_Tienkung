"""Model hash reuse must still detect rewritten and replaced assets."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from supermimic.acp import robots


class RobotAssetCacheTests(unittest.TestCase):
    def setUp(self):
        robots._asset_digest.cache_clear()

    def test_unchanged_asset_is_read_once(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'mesh'; p.write_bytes(b'original')
            # Simulate an asset outside the filesystem's recent-write window.
            with patch.object(robots.time, 'time_ns', return_value=p.stat().st_ctime_ns+2_000_000_000), \
                 patch.object(robots, 'sha256', wraps=robots.sha256) as digest:
                self.assertEqual(robots.asset_sha256(p), hashlib.sha256(b'original').hexdigest())
                self.assertEqual(robots.asset_sha256(p), hashlib.sha256(b'original').hexdigest())
                self.assertEqual(digest.call_count, 1)

    def test_same_size_rewrite_with_restored_mtime_rehashes(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'mesh'; p.write_bytes(b'original')
            before = p.stat(); first = robots.asset_sha256(p)
            p.write_bytes(b'modified')
            os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertNotEqual(robots.asset_sha256(p), first)

    def test_atomic_replacement_rehashes(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'mesh'; p.write_bytes(b'original')
            first = robots.asset_sha256(p)
            replacement = Path(directory)/'new'; replacement.write_bytes(b'modified')
            replacement.replace(p)
            self.assertNotEqual(robots.asset_sha256(p), first)

    def test_mutation_during_hashing_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'mesh'; p.write_bytes(b'original')
            def changed(path):
                digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                Path(path).write_bytes(b'modified')
                return digest
            with patch.object(robots, 'sha256', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'changed while hashing'):
                    robots.asset_sha256(p)


if __name__ == '__main__':
    unittest.main()
