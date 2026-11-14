"""Versioned artifacts and immutable source snapshots for v2 runs."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import torch


def json_save(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            x = x.item()
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def save_checkpoint(path, checkpoint):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(path) + '.tmp')
    os.replace(str(path) + '.tmp', path)


def code_identity():
    paths = list(Path('supermimic').rglob('*.py')) + list(Path('tools').glob('*fullbody*.py'))
    paths += list(Path('docs/config/standing_aiming_fullbody_v2').glob('*.json'))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def snapshot(source, destination):
    """Copy an opened inode so atomic replacement by the old trainer is harmless."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(source, 'rb') as src, destination.open('xb') as dst:
        before = os.fstat(src.fileno())
        shutil.copyfileobj(src, dst)
        after = os.fstat(src.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        destination.unlink()
        raise RuntimeError('Source changed in place during snapshot; retry preparation in a new run directory')
    from .robots import sha256
    return sha256(destination)
