"""Materialize the audited PHP index as one full clip per original terrain slot.

This keeps uniform clip sampling uniform over original source/variant pairs and
makes the existing first-variant evaluator cover every original terrain slot.
No retargeting, resampling, relabeling, or quality threshold changes occur here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

VARIANT_FIELDS = {
    'contact_info', 'contact_dist', 'contact_pos_w', 'contact_normal_w',
    'contact_surface', 'other_dist', 'other_contact', 'boxes',
    'quality_pass', 'source_variants',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_labels(labels, slot, motion_hash, terrain_hash):
    count = len(labels['contact_info'])
    if not 0 <= slot < count or not bool(labels['quality_pass'][slot]):
        raise ValueError('unaccepted or invalid export slot')
    result = {}
    for key, value in labels.items():
        if key in VARIANT_FIELDS:
            if value.shape[0] != count:
                raise ValueError(f'inconsistent variant axis: {key}')
            result[key] = value[slot:slot + 1]
        else:
            result[key] = value
    result['motion_hash'] = np.asarray(motion_hash)
    result['terrain_hash'] = np.asarray(terrain_hash)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--delivery', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    receipt = json.loads(args.delivery.read_text())
    for path_key, hash_key in [('asset_index', 'asset_index_sha256'), ('final_coverage', 'final_coverage_sha256')]:
        if sha(receipt[path_key]) != receipt[hash_key]:
            raise ValueError(f'delivery hash mismatch: {path_key}')
    index = json.loads(Path(receipt['asset_index']).read_text())
    coverage = json.loads(Path(receipt['final_coverage']).read_text())
    if not coverage['all_original_variants_pass'] or coverage['missing_variants'] or index['snapshot_of_running_parents']:
        raise ValueError('requires completed full acceptance')
    expected = {(s, v) for s, ident in index['source_identities'].items() for v in range(ident['variants'])}
    entries = sorted(index['entries'], key=lambda e: (e['source_key'], e['source_variant']))
    if {(e['source_key'], e['source_variant']) for e in entries} != expected or len(entries) != len(expected):
        raise ValueError('missing or duplicate original pair')
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows, cache, verified = [], {}, {}
    for i, entry in enumerate(entries):
        for field in ('motion', 'labels', 'quality'):
            path = entry[field]
            actual = verified.setdefault(path, sha(path)) if path not in verified else verified[path]
            if actual != entry[field + '_sha256']:
                raise ValueError(f'changed accepted artifact: {path}')
        label_path = entry['labels']
        if cache.get('path') != label_path:
            with np.load(label_path, allow_pickle=False) as z:
                labels = {k: z[k] for k in z.files}
            terrain = np.load(entry['terrain'], allow_pickle=False)
            if str(labels['terrain_hash'].item()) != sha(entry['terrain']):
                raise ValueError('accepted terrain hash mismatch')
            cache = dict(path=label_path, labels=labels, terrain=terrain)
        labels, terrain = cache['labels'], cache['terrain']
        slot, variant = entry['export_slot'], entry['source_variant']
        if int(labels['source_variants'][slot]) != variant:
            raise ValueError('original variant/export slot mismatch')
        key = entry['source_key'] + f'__variant_{variant:02d}'
        motion_out, terrain_out, labels_out = root/'motions'/f'{key}_motion.npz', root/'motions'/f'{key}_terrain.npy', root/'contacts'/f'{key}.npz'
        motion_out.parent.mkdir(parents=True, exist_ok=True)
        labels_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry['motion'], motion_out)
        np.save(terrain_out, terrain[:, slot:slot+1])
        selected = select_labels(labels, slot, sha(motion_out), sha(terrain_out))
        np.savez_compressed(labels_out, **selected)
        with np.load(motion_out, allow_pickle=False) as motion:
            frames = len(motion['joint_pos'])
            identity = index['source_identities'][entry['source_key']]
            if frames != identity['source_frames'][1] or list(motion['source_frames']) != identity['source_frames'] or float(motion['fps'].item()) != 50:
                raise ValueError('full source frame/rate identity mismatch')
        with np.load(labels_out, allow_pickle=False) as saved:
            for field, value in selected.items():
                np.testing.assert_equal(saved[field], value)
            if saved['contact_info'].shape[1] != frames:
                raise ValueError('motion/label frame mismatch')
        np.testing.assert_equal(np.load(terrain_out), terrain[:, slot:slot+1])
        rows.append(dict(key=key, source_key=entry['source_key'], source_variant=variant,
                         parent=entry['parent'], parent_export_slot=slot, parent_labels_sha256=entry['labels_sha256'],
                         quality=entry['quality'], quality_sha256=entry['quality_sha256'], frames=frames,
                         motion=str(motion_out), terrain=str(terrain_out), labels=str(labels_out),
                         motion_sha256=sha(motion_out), terrain_sha256=sha(terrain_out), labels_sha256=sha(labels_out)))
        if (i+1) % 200 == 0:
            print(f'materialized {i+1}/{len(entries)}', flush=True)
    manifest = dict(created_at=dt.datetime.now().astimezone().isoformat(), kind='stage1-original-pairs-v1',
                    delivery=str(args.delivery.resolve()), delivery_sha256=sha(args.delivery),
                    asset_index_sha256=receipt['asset_index_sha256'], robot_metadata=index['robot_metadata'],
                    source_count=len(index['source_identities']), pair_count=len(rows), total_frames=sum(r['frames'] for r in rows),
                    categories=dict(Counter(r['source_key'].split('/')[0] for r in rows)),
                    sampling='one full clip per original terrain slot; all slots used for training; no held-out claim',
                    entries=rows)
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('entries', 'robot_metadata')}, indent=2))


if __name__ == '__main__':
    main()
