"""Build hash-bound, source-disjoint train/validation views of accepted data."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_key(row):
    # Screened intervals of the same original clip must never cross the split.
    return row.get('parent_key', row['key']).split('__f')[0]


def split_sources(rows, seed=0):
    groups = sorted({source_key(row) for row in rows})
    if len(groups) < 2:
        raise ValueError('Need at least two source clips for independent validation')
    rng=random.Random(seed)
    count = max(1, round(len(groups) * .1))
    chosen=[]
    # The final task needs held-out aiming and walking, not only action classes
    # such as jumps. Reserve these families when there are multiple originals.
    for family in ('aiming','walk'):
        candidates=[g for g in groups if g.startswith('lafan/') and
                    re.match(r'[A-Za-z]+',g.split('/')[-1]).group(0)==family]
        if len(candidates)>=2 and len(chosen)<count:
            chosen.append(rng.choice(candidates))
    remaining=[g for g in groups if g not in chosen];rng.shuffle(remaining)
    return set(chosen+remaining[:count-len(chosen)])


def read_accepted(paths, namespace):
    rows, identities, seen = [], [], set()
    robot_metadata = None
    for path in paths:
        path = Path(path).resolve()
        manifest = json.loads(path.read_text())
        current_metadata = manifest['robot_metadata']
        if robot_metadata is not None and robot_metadata != current_metadata:
            raise ValueError('Dataset manifests refer to different robot definitions')
        robot_metadata = current_metadata
        if manifest.get('failed_count', 0):
            raise ValueError(f'Incomplete conversion: {path}')
        identities.append(dict(path=str(path), sha256=sha(path)))
        for entry in manifest['entries']:
            if not entry.get('strict_quality_pass'):
                continue
            if not entry['key'].startswith(namespace + '/'):
                raise ValueError('Wrong namespace')
            if entry['key'] in seen:
                raise ValueError('Duplicate motion key')
            for field in ('motion', 'terrain', 'labels', 'quality'):
                if sha(entry[field]) != entry[field + '_sha256']:
                    raise ValueError(f'Changed {field}: {entry["key"]}')
            quality = json.loads(Path(entry['quality']).read_text())
            if not quality['strict_quality_pass']:
                raise ValueError('Manifest/quality disagreement')
            rows.append(entry)
            seen.add(entry['key'])
    if not rows:
        raise ValueError(f'No accepted {namespace} data')
    return rows, identities, robot_metadata


def link_view(root, rows, robot_metadata):
    for row in rows:
        key = row['key']
        for field, relative in [('motion', f'motions/{key}_motion.npz'),
                                ('terrain', f'motions/{key}_terrain.npy'),
                                ('labels', f'contacts/{key}.npz')]:
            dst = root / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(Path(row[field]).resolve())
    sources = sorted({source_key(row) for row in rows})
    result = dict(clips=len(rows), frames=sum(row['frames'] for row in rows),
                  sources=sources, entries=rows, source_count=len(sources),
                  total_frames=sum(row['frames'] for row in rows), robot_metadata=robot_metadata)
    (root / 'manifest.json').write_text(json.dumps(result, indent=2))
    return dict(clips=result['clips'], frames=result['frames'], sources=sources)


def prepare(lafan, jwp, out, seed=0, split=None):
    if out.exists():
        raise ValueError('Use a new output directory; never rewrite a bound dataset')
    lrows, li, lmeta = read_accepted(lafan, 'lafan')
    jrows, ji, jmeta = read_accepted([jwp], 'jwp')
    if lmeta != jmeta:
        raise ValueError('LAFAN and JWP robot definitions differ')
    partitions = {}
    for name, rows in [('lafan', lrows), ('jwp', jrows)]:
        held = set(split[name]['validation_sources']) if split is not None else split_sources(rows, seed)
        if split is not None and not {source_key(r) for r in rows} <= set(split[name]['all_sources']):
            raise ValueError('Accepted source absent from fixed split inventory')
        partitions[name] = {part: [r for r in rows if (source_key(r) in held) == (part == 'validation')]
                            for part in ('train', 'validation')}
        assert not ({source_key(r) for r in partitions[name]['train']} & held)
        if not all(partitions[name].values()):
            raise ValueError(f'{name} needs accepted data in both fixed train and validation sets')
    out.mkdir(parents=True)
    views = {}
    for part in ('train', 'validation'):
        views[f'lafan/{part}'] = link_view(out/'lafan'/part, partitions['lafan'][part], lmeta)
        focus=[row for row in partitions['lafan'][part]
               if row['key'].split('/')[-1].startswith(('aiming','walk'))]
        if focus:
            views[f'lafan_focus/{part}'] = link_view(out/'lafan_focus'/part, focus, lmeta)
            views[f'finetune_focus/{part}'] = link_view(out/'finetune_focus'/part,
                                                       partitions['jwp'][part] + focus, lmeta)
        views[f'jwp/{part}'] = link_view(out/'jwp'/part, partitions['jwp'][part], lmeta)
        views[f'finetune/{part}'] = link_view(out/'finetune'/part,
                                              partitions['jwp'][part] + partitions['lafan'][part], lmeta)
    result = dict(version=1, seed=seed, split_unit='original source clip',
                  source_manifests=li+ji, views=views, finetune_dataset_weights={'jwp': .8, 'lafan': .2},
                  fixed_split=split,
                  scope='All accepted clips; training curricula select task-relevant views explicitly')
    (out/'manifest.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lafan', nargs='+', required=True, type=Path)
    ap.add_argument('--jwp', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split', type=Path, help='Fixed source inventory and holdout assignment JSON')
    args = ap.parse_args()
    result = prepare(args.lafan, args.jwp, args.out, args.seed,
                     json.loads(args.split.read_text()) if args.split else None)
    print(json.dumps(result['views'], indent=2))
