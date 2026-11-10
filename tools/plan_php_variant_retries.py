"""Plan missing original variants as separate full-length, terrain-specific solves."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from supermimic.acp.php_conversion import atomic_json
from tools.convert_php import load_json


def selections(coverage_report, inventory):
    sources = {}
    for row in inventory['records']:
        path = Path(row['source'])
        sources[path.parent.name+'/'+path.name.removesuffix('_motion.npz')] = row
    rows = []
    for status in coverage_report['sources']:
        key = status['source_key']
        record = sources[key]
        representatives = sorted({record['variant_aliases'][v] for v in status['missing_variants']})
        for v in representatives:
            category, name = key.split('/')
            if len(name) > 180:
                name = name[:170]+'_'+hashlib.sha256(key.encode()).hexdigest()[:8]
            rows.append(dict(source_key=key, key=f'{category}/{name}__terrain_{v:02d}',
                             start=0, end=record['frames'], solve_variant=v))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('coverage', type=Path)
    ap.add_argument('--inventory', type=Path, default=Path('docs/php_conversion/evidence/source_inventory.json'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rows = selections(load_json(args.coverage), load_json(args.inventory))
    atomic_json(args.out, rows)
    print(dict(terrain_specific_solves=len(rows), full_source_files=len({r['source_key'] for r in rows}), out=str(args.out)))


if __name__ == '__main__':
    main()
