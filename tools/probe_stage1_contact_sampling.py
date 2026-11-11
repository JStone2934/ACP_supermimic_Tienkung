"""CPU-only screening of sparse reward geometry; never changes training assets.

Compare point-only and G1-style sphere proxies while retaining the same convex
hull penetration calculation. This is a diagnostic, not a promotion gate.
"""
import argparse
from collections import defaultdict
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import torch

from supermimic.acp.geometry import hull_box_penetration
from supermimic.acp.robots import get_robot
from supermimic.acp.task import BodySpheres, ContactCfg, _farthest_points, debounce
from supermimic.acp.terrain import box_sdf
from supermimic.rotations import quat_to_mat


def summary(x):
    x = np.asarray(x)
    return dict(count=int(x.size), mean=float(x.mean()), p95=float(np.quantile(x, .95)), max=float(x.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--chunk', type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    groups = defaultdict(list)
    for row in manifest['entries']:
        if row['source_variant'] == 0:
            groups[row['source_key'].split('/')[0]].append(row)
    entries = {(r['source_key'], r['source_variant']): r for r in manifest['entries']}
    rng = np.random.default_rng(20261110)
    selected = []
    for category, rows in sorted(groups.items()):
        for j, idx in enumerate(rng.choice(len(rows), 3, replace=False)):
            selected.append(entries[(rows[int(idx)]['source_key'], (0, 5, 9)[j])])
    robot = get_robot('tienkung3_29dof')
    sp = BodySpheres(robot)
    names = list(robot.contact_bodies)
    candidates = {'dense': (sp.pts, sp.rad, sp.bounds)}
    # The promoted G1 distance path removes the independent hull override as well as surface sampling.
    accepted = BodySpheres(robot, 'spheres224')
    candidates['spheres224_no_hull'] = (accepted.pts, accepted.rad, accepted.bounds)
    for label, counts, spheres in [
        ('surface_224', [28, 28]+[24]*7, False),
        ('g1_style_224', [28, 28]+[24]*7, True),
        ('surface_864', [96]*9, False),
        ('surface_2304', [256]*9, False),
    ]:
        points, radius, bounds = [], [], []
        offset = 0
        for b, ((s, e), count) in enumerate(zip(sp.bounds, counts)):
            cloud = sp.hulls[b]['vertices'].numpy() if spheres else sp.pts[s:e].numpy()
            chosen = _farthest_points(cloud, count)
            if spheres:
                inward = chosen-cloud.mean(0)
                chosen = chosen-.01*inward/np.linalg.norm(inward, axis=-1, keepdims=True).clip(min=1e-6)
            points.append(chosen)
            radius.extend([.01 if spheres else 0.]*len(chosen))
            bounds.append((offset, offset+len(chosen)))
            offset += len(chosen)
        candidates[label] = (torch.tensor(np.concatenate(points)), torch.tensor(radius), bounds)
    samples = []
    for entry in selected:
        with np.load(entry['labels'], allow_pickle=False) as z:
            bp, bq = z['body_pos_w'], z['body_quat_w']
            labels = debounce(z['contact_info'][0, ..., 0]>.5, 2)
            native = z['contact_dist'][0]
            boxes = z['boxes'][0]
        with np.load(entry['motion'], allow_pickle=False) as z:
            root = z['joint_pos'][:, :3]
        frames = list(np.linspace(0, len(bp)-1, 8).astype(int))
        support = np.flatnonzero(labels[:, 2:].any(-1))
        if len(support):
            frames[3:5] = [int(support[0]), int(support[len(support)//2])]
        for frame in sorted(set(frames)):
            samples.append(dict(key=entry['key'], frame=frame, root=root[frame], bp=bp[frame], bq=bq[frame],
                                labels=labels[frame], native=native[frame], boxes=boxes))
    max_boxes = max(1, max(len(s['boxes']) for s in samples))
    for sample in samples:
        sample['boxes'] = np.pad(sample['boxes'], ((0, max_boxes-len(sample['boxes'])), (0, 0)))
    result = {label: [] for label in candidates}
    elapsed = defaultdict(float)
    cfg = ContactCfg()
    for begin in range(0, len(samples), args.chunk):
        batch = samples[begin:begin+args.chunk]
        bp = torch.tensor(np.stack([s['bp'] for s in batch]))
        bR = quat_to_mat(torch.tensor(np.stack([s['bq'] for s in batch])))
        roots = torch.tensor(np.stack([s['root'] for s in batch]))
        boxes = torch.tensor(np.stack([s['boxes'] for s in batch]))
        cx, cy, _, c, s, hx, hy, _ = boxes.unbind(-1)
        dx, dy = roots[:, 0:1]-cx, roots[:, 1:2]-cy
        q = torch.stack(((c*dx+s*dy).abs()-hx, (c*dy-s*dx).abs()-hy), -1)
        distance = torch.where(hx>0, torch.relu(q).norm(dim=-1), torch.inf)
        indices = distance.topk(min(cfg.near_boxes, max_boxes), largest=False).indices
        near = boxes.gather(1, indices[..., None].expand(-1, -1, 8))
        before = time.monotonic()
        penetration = {(b, k): hull_box_penetration(bp[:, b], bR[:, b], near[:, k], sp.hulls[b])
                       for b in range(9) for k in range(near.shape[1])}
        elapsed['shared_hull_cpu_seconds'] += time.monotonic()-before
        for label, (pts, rad, bounds) in candidates.items():
            before = time.monotonic()
            distances = []
            for b, (s, e) in enumerate(bounds):
                world = bp[:, b, None]+pts[s:e]@bR[:, b].transpose(-1, -2)
                d = (world[..., 2]-rad[s:e]).amin(-1)
                for k in range(near.shape[1]):
                    local = (box_sdf(world, near[:, k])-rad[s:e]).amin(-1)
                    if label != 'spheres224_no_hull':
                        local = torch.where((local<0)&(near[:, k, 5]>0), penetration[b, k], local)
                    d = torch.minimum(d, local)
                distances.append(d)
            result[label].append(torch.stack(distances, -1).numpy())
            elapsed[label+'_point_cpu_seconds'] += time.monotonic()-before
        print(f'checked {min(begin+args.chunk, len(samples))}/{len(samples)} poses', flush=True)
    arrays = {label: np.concatenate(value) for label, value in result.items()}
    dense = arrays['dense']
    native = np.stack([s['native'] for s in samples])
    labels = np.stack([s['labels'] for s in samples])
    sigmoid = lambda x: 1/(1+np.exp(-np.clip((cfg.touch_dist-x)/cfg.touch_temp, -80, 80)))
    dense_soft = sigmoid(dense)
    near_native = native < .05
    reports = {}
    for label, distances in arrays.items():
        predicted = distances <= cfg.touch_dist
        dense_touch, native_touch = dense <= cfg.touch_dist, native <= cfg.touch_dist
        soft = sigmoid(distances)
        reward_delta = ((soft-dense_soft)*(2*labels-1)).mean(-1)
        reports[label] = dict(points=len(candidates[label][0]),
            distance_error_vs_dense_near_native_m=summary(np.abs(distances-dense)[near_native]),
            distance_error_vs_native_near_m=summary(np.abs(distances-native)[near_native]),
            disagreements_vs_dense=int(np.sum(predicted!=dense_touch)),
            disagreements_vs_native=int(np.sum(predicted!=native_touch)),
            missed_native_touch=int(np.sum(~predicted & native_touch)),
            false_native_touch=int(np.sum(predicted & ~native_touch)),
            native_touch_count=int(native_touch.sum()), body_frame_count=int(native.size),
            soft_touch_error_vs_dense=summary(np.abs(soft-dense_soft)),
            weighted_stage1_reward_delta=summary(np.abs(reward_delta)*5*.02),
            per_body={names[b]: dict(missed_native_touch=int(np.sum(~predicted[:, b]&native_touch[:, b])),
                                    false_native_touch=int(np.sum(predicted[:, b]&~native_touch[:, b])),
                                    disagreements_vs_dense=int(np.sum(predicted[:, b]!=dense_touch[:, b]))) for b in range(9)})
    out = dict(time=dt.datetime.now().astimezone().isoformat(), diagnostic_only=True, promoted=False,
               samples=len(samples), source_motions=len(selected), sampling='3 sources per category, original variants 0/5/9, up to 8 full-clip-spanning frames plus nonfoot support',
               reports=reports, cpu_seconds=dict(elapsed), total_seconds=time.monotonic()-started,
               limitation='Reference poses only; no learned-policy perturbations or gradient/rollout validation. CPU timings are not GPU training speed estimates. Only spheres224_no_hull removes the convex-hull penetration override.',
               sample_ids=[dict(key=s['key'], frame=int(s['frame'])) for s in samples])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='per_body'} for k,v in reports.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
