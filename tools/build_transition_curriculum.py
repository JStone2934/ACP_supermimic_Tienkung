"""Build a versioned Stage1 transition reset plan from an existing strict full evaluation and contact labels.

No conversion or training is launched. All climb clips remain included, and the training
CLI validates the resulting dataset identity. Example:
python -m tools.build_transition_curriculum --evaluation <strict_eval.json> \
    --motion-dir <motions> --contacts-dir <contacts> --out <plan.json>
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from supermimic.acp.task import debounce, drop_short
from supermimic.acp.transition_curriculum import TransitionCfg
import dataclasses


def event_windows(labels, failure, failed, dt, before=.3, after=.6, advance=.15, max_events=6):
    """Prioritize failure, first hand support and nearby foot/knee/hand switches; keep full clip timing."""
    T = len(labels)
    before_n, after_n = max(1, round(before/dt)), max(1, round(after/dt))
    if T < before_n + after_n + 1:
        raise ValueError('clip too short for transition window')
    onset = labels & ~np.vstack((labels[:1], labels[:-1]))
    hands = np.flatnonzero(onset[:,4:6].any(-1))
    first_hand = int(hands[0]) if len(hands) else None
    candidates = {}
    if failed:
        candidates[min(max(int(failure), 1), T-2)] = 'evaluation_failure'
    if first_hand is not None and first_hand < T-1:
        candidates.setdefault(first_hand, 'first_hand_contact')
    center = min(max(int(failure), 1), T-2) if failed else first_hand
    if center is None:
        raise ValueError('climb has neither a failure nor a hand-contact event')
    switches = np.flatnonzero((labels[1:,:6] != labels[:-1,:6]).any(-1))+1
    radius = round(1.5/dt)
    for t in sorted(switches.tolist(), key=lambda t:(abs(t-center), t)):
        if abs(t-center) > radius or t >= T-1:
            continue
        if any(abs(t-x) < max(1, round(advance/dt)) for x in candidates):
            continue
        candidates.setdefault(t, 'nearby_support_switch')
        if len(candidates) >= max_events:
            break
    result = []
    for t, reason in sorted(candidates.items()):
        end = min(T-1, t+after_n)
        start = max(0, min(t-before_n, end-before_n-after_n))
        result.append(dict(start=int(start), anchor=int(t), end=int(end), reason=reason))
    return result


def build(evaluation, motion_dir, contacts_dir):
    evaluation = Path(evaluation); motion_dir=Path(motion_dir); contacts_dir=Path(contacts_dir)
    ev = json.loads(evaluation.read_text())
    if ev.get('term_window') != 0 or ev.get('interface') != 'reference':
        raise ValueError('use a strict reference-target evaluation')
    actual = {str(p.relative_to(motion_dir))[:-len('_motion.npz')] for p in motion_dir.glob('*/*_motion.npz')}
    if actual != set(ev['per_clip']):
        raise ValueError('evaluation must cover the full requested dataset exactly')
    if ev.get('evaluation_motion_dir') and Path(ev['evaluation_motion_dir']).resolve() != motion_dir.resolve():
        raise ValueError('evaluation motion directory mismatch')
    groups = ['motions-high-climb-76','motions-low-climb-76']
    clips={}; dt=None
    for name in sorted(actual):
        if name.split('/')[0] not in groups:continue
        path=motion_dir/(name+'_motion.npz'); lp=contacts_dir/(name+'.npz')
        with np.load(path) as d:
            frames=len(d['joint_pos'])
        with np.load(lp) as d:
            if str(d['motion_hash'].item()) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f'stale label motion hash: {name}')
            bodies=list(d['body_names'])
            if bodies[:6] != ['left_foot','right_foot','left_knee','right_knee','left_hand','right_hand']:
                raise ValueError('unexpected contact-body order')
            sample_dt=1/float(d['fps'])
            if dt is not None and abs(dt-sample_dt)>1e-9:raise ValueError('mixed control rates')
            dt=sample_dt
            # Expanded stage1 inputs have one original terrain variant per clip.
            if len(d['contact_info']) != 1:raise ValueError('plan builder requires expanded one-variant clips')
            labels=drop_short(debounce(d['contact_info'][0,...,0]>.5,2),10)
        if len(labels)!=frames:raise ValueError('label length mismatch')
        r=ev['per_clip'][name]
        clips[name]=dict(frames=frames,events=event_windows(labels,r['alive_steps'],r['failed'],dt))
    if not clips:raise ValueError('no climbing clips')
    return dict(version=1,data_hash=ev['data_hash'],control_dt=dt,groups=groups,config=dataclasses.asdict(TransitionCfg()),
                source=dict(evaluation=str(evaluation.resolve()),evaluation_sha256=hashlib.sha256(evaluation.read_bytes()).hexdigest(),
                            checkpoint=ev['ckpt'],motion_dir=str(motion_dir.resolve()),contacts_dir=str(contacts_dir.resolve()),
                            initial_lead_s=.3,post_event_s=.6),clips=clips)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',required=True);p.add_argument('--motion-dir',required=True)
    p.add_argument('--contacts-dir',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(); out=Path(a.out)
    if out.exists():raise FileExistsError(f'refuse to overwrite {out}')
    plan=build(a.evaluation,a.motion_dir,a.contacts_dir)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(serialize_plan(plan))
    print(json.dumps(dict(out=str(out),clips=len(plan['clips']),windows=sum(len(c['events']) for c in plan['clips'].values()),config=plan['config']),indent=2))

def serialize_plan(plan):
    # Keep each generated clip on one line; configuration remains easy to review.
    header=json.dumps({k:v for k,v in plan.items() if k!='clips'},indent=2)
    clips=',\n'.join('    '+json.dumps(n)+': '+json.dumps(c,separators=(',',':')) for n,c in plan['clips'].items())
    return header[:-2]+',\n  "clips": {\n'+clips+'\n  }\n}\n'


if __name__=='__main__':main()
