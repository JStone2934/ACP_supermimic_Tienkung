"""Read-only census of reference contact labels and evaluation termination timing."""
import json
from pathlib import Path
import numpy as np
from supermimic.acp.task import debounce, drop_short, next_onsets

ROOT = Path.cwd()
OUT = ROOT/'results/stage_training_analysis_20261111'
DATA = ROOT/'data/tienkung3_29dof/stage1/php_all_variants_20261110'
EVAL = json.loads((ROOT/'logs/tienkung3_29dof/stage1_php_all_spheres224_20261110/stage1_final_eval.json').read_text())
body_names = ['left_foot','right_foot','left_knee','right_knee','left_hand','right_hand','left_elbow','right_elbow','pelvis']
summary = {}
for group in sorted(EVAL['per_folder']):
    counts = {key: np.zeros(9, dtype=np.int64) for key in
              ['raw_frames','filtered_frames','onsets','target_frames','inside_scan_target_frames']}
    frames = 0
    details = []
    for path in sorted((DATA/'contacts'/group).glob('*.npz')):
        name = group+'/'+path.stem
        with np.load(path) as d:
            raw = d['contact_info'][0,...,0] > .5
            points = d['contact_pos_w'][0]
        filtered = drop_short(debounce(raw, 2), 10)
        on = next_onsets(filtered, 75)
        valid = on >= 0
        with np.load(DATA/'motions'/(name+'_motion.npz')) as d:
            jp = d['joint_pos']
        xyz = points[on.clip(min=0),np.arange(9)[None,:]] - jp[:,None,:3]
        w,x,y,z = jp[:,3:7].T
        yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        local_x = np.cos(yaw)[:,None]*xyz[...,0]+np.sin(yaw)[:,None]*xyz[...,1]
        local_y = np.cos(yaw)[:,None]*xyz[...,1]-np.sin(yaw)[:,None]*xyz[...,0]
        inside = (np.abs(local_x)<=.800001)&(np.abs(local_y)<=.500001)&valid
        counts['raw_frames'] += raw.sum(0)
        counts['filtered_frames'] += filtered.sum(0)
        counts['onsets'] += (filtered & ~np.vstack([np.zeros((1,9),bool),filtered[:-1]])).sum(0)
        counts['target_frames'] += valid.sum(0)
        counts['inside_scan_target_frames'] += inside.sum(0)
        frames += len(raw)
        alive = EVAL['per_clip'][name]['alive_steps']
        hand_frames = np.flatnonzero(filtered[:,4:6].any(-1))
        details.append(dict(name=name,frames=len(raw),alive_steps=alive,
                            alive_fraction=alive/(len(raw)-1),
                            first_hand_step=int(hand_frames[0]) if len(hand_frames) else None,
                            ended_before_first_hand=bool(alive < hand_frames[0]) if len(hand_frames) else None))
    summary[group] = dict(frames=frames,clips=len(details),
                         median_alive_fraction=float(np.median([r['alive_fraction'] for r in details])),
                         n_with_hand=sum(r['first_hand_step'] is not None for r in details),
                         n_ended_before_first_hand=sum(r['ended_before_first_hand'] is True for r in details),
                         **{k:v.tolist() for k,v in counts.items()},details=details)
    print(group,{k:v for k,v in summary[group].items() if k!='details'},flush=True)
(OUT/'reference_label_census.json').write_text(json.dumps(dict(body_names=body_names,groups=summary),indent=2))
