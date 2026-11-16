"""Compare two explicit, target-independent ready poses observed from this policy."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.aiming_fullbody_io import json_save, save_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'trajectory', 'out', 'acceptance-contract'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--allow-settled-center', action='store_true', help='Allow a standing center rollout whose final15s meet aim and foot stability; does not waive evaluation criteria')
    a = p.parse_args(); torch.set_num_threads(2)
    ck = torch.load(a.source, map_location='cpu', weights_only=False)
    digest = sha256(a.source)
    assert ck['stage'] in ('training','calibration') and not ck.get('smoke')
    report = json.loads(a.trajectory.with_suffix('.json').read_text())
    assert report['checkpoint_sha256'] == digest and report['continuous_without_reset']
    assert report['seconds'] == 30 and report['level'] == ck['config']['curriculum_level']
    params = np.array(report['target_parameters'][0])
    # Always use the single fixed, central 3 m scenario at 15 s; never select a favorable random case.
    candidates = np.where(np.isclose(params[:report['fixed_count'],0],3.) &
                          np.isclose(params[:report['fixed_count'],1],0.) &
                          np.isclose(params[:report['fixed_count'],2],0.))[0]
    assert len(candidates)==1
    index=int(candidates[0]); frame=750
    if not a.allow_settled_center:
        assert report['scenarios'][index]['passed']
    else:
        assert report['scenarios'][index]['standing_pass']
    task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
    with np.load(a.trajectory) as z:
        assert str(z['checkpoint_sha256'])==digest
        if a.allow_settled_center:
            error=z['error'][frame:,index];feet=z['feet'][frame:,index]
            assert np.isfinite(error).all() and (error <= report['hit_radius_m']).mean() >= .95
            assert np.linalg.norm(feet-feet[0],axis=-1).max() <= .05
        q=z['q'][frame,index].copy();quat=z['root_quat'][frame,index].copy();height=float(z['root'][frame,index,2])
    assert np.isfinite(q).all() and np.isfinite(quat).all() and np.isfinite(height)
    a.out.mkdir(parents=True,exist_ok=True)
    for number,mode in enumerate(('right_arm_only','full_observed_pose'),1):
        out=a.out/f'ckpt_{number:06d}.pt'
        if out.exists():raise FileExistsError(out)
        cfg=dict(ck['config'],initialization_mode='observed_ready_pose_calibration',distill_iters=0,wm_only_iters=100,critic_warmup_iters=100)
        if mode=='right_arm_only':
            initial=np.array(cfg['initial_q']).copy();initial[task.arm_ids.numpy()]=q[task.arm_ids.numpy()]
            cfg['initial_q']=initial.tolist()
        else:
            cfg.update(initial_q=q.tolist(),initial_height=height,initial_quat=quat.tolist())
        for key in ('reset_position_noise_m','reset_rotation_noise_rad','reset_joint_noise_rad'):
            assert cfg.get(key)==ck['config'].get(key)
        new_task=WideAimingTask('cpu',cfg)
        calibration=dict(method=__doc__,mode=mode,trajectory=str(a.trajectory.resolve()),trajectory_sha256=sha256(a.trajectory),
            scenario=index,frame=frame,source_center_full_pass=report['scenarios'][index]['passed'],settled_center_only=a.allow_settled_center,selection='Fixed center 3m at15s; identical reset pose for every requested target and random seed',
            all_policy_and_model_weights_exact=True,all_normalizers_exact=True,reset_noise_unchanged=True,
            initial_pose_changed=True,old_initial_pose={k:ck['config'][k] for k in ('initial_q','initial_height','initial_quat')},
            acceptance_contract_sha256=sha256(a.acceptance_contract),acceptance_criteria_changed=False,
            runtime_motion_files=False,physical_acceptance=False,
            limitation='A policy operating from a ready stance, not evidence for arbitrary-pose recovery. Must be evaluated afresh with the full original target/noise protocol.')
        artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=new_task.metadata(),
            robot_metadata=ck['robot_metadata'],pi=ck['pi'],wm=ck['wm'],provenance=ck.get('provenance'),
            source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),calibration=calibration)
        validate_checkpoint(artifact,new_task);save_checkpoint(out,artifact)
        json_save(out.with_suffix('.json'),dict(checkpoint_sha256=sha256(out),calibration=calibration))
        print(json.dumps(dict(output=str(out),mode=mode)),flush=True)
    assert sha256(a.source)==digest


if __name__=='__main__':main()
