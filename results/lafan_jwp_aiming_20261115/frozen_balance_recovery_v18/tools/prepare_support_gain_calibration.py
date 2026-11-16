"""Fold a scalar feedback calibration into learned support-error input weights."""
import argparse,math
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from supermimic.acp.robots import sha256


def scaled_feedback_state(state,gain,axes='all'):
    if not math.isfinite(gain) or gain<=0:raise ValueError('Positive finite feedback gain required')
    if axes not in ('all','lateral'):raise ValueError('Unknown support feedback axes')
    result={k:v.detach().clone() for k,v in state.items()}
    result['enc.0.weight'][:,:6].mul_(gain)
    if axes=='lateral':
        # Command order is left xyz, right xyz in the current base frame.
        result['enc.0.weight'][:,[0,2,3,5]]=0
    if not all(bool(v.isfinite().all()) for v in result.values()):raise ValueError('Nonfinite scaled policy')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--gain',type=float,required=True)
    p.add_argument('--axes',choices=('all','lateral'),default='all')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    digest=sha256(a.source);ck=torch.load(a.source,map_location='cpu',weights_only=False)
    if (ck.get('smoke') or ck.get('task')!='wide_aiming_v1' or ck.get('stage')!='training'
            or ck['config'].get('curriculum_level')!=0 or ck['config'].get('support_command_mode')!='error'
            or ck['config'].get('policy_training_scope')!='support_error_inputs'):
        raise ValueError('Expected a formal balance-stage support-error feedback checkpoint')
    task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
    parent=Path(ck['source_checkpoint']['path'])
    if sha256(parent)!=ck['source_checkpoint']['sha256']:raise ValueError('Initialization identity changed')
    initial=torch.load(parent,map_location='cpu',weights_only=False)
    for k,v in initial['pi'].items():
        original,current=(v[:,6:],ck['pi'][k][:,6:]) if k=='enc.0.weight' else (v,ck['pi'][k])
        if not torch.equal(original,current):raise ValueError(f'Previously frozen parameter changed: {k}')
    policy=scaled_feedback_state(ck['pi'],a.gain,a.axes)
    cfg=dict(ck['config'],initialization_mode='support_feedback_calibration',distill_iters=0,
             wm_only_iters=100,critic_warmup_iters=100)
    calibration=dict(method='gain and optional axis selection on previously learned support feedback',gain=a.gain,axes=a.axes,
        axis_frame='current base; left xyz then right xyz',
        changed_tensor='enc.0.weight[:, :6]',all_other_policy_tensors_exact=True,
        zero_error_policy_unchanged=True,source_iteration=ck['iteration'],
        optimizer_states_omitted=True,critic='fresh if training starts from this artifact',
        acceptance='Unaccepted candidate; requires exact-checkpoint full physical reports',physical_acceptance=False)
    artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,
        task_config=WideAimingTask('cpu',cfg).metadata(),robot_metadata=ck['robot_metadata'],
        pi=policy,wm=ck['wm'],source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),
        provenance=ck.get('provenance'),calibration=calibration)
    assert sha256(a.source)==digest
    save_checkpoint(a.out,artifact)
    json_save(a.out.with_suffix('.json'),dict(checkpoint_sha256=sha256(a.out),source_sha256=digest,calibration=calibration))
    print(dict(gain=a.gain,axes=a.axes,output=str(a.out),physical_acceptance=False))


if __name__=='__main__':main()
