"""Add a zero-initialized, 45-parameter distal target readout without changing actions."""
import argparse
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_io import save_checkpoint, json_save
from supermimic.acp.robots import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    digest=sha256(a.source);ck=torch.load(a.source,map_location='cpu',weights_only=False)
    if ck.get('smoke') or ck.get('stage') not in ('training','calibration') or ck.get('task')!='wide_aiming_v1':
        raise ValueError('Identified formal standing-aiming source required')
    if ck['config'].get('distal_command_skip'):raise ValueError('Source already has a direct target readout')
    task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
    policy=FullbodyPolicy(task.robot);policy.load_state_dict(ck['pi']);policy.eval()
    torch.manual_seed(827);history=torch.randn(16,10,99);command=torch.randn(16,107)
    with torch.no_grad():before=policy(history,command)
    policy.enable_distal_command_skip()
    with torch.no_grad():torch.testing.assert_close(policy(history,command),before,atol=0,rtol=0)
    state=policy.state_dict()
    assert all(torch.equal(value,state[key]) for key,value in ck['pi'].items())
    cfg=dict(ck['config'],distal_command_skip=True,policy_training_scope='right_distal_with_command_skip',
             initialization_mode='zero_distal_command_skip',distill_iters=0,wm_only_iters=100,critic_warmup_iters=100)
    cfg.pop('support_actor_lr',None)
    task_new=WideAimingTask('cpu',cfg)
    record=dict(method=__doc__,old_weights_and_normalizers_exact=True,initial_actions_exact=True,
                new_parameters=45,runtime_motion_files=False,online_ik=False,physical_acceptance=False)
    result=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=task_new.metadata(),
                robot_metadata=ck['robot_metadata'],pi=state,wm=ck['wm'],provenance=ck.get('provenance'),
                source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),calibration=record)
    validate_checkpoint(result,task_new);assert sha256(a.source)==digest
    save_checkpoint(a.out,result);json_save(a.out.with_suffix('.json'),dict(record,checkpoint_sha256=sha256(a.out),source_sha256=digest))
    print(a.out)


if __name__=='__main__':main()
