"""Compare earlier measured-foot support acquisition within the permitted adjustment window."""
import argparse
import json
from pathlib import Path

import torch

from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.aiming_fullbody_io import json_save,save_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','out','acceptance-contract'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,nargs='+',default=[25,75])
    a=p.parse_args();torch.set_num_threads(2)
    assert all(1<=n<=150 for n in a.steps)
    ck=torch.load(a.source,map_location='cpu',weights_only=False);digest=sha256(a.source)
    assert not ck.get('smoke') and ck['stage'] in ('calibration','training')
    assert ck['config']['support_command_mode']=='error' and ck['config']['support_capture_steps']==150
    assert ck['config']['curriculum_level'] in range(1,7)
    validate_checkpoint(ck,WideAimingTask('cpu',ck['config']))
    a.out.mkdir(parents=True,exist_ok=True)
    for number,steps in enumerate(a.steps,1):
        out=a.out/f'ckpt_{number:06d}.pt'
        if out.exists():raise FileExistsError(out)
        cfg=dict(ck['config'],support_capture_steps=steps,initialization_mode='earlier_measured_support_capture_calibration',
            distill_iters=0,wm_only_iters=100,critic_warmup_iters=100)
        task=WideAimingTask('cpu',cfg)
        calibration=dict(method=__doc__,old_capture_steps=150,new_capture_steps=steps,new_capture_seconds=steps*.02,
            all_policy_and_model_weights_exact=True,normalizers_exact=True,initial_pose_and_reset_noise_unchanged=True,
            future_contacts=False,support_anchor='Actual measured feet at acquisition; release on target change or movement',
            acceptance_criteria_changed=False,acceptance_contract_sha256=sha256(a.acceptance_contract),
            physical_acceptance=False,limitation='Controller acquisition timing changed; full physical re-evaluation required. Acceptance still uses the original3s window.')
        artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=task.metadata(),
            robot_metadata=ck['robot_metadata'],pi=ck['pi'],wm=ck['wm'],provenance=ck.get('provenance'),
            source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),calibration=calibration)
        validate_checkpoint(artifact,task);save_checkpoint(out,artifact)
        json_save(out.with_suffix('.json'),dict(checkpoint_sha256=sha256(out),calibration=calibration))
        print(json.dumps(dict(output=str(out),capture_steps=steps)),flush=True)
    assert sha256(a.source)==digest


if __name__=='__main__':main()
