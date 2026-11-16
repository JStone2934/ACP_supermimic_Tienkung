"""Bounded exploration-noise diagnostic; never an acceptance report."""
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from tools.evaluate_wide_aiming import evaluate


def main():
    torch.set_num_threads(2)
    root=Path.cwd();b=root/'results/lafan_jwp_aiming_20261115'
    source=root/'logs/tienkung3_29dof/lafan_jwp_20261115/narrow_support/train/ckpt_000400.pt'
    out=b/'diagnostic_training_noise400';out.mkdir(exist_ok=True)
    path=out/'seed_53.json'
    if path.exists():raise FileExistsError(path)
    ck=torch.load(source,map_location='cuda:0',weights_only=False)
    task=WideAimingTask('cuda:0',ck['config']);validate_checkpoint(ck,task)
    pi=FullbodyPolicy(task.robot).to('cuda:0');pi.load_state_dict(ck['pi'])
    noise=task.alpha.new_full((29,),ck['config']['body_noise_rad']);noise[task.arm_ids]=ck['config']['arm_noise_rad']
    fraction=max(ck['config']['noise_min_fraction'],1.-400/ck['config']['noise_decay_updates'])
    generator=torch.Generator(device='cuda:0').manual_seed(1653)
    class NoisyPolicy(torch.nn.Module):
        def __init__(self):super().__init__();self.policy=pi
        def forward(self,history,command):
            raw=self.policy(history,command)
            return raw+torch.randn(raw.shape,device=raw.device,generator=generator)*noise/task.alpha*fraction
    temporary=out/'diagnostic_pending.json'
    report=evaluate(NoisyPolicy(),task,temporary,seed=53,checkpoint=source,stage=2)
    baseline=b/'evaluation_narrow_support/iteration_000400/seed_53.npz'
    with np.load(baseline) as x,np.load(temporary.with_suffix('.npz')) as y:
        for key in ('root','root_quat','q','targets'):
            np.testing.assert_array_equal(x[key][0],y[key][0])
    report['diagnostic_metrics_passed']=report.pop('passed');report['passed']=False
    report.update(scope=__doc__,physical_acceptance=False,initial_states_and_targets_exact=True,
        exploration_fraction=fraction,body_noise_rad=ck['config']['body_noise_rad'],arm_noise_rad=ck['config']['arm_noise_rad'],noise_seed=1653)
    path.write_text(json.dumps(report,indent=2));temporary.with_suffix('.npz').rename(path.with_suffix('.npz'));temporary.unlink()
    print(json.dumps(dict(scope=__doc__,random_success_rate=report['random_success_rate'],
        failures=[i for i,s in enumerate(report['scenarios']) if not s['passed']])),flush=True)


if __name__=='__main__':main()
