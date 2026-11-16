"""Verify joint right-arm and support-input updates with all other parameters frozen."""
import argparse,json
from pathlib import Path
import torch
from supermimic.acp.robots import get_robot,sha256
from supermimic.acp.standing import ARM_NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    source=Path(ck['source_checkpoint']['path']);digest=sha256(source)
    assert digest==ck['source_checkpoint']['sha256']
    initial=torch.load(source,map_location='cpu',weights_only=False)
    assert bool(ck['smoke'])==a.smoke and not initial.get('smoke')
    assert ck['config']['policy_training_scope']=='support_and_right_arm_outputs' and ck['config']['curriculum_level'] in range(1,7)
    r=get_robot('tienkung3_29dof');r.validate_metadata(ck['robot_metadata'])
    arm=[r.joint_names.index(n) for n in ARM_NAMES];body=[i for i in range(29) if i not in arm]
    unchanged={};deltas={}
    for key,value in initial['pi'].items():
        if key=='enc.0.weight':
            unchanged[key]=torch.equal(value[:,6:],ck['pi'][key][:,6:])
            deltas[key]=float((ck['pi'][key][:,:6]-value[:,:6]).norm())
        elif key in ('dec.8.weight','dec.8.bias'):
            unchanged[key]=torch.equal(value[body],ck['pi'][key][body])
            deltas[key]=float((ck['pi'][key][arm]-value[arm]).norm())
        else:unchanged[key]=torch.equal(value,ck['pi'][key])
    assert all(unchanged.values()),[k for k,v in unchanged.items() if not v]
    d=ck['distributed'];assert d['world_size']==2 and not d['skipped'] and d['state_sha256']
    assert {rank['device'] for rank in d['ranks']}=={'cuda:0','cuda:1'}
    assert bool(deltas['enc.0.weight']>0)==bool(d['updates'].get('pi',0))
    assert bool(deltas['dec.8.weight']+deltas['dec.8.bias']>0)==bool(d['updates'].get('pi',0))
    rates=[group['lr'] for group in ck['optimizers'][0]['param_groups']]
    assert rates==[ck['config']['support_actor_lr'],ck['config']['actor_lr']]
    finite={name:all(bool(v.isfinite().all()) for v in ck[name].values() if torch.is_tensor(v)) for name in ('pi','wm','critic')}
    assert all(finite.values())
    if a.smoke:assert ck['iteration']==2 and all(d['updates'][name]==2 for name in ('wm','pi','critic'))
    report=dict(passed=True,scope='Artifact and two-GPU training integrity; not physical policy acceptance',
        checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha256(a.checkpoint),source_sha256=digest,
        smoke=a.smoke,formal_initialization_allowed=not a.smoke,iteration=ck['iteration'],
        all_non_support_input_and_non_arm_output_weights_unchanged=True,normalizers_unchanged=True,adaptation_deltas=deltas,optimizer_learning_rates=rates,
        all_weights_finite=finite,updates=d['updates'],world_size=d['world_size'],state_sha256=d['state_sha256'],
        model_ready=ck['model_ready'],physical_acceptance=False)
    a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
