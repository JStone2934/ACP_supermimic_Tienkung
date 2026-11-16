"""Export an accepted inference-only policy with matching evaluation and video evidence."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from supermimic.acp.robots import sha256
from supermimic.acp.aiming_policy import AimingPolicy
from supermimic.acp.chest_posture import verify_posture_reports
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from tools.prepare_moving_aiming_init import verify_standing_acceptance


def verify_acceptance(checkpoint,reports,moving):
    if not moving:return verify_standing_acceptance(checkpoint,reports)
    digest=sha256(checkpoint)
    if len(reports)!=2 or len({r.get('seed') for r in reports})!=2:raise ValueError('Two independent full evaluations required')
    for r in reports:
        if not (r.get('checkpoint_sha256')==digest and r.get('scope')=='Full moving aiming acceptance'
                and r.get('level') is None and r.get('passed') is True and r.get('fixed_grid_all_pass') is True
                and r.get('fixed_count')==25 and r.get('random_count')==100 and r.get('random_success_rate',0)>=.95
                and r.get('seconds')==30 and r.get('continuous_without_reset') is True and r.get('reference_free') is True
                and r.get('online_target_switches') is True and r.get('hit_radius_m')==.05
                and r.get('maximum_speed_m_s')==.3 and r.get('velocity_rmse_limit_m_s')==.08):
            raise ValueError('Missing full moving acceptance evidence')
    return digest


def verify_video(video,report_paths,reports,digest):
    meta=json.loads(video.with_suffix('.json').read_text())
    trajectory=Path(meta['trajectory'])
    if (meta.get('checkpoint_sha256')!=digest or meta.get('seconds')!=30. or meta.get('hit_radius_m')!=.05
            or meta.get('video_sha256')!=sha256(video) or meta.get('trajectory_sha256')!=sha256(trajectory)):
        raise ValueError('Video does not match accepted checkpoint and full physical rollout')
    for path,report in zip(report_paths,reports):
        if trajectory.resolve()==path.with_suffix('.npz').resolve():
            index=meta['scenario']
            if not isinstance(index,int) or not 0<=index<len(report['scenarios']) or report['scenarios'][index].get('passed') is not True:
                raise ValueError('Selected video scenario did not pass')
            return meta
    raise ValueError('Video is not from either independent acceptance rollout')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reports',type=Path,nargs=2,required=True);p.add_argument('--videos',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError('Use a new immutable delivery directory')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck.get('task') not in ('wide_aiming_v1','moving_aiming_v1') or ck.get('smoke'):raise ValueError('Expected formal aiming weights')
    moving=ck['task']=='moving_aiming_v1';reports=[json.loads(x.read_text()) for x in a.reports]
    verify_posture_reports(reports)
    digest=verify_acceptance(a.checkpoint,reports,moving)
    videos=[verify_video(x,a.reports,reports,digest) for x in a.videos]
    # Retain only fields that define the actual inference task; remove training
    # paths, teacher dependencies, replay settings and optimizer hyperparameters.
    keys=('initial_q','initial_height','initial_quat','physics_substeps','switch_interval_steps',
          'reward_weights','chest_posture','feet_position_scale_m','term_height','term_tilt_deg',
          'soft_termination_mode','soft_termination_tilt_margin_deg','soft_termination_height_margin_m',
          'reset_position_noise_m','reset_rotation_noise_rad','reset_joint_noise_rad','support_capture_steps','support_command_mode','distal_command_skip','distal_kinematic_head')
    config={k:ck['config'][k] for k in keys if k in ck['config']}
    artifact=dict(task=ck['task'],stage='accepted_inference',config=config,task_config=ck['task_config'],
        robot_metadata=ck['robot_metadata'],pi=ck['pi'],source_checkpoint_sha256=digest,
        acceptance_report_sha256=[sha256(x) for x in a.reports])
    a.out.mkdir(parents=True);save_checkpoint(a.out/'policy.pt',artifact)
    loaded=AimingPolicy(a.out/'policy.pt')
    if any(not torch.equal(v,loaded.policy.state_dict()[k]) for k,v in ck['pi'].items()):
        raise ValueError('Export changed policy tensors or normalization buffers')
    json_save(a.out/'config.json',config);json_save(a.out/'robot_metadata.json',ck['robot_metadata'])
    for path,report in zip(a.reports,reports):shutil.copyfile(path,a.out/f"evaluation_seed_{report['seed']}.json")
    for index,(path,meta) in enumerate(zip(a.videos,videos)):
        shutil.copyfile(path,a.out/f'rollout_{index}.mp4');json_save(a.out/f'rollout_{index}.json',meta)
    runtime_files=list(Path('supermimic').rglob('*.py'))
    manifest=dict(status='accepted',task=ck['task'],source_checkpoint_sha256=digest,
        inference_checkpoint_sha256=sha256(a.out/'policy.pt'),weights_and_normalizers_exactly_preserved=True,
        runtime_entry='supermimic.acp.aiming_policy:AimingPolicy',control_dt=.02,scope='Simulation only',
        state_layout=['world_position','body_to_world_rotation_matrix','world_linear_velocity','world_angular_velocity',
                      'native29_joint_positions','native29_joint_velocities'],
        action='29 canonical PD joint position targets, radians',reference_motion_files_required=False,
        robot_metadata=ck['robot_metadata'],runtime_source_sha256={str(x):sha256(x) for x in runtime_files},
        artifacts={x.name:sha256(x) for x in a.out.iterdir() if x.is_file()},
        inference_call_sequence=['reset(state, spherical_parameters)','set_target(world_position,world_normal) on explicit changes',
                                 'set_velocity(world_velocity) for moving policy','step(state) every20ms'])
    if sha256(a.checkpoint)!=digest:raise ValueError('Source changed during export')
    json_save(a.out/'manifest.json',manifest);print(json.dumps(dict(status='accepted',output=str(a.out.resolve()))))


if __name__=='__main__':main()
