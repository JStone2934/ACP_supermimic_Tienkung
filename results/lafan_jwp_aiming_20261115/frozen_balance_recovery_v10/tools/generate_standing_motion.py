"""Generate a clearly identified synthetic standing fixture, not a trained motion dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
from supermimic.acp.robots import get_robot, sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--robot',default='tienkung3_29dof',choices=['g1','tienkung3_29dof'])
    ap.add_argument('--out'); ap.add_argument('--seconds',type=float,default=12.)
    ap.add_argument('--pose', choices=['default', 'standing_aiming'], default='default')
    args=ap.parse_args(); robot=get_robot(args.robot)
    if args.seconds < .1: raise ValueError('need at least .1 seconds')
    root=Path(args.out or robot.motion_dir); folder=root/'synthetic'; folder.mkdir(parents=True,exist_ok=True)
    n=round(args.seconds/robot.control_dt)+1
    jp=np.zeros((n,7+robot.num_joints)); jp[:,2]=robot.stand_height; jp[:,3]=1.; jp[:,7:]=robot.default_q
    if args.pose == 'standing_aiming':
        from supermimic.acp.standing import held_pose
        jp[:,7:] = held_pose(robot)
    motion=folder/'standing_motion.npz'; terrain=folder/'standing_terrain.npy'
    np.savez_compressed(motion,joint_pos=jp,joint_names=np.array(robot.joint_names),fps=np.array([1/robot.control_dt]),
                        robot_metadata=json.dumps(robot.metadata()),source='synthetic constant default pose',data_version='standing-fixture-v1',
                        purpose='static task reference; not a demonstrated controller', pose=args.pose)
    np.save(terrain,np.zeros((0,1,10)))
    manifest=dict(robot_metadata=robot.metadata(),frames=n,fps=1/robot.control_dt,source='synthetic constant pose',pose=args.pose,
                  purpose='not evidence of standing, locomotion or generalization ability',
                  files={str(p.relative_to(root)):sha256(p) for p in (motion,terrain)})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'frames':n,'height':robot.stand_height,'out':str(root)},indent=2))

if __name__=='__main__': main()
