"""Pack robot-specific motion references for the existing render CLI (kinematic preview only)."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic.acp.robots import get_robot,sha256
from supermimic.acp.motion import load_php_npz
from supermimic.acp.terrain import load_terrain


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('motion_dir');ap.add_argument('out');ap.add_argument('--robot',default='tienkung3_29dof');args=ap.parse_args()
    robot=get_robot(args.robot);root=Path(args.motion_dir);files=sorted(root.glob('*/*_motion.npz'))
    if not files:raise ValueError('no motions')
    clips=[load_php_npz(p,robot) for p in files]
    if any(fps!=50 for p,r,q,fps in clips):raise ValueError('preview expects canonical 50 Hz motion')
    lengths=np.array([len(x[0]) for x in clips]);T=int(lengths.max());C=len(clips)
    p=np.zeros((T,C,3));r=np.zeros((T,C,4));q=np.zeros((T,C,robot.num_joints));alive=np.zeros((T,C),bool);f=np.zeros((T,C),int)
    terrains=[load_terrain(str(p))[0] for p in files];K=max(len(x) for x in terrains);boxes=np.zeros((C,K,8))
    start=np.r_[0,np.cumsum(lengths)[:-1]]
    for i,(pp,rr,jj,_) in enumerate(clips):
        idx=np.minimum(np.arange(T),len(pp)-1);p[:,i]=pp[idx];r[:,i]=rr[idx];q[:,i]=jj[idx];alive[:len(pp),i]=True;f[:,i]=start[i]+idx
        boxes[i,:len(terrains[i])]=terrains[i]
    refp=np.concatenate([x[0] for x in clips]);refq=np.concatenate([x[2] for x in clips]);refR=Rotation.from_quat(np.concatenate([x[1] for x in clips])[:,[1,2,3,0]]).as_matrix()
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,robot_metadata=json.dumps(robot.metadata()),purpose='kinematic reference preview, no physics or policy',source_hashes=json.dumps({str(p):sha256(p) for p in files}),names=np.array([str(x.relative_to(root)).replace('_motion.npz','') for x in files]),root_p=p,root_quat=r,q=q,f=f,alive=alive,clip_len=lengths,clip_start=start,boxes=boxes,ref_root_p=refp,ref_root_R=refR,ref_q=refq,tgt_w=np.zeros((T,C,9,3)),tgt_valid=np.zeros((T,C,9),bool))
    print('wrote',args.out)

if __name__=='__main__':main()
