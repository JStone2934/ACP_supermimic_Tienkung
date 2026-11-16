"""Side-by-side BVH/native-robot skeletons at matching physical times."""
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.jwp_pipeline import configure,load_human


def compare(dataset,clip,output,acp_root):
    configure(acp_root,ROOT/".cache/GMR")
    from supermimic.acp.robots import get_robot
    from supermimic.acp.motion import load_php_npz
    human,_,_=load_human(ROOT.parent/"ACP_supermimic_Tienkung/data/JWP/bvh"/(clip+".bvh"))
    robot=get_robot("tienkung3_29dof")
    p,quat,q,fps=load_php_npz(dataset/"motions/jwp"/(clip+"_motion.npz"),robot)
    model=mujoco.MjModel.from_xml_path(robot.mjcf_path);data=mujoco.MjData(model)
    qa,_=robot.joint_addresses(model)
    source_edges=[("Hips","Spine"),("Spine","Spine1"),("Spine1","Spine2"),("Spine2","Neck"),("Neck","Neck1"),("Neck1","Head")]
    robot_edges=[("pelvis","waist_yaw_link"),("waist_yaw_link","waist_roll_link"),("waist_roll_link","waist_pitch_link"),("waist_pitch_link","head_yaw_link")]
    for side,s in (("Left","l"),("Right","r")):
        source_edges += [("Hips",side+"UpLeg"),(side+"UpLeg",side+"Leg"),(side+"Leg",side+"Foot"),
                         ("Spine2",side+"Shoulder"),(side+"Shoulder",side+"Arm"),(side+"Arm",side+"ForeArm"),(side+"ForeArm",side+"Hand")]
        robot_edges += [("pelvis",f"hip_pitch_{s}_link"),(f"hip_pitch_{s}_link",f"hip_yaw_{s}_link"),
                        (f"hip_yaw_{s}_link",f"knee_pitch_{s}_link"),(f"knee_pitch_{s}_link",f"ankle_roll_{s}_link"),
                        ("waist_pitch_link",f"shoulder_roll_{s}_link"),(f"shoulder_roll_{s}_link",f"shoulder_yaw_{s}_link"),
                        (f"shoulder_yaw_{s}_link",f"elbow_pitch_{s}_link"),(f"elbow_pitch_{s}_link",f"wrist_roll_{s}_link")]
    fig=plt.figure(figsize=(10,12),layout="constrained")
    for row,t in enumerate((0,len(p)//2,len(p)-1)):
        data.qpos[:3],data.qpos[3:7],data.qpos[qa]=p[t],quat[t],q[t]
        mujoco.mj_kinematics(model,data)
        hpoints={n:x[0] for n,x in human[t].items()}
        rpoints={model.body(i).name:data.xpos[i].copy() for i in range(1,model.nbody)}
        for col,(points,edges,root,title) in enumerate(((hpoints,source_edges,"Hips","JWP BVH (source)"),(rpoints,robot_edges,"pelvis","TienKung 29 DoF (converted)"))):
            ax=fig.add_subplot(3,2,row*2+col+1,projection="3d")
            origin=points[root]*np.array([1,1,0])
            for a,b in edges:
                if a not in points or b not in points:continue
                xy=np.array([points[a],points[b]])-origin
                color="#2171b5" if "Left" in b or "_l_" in b else "#d95f0e" if "Right" in b or "_r_" in b else "#333333"
                ax.plot(*xy.T,color=color,lw=2.5,marker="o",markersize=3)
            ax.set(xlim=(-.75,.75),ylim=(-.75,.75),zlim=(0,1.9),xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)",title=f"{title} | {t/fps:.2f} s")
            ax.set_box_aspect((1.5,1.5,1.9));ax.view_init(elev=15,azim=130)
    fig.suptitle(f"{clip} — same timestamps and world axes; blue=left, orange=right\nXY roots centered independently; source and robot have different proportions")
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=150);plt.close(fig)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",type=Path,required=True);ap.add_argument("--clip",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--acp-root",type=Path,default=ROOT.parent/"standing-aiming")
    args=ap.parse_args();compare(args.dataset,args.clip,args.output,args.acp_root)
