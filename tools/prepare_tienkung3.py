"""Fetch pinned TienKung 3 assets and build a reproducible 29-hinge, free-root MJCF."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import trimesh
from supermimic.acp.robots import ROOT, TIENKUNG_SHA, CONTACT_NAMES, sha256

LAB_SHA='ffea8bde32fcb632b5ef2a700e452f99e907ea40'
XM_SHA='0385122a232279210e941c7bb28da6a0bbb42356'
CONTROLS={
 'hip_pitch':(300,10,200,12.566),'hip_roll':(300,10,200,12.566),'hip_yaw':(150,5,142,13.823),
 'knee_pitch':(330,10,330,11.106),'ankle_pitch':(30,2.5,100,14.6),'ankle_roll':(16.8,1.4,50,14.6),
 'waist_yaw':(400,5,91,9.2153),'waist_roll':(400,10,91,9.2153),'waist_pitch':(400,10,91,9.2153),
 'shoulder_pitch':(150,5,90,15.184),'shoulder_roll':(50,2.5,60,14.137),'shoulder_yaw':(50,2.5,36,9.739),
 'elbow_pitch':(150,5,60,14.137),'elbow_yaw':(50,5,40,18.325957145940457),
 'wrist_pitch':(20,2,40,18.325957145940457),'wrist_roll':(20,2,40,18.325957145940457)}


def download(url,path):
    if not path.exists():
        tmp=path.with_suffix(path.suffix+'.part')
        subprocess.run(['curl','-fL','--retry','3','--max-time','300',url,'-o',str(tmp)],check=True)
        tmp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',type=Path,default=ROOT/'data/tienkung3_29dof'); args=ap.parse_args()
    root=args.root.resolve(); upstream=root/'upstream'; generated=root/'generated'; manifests=root/'manifests'
    for p in (upstream,generated,manifests): p.mkdir(parents=True,exist_ok=True)
    archive=upstream/'source.tar.gz'
    url=f'https://codeload.github.com/Open-X-Humanoid/TienKung_URDF/tar.gz/{TIENKUNG_SHA}'
    download(url,archive)
    source=upstream/'tiangong3_urdf/urdf/tiangong3.urdf'
    licenses=[]
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            rel=PurePosixPath(*PurePosixPath(member.name).parts[1:])
            if rel.name.lower().startswith(('license','copying')): licenses.append(str(rel))
            if str(rel).startswith('tiangong3_urdf/') or str(rel) in ('README.md','LICENSE'):
                member.name=str(rel)
                tar.extract(member,upstream,filter='data')
    refs={
      'lab_control.py':f'https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/{LAB_SHA}/legged_lab/assets/tiangong3/tiangong3.py',
      'xmimic_control.py':f'https://raw.githubusercontent.com/Open-X-Humanoid/xMimic/{XM_SHA}/source/whole_body_tracking/whole_body_tracking/robots/tiangong3.py'}
    for name,url0 in refs.items(): download(url0,upstream/name)
    original=ET.parse(source); urdf=ET.parse(source); robot=urdf.getroot()
    joints={j.get('name'):j for j in robot.findall('joint')}
    assert sum(j.get('type')=='revolute' for j in joints.values())==31
    locked=['head_yaw_joint','head_pitch_joint']
    for n in locked: joints[n].set('type','fixed')
    mesh_paths=[]
    (generated/"meshes").mkdir(exist_ok=True)
    converted = {}
    for mesh in robot.iter('mesh'):
        name=PurePosixPath(mesh.get('filename')).name
        path=upstream/'tiangong3_urdf/meshes'/name
        if not path.is_file(): raise FileNotFoundError(path)
        mesh_paths.append(path)
        obj=generated/'meshes'/f'{path.stem}.obj'
        if path not in converted:
            # MuJoCo STL decoder caps at 200k faces. OBJ preserves every triangle.
            mesh_data=trimesh.load_mesh(path,process=False)
            mesh_data.export(obj,file_type='obj',digits=10)
            converted[path]=obj
            mesh_paths.append(obj)
        mesh.set('filename',os.path.relpath(obj,generated))
    mj=robot.find('mujoco')
    if mj is not None: robot.remove(mj)
    mj=ET.SubElement(robot,'mujoco')
    ET.SubElement(mj,'compiler',meshdir='.',discardvisual='false',fusestatic='false',strippath='false',balanceinertia='false')
    derived=generated/'robot.urdf'; urdf.write(derived,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(derived))
    out=generated/'robot.xml'; mujoco.mj_saveLastXML(str(out),model)
    tree=ET.parse(out); wb=tree.getroot().find('worldbody'); pelvis=wb.find("body[@name='pelvis']")
    if pelvis is None: raise ValueError('URDF root pelvis missing after conversion')
    ET.SubElement(pelvis,'freejoint',name='floating_base_joint')
    # Explicit inertial data only: visuals/collisions must never change mass.
    tree.write(out,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(out))
    leg=['hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_pitch','ankle_roll']
    arm=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','elbow_yaw','wrist_pitch','wrist_roll']
    names=[f'{n}_{s}_joint' for s in ('l','r') for n in leg]+[f'waist_{a}_joint' for a in ('yaw','roll','pitch')]+[f'{n}_{s}_joint' for s in ('l','r') for n in arm]
    tracked=['pelvis','hip_roll_l_link','knee_pitch_l_link','ankle_roll_l_link','hip_roll_r_link','knee_pitch_r_link','ankle_roll_r_link','waist_pitch_link','shoulder_roll_l_link','elbow_pitch_l_link','left_tcp_link','shoulder_roll_r_link','elbow_pitch_r_link','right_tcp_link']
    frames=['ankle_roll_l_link','ankle_roll_r_link','knee_pitch_l_link','knee_pitch_r_link','left_tcp_link','right_tcp_link','elbow_pitch_l_link','elbow_pitch_r_link','pelvis']
    cfg=dict(robot_id='tienkung3_29dof',joint_names=names,tracked_bodies=tracked,contact_bodies={n:(f,(f,)) for n,f in zip(CONTACT_NAMES,frames)},
             torso_body='waist_pitch_link',root_body='pelvis',control_dt=.02,decimation=4,locked_joints=locked,
             geometry_version='convex-surface-8mm-v1',armature=[0.]*29,action_scales=[.25]*29)
    rows=[]
    for n in names:
        key=n.removesuffix('_joint').removesuffix('_l').removesuffix('_r')
        k,d,e,v=CONTROLS[key]; lim=joints[n].find('limit')
        rows.append(dict(joint=n,kp=k,kd=d,reference_effort=e,urdf_effort=float(lim.get('effort')),effort=min(e,float(lim.get('effort'))),
                         reference_velocity=v,urdf_velocity=float(lim.get('velocity')),velocity=min(v,float(lim.get('velocity'))),
                         source='xmimic_control.py' if key in ('elbow_yaw','wrist_pitch','wrist_roll') else 'lab_control.py'))
    for k in ('kp','kd','effort','velocity'): cfg[k]=[r[k] for r in rows]
    defaults={'hip_pitch':-.15,'knee_pitch':.30,'ankle_pitch':-.15,'shoulder_pitch':.20,'elbow_pitch':-.50}
    cfg['default_q']=[defaults.get(n.removesuffix('_joint').removesuffix('_l').removesuffix('_r'),0.) for n in names]
    cfg['default_q'][names.index('shoulder_roll_l_joint')]=.10; cfg['default_q'][names.index('shoulder_roll_r_joint')]=-.10
    spec=mujoco.MjSpec.from_file(str(out)); spec.worldbody.add_geom(name='height_plane',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,1))
    hm=spec.compile(); hd=mujoco.MjData(hm)
    adr=[hm.jnt_qposadr[hm.joint(n).id] for n in names]; hd.qpos[adr]=cfg['default_q']; hd.qpos[:3]=[0,0,2]
    mujoco.mj_forward(hm,hd)
    foot_ids=[hm.body(f).id for f in frames[:2]]
    feet=[g for g in range(hm.ngeom) if hm.geom_bodyid[g] in foot_ids and (hm.geom_contype[g] or hm.geom_conaffinity[g])]
    z=[mujoco.mj_geomDistance(hm,hd,g,hm.geom('height_plane').id,10,None) for g in feet]
    cfg['stand_height']=float(2-min(z))
    mass=sum(float(x.get('value')) for x in original.findall('./link/inertial/mass'))
    assert (model.nq,model.nv,model.nu)==(36,35,0)
    assert abs(model.body_mass.sum()-mass)<1e-4
    assert np.all(model.body_inertia[model.body_mass>0]>0)
    assert set(names)=={model.joint(j).name for j in range(model.njnt) if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE}
    (manifests/'robot.json').write_text(json.dumps(cfg,indent=2)+'\n')
    paths=sorted(set(mesh_paths+[source,derived,out,manifests/'robot.json',upstream/'lab_control.py',upstream/'xmimic_control.py']))
    manifest=dict(upstream_sha=TIENKUNG_SHA,url=url,archive_sha256=sha256(archive),converter='tools/prepare_tienkung3.py:v1',
                  mujoco=mujoco.__version__,trimesh=trimesh.__version__,mesh_conversion='lossless STL to OBJ (no decimation)',licenses_found=licenses,license_note='No LICENSE/COPYING file in pinned source archive; preserve upstream provenance, do not invent a license.',
                  control_sources=refs,control_parameters=rows,armature_note='zero: unidentified simulation assumption',self_collision=False,
                  locked_joints=locked,canonical_joint_order=names,source_mass=mass,generated_mass=float(model.body_mass.sum()),
                  nq=model.nq,nv=model.nv,nu=model.nu,stand_height=cfg['stand_height'],files={str(p.relative_to(root)):sha256(p) for p in paths})
    (manifests/'assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('nq','nv','nu','source_mass','generated_mass','stand_height')},indent=2))


if __name__=='__main__': main()
