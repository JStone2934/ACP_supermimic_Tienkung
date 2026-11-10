import json,pathlib,subprocess,sys,os,concurrent.futures
from tools.convert_php import verify_task
from supermimic.acp.robots import sha256
root=pathlib.Path.cwd();out=root/'results/php_success_previews_v3_20261109';out.mkdir(parents=True,exist_ok=True)
old=root/'data/tienkung3_29dof/motions/php_full_v2_20261109';new=root/'data/tienkung3_29dof/motions/php_repair_v3_20261109'
selections=[('01_straight','Flat ground - forward',new,'motions-locomotion/cfg0006_stand_2.0s_run_forward_1.4s_stand_2.0s'),('02_turn','Flat ground - right turn',new,'motions-locomotion/cfg0012_stand_2.0s_run_right_90_1.2s_stand_2.0s'),('03_low_step','Low step - forward',old,'motions-low-step/forward_walk1.2s_step0s'),('04_high_step','High step - forward',new,'motions-high-step/run_forward_0.9s_step1.0s')]
rows=[]
for name,title,parent,key in selections:
 plan=json.loads((parent/'plan.json').read_text());task=next(t for t in plan['tasks'] if t['record']['key']==key);state=json.loads((parent/'tasks'/(task['id']+'.json')).read_text());verify_task(parent,state)
 quality=json.loads((parent/state['quality_path']).read_text());assert quality['accepted_variants']==list(range(10))
 category=key.split('/')[0];dest=out/'inputs'/category;dest.mkdir(parents=True,exist_ok=True)
 for suffix in ['_motion.npz','_terrain.npy']:
  source=parent/'accepted'/(key+suffix);link=dest/(name+suffix)
  if not link.exists():link.symlink_to(source)
 rows.append(dict(name=name,title=title,source_key=key,parent=str(parent),quality_path=str(parent/state['quality_path']),source_variant=state['accepted_variants'][0],export_slot=0,geometry_pass=True,frames=task['record']['frames'],fps=50,motion_sha256=sha256(parent/'accepted'/(key+'_motion.npz')),clip=category+'/'+name,video=str(out/(name+'.mp4'))))
traj=out/'reference.npz';subprocess.run([sys.executable,'-m','tools.record_motion_reference',str(out/'inputs'),str(traj)],check=True)
def render(row):
 cmd=[sys.executable,'-m','supermimic.acp.render',str(traj),row['video'],'--clips',row['clip'],'--tail','0','--width','1280','--height','720','--title',row['title'],'--solid-label','converted reference (kinematic)']
 with (out/(row['name']+'.log')).open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
 row['video_sha256']=sha256(row['video']);print('FINISHED',row['video'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(render,rows))
(out/'manifest.json').write_text(json.dumps(dict(purpose='Geometrically accepted kinematic references; no dynamics or policy playback',clips=rows),indent=2)+'\n')
print('ALL DONE',out,flush=True)
