"""Render six examples from this directory's newly recorded Stage2 evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
evaluation = json.loads((OUT / 'record_eval.json').read_text())
selection = json.loads((OUT / 'selection.json').read_text())
specs = [
    ('01_success_locomotion', 'motions-locomotion', False, 'SUCCESS | Locomotion'),
    ('02_success_low_step', 'motions-low-step', False, 'SUCCESS | Low step'),
    ('03_success_high_step', 'motions-high-step', False, 'SUCCESS | High step'),
    ('04_failure_high_step', 'motions-high-step', True, 'FAILURE | High step'),
    ('05_failure_high_climb', 'motions-high-climb-76', True, 'FAILURE | High climb'),
    ('06_failure_low_climb', 'motions-low-climb-76', True, 'FAILURE | Low climb'),
]
videos = []
for stem, group, failed, title in specs:
    candidates = [(n, m) for n, m in evaluation['per_clip'].items()
                  if n.split('/')[0] == group and m['failed'] == failed]
    if not candidates:
        raise RuntimeError(f'No recorded example for {stem}; do not reuse an old label')
    candidates.sort(key=lambda item: item[1]['root_rmse_m'], reverse=failed)
    name, metrics = candidates[0]
    video = OUT / (stem + '.mp4')
    command = [sys.executable, '-m', 'supermimic.acp.render',
               str(OUT / 'trajectories.npz'), str(video), '--clips', name,
               '--width', '1280', '--height', '720', '--fps', '50', '--tail', '1.5',
               '--title', title + ' | Stage2 40k', '--solid-label', 'Stage2 policy']
    print('Rendering', stem, name, metrics, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration,size:stream=codec_name,width,height,nb_frames,r_frame_rate',
        '-of', 'json', str(video)]))
    videos.append(dict(file=video.name, name=name, metrics=metrics, probe=probe,
                       sha256=hashlib.sha256(video.read_bytes()).hexdigest()))
    (OUT / 'videos.json').write_text(json.dumps(dict(protocol=selection, videos=videos), indent=2))

for filename, items in [('successes.mp4', videos[:3]), ('failures.mp4', videos[3:]),
                        ('stage2_success_failure_overview.mp4', videos)]:
    listing = OUT / (filename + '.concat.txt')
    listing.write_text(''.join(f"file '{item['file']}'\n" for item in items))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(listing), '-c', 'copy', '-movflags', '+faststart',
                    str(OUT / filename)], check=True)
print('All videos complete', flush=True)
