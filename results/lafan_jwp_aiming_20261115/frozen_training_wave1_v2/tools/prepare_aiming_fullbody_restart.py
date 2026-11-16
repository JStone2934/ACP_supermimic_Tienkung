"""Fork an identified v2 policy/WM into a new reward and stage-exploration experiment."""
import argparse
import json
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyTask, TASK_NAME, validate_checkpoint
from supermimic.acp.aiming_fullbody_curriculum import StageExploration
from supermimic.acp.aiming_fullbody_io import json_save, save_checkpoint, snapshot, code_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(); torch.set_num_threads(2)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Restart preparation requires an empty directory')
    out.mkdir(parents=True, exist_ok=True)
    digest = snapshot(a.source, out / 'source.pt')
    ck = torch.load(out / 'source.pt', map_location='cpu', weights_only=False)
    if ck.get('task') != TASK_NAME or ck.get('smoke') or ck.get('stage') != 'training':
        raise ValueError('A formal fullbody v2 checkpoint is required')
    validate_checkpoint(ck, FullbodyTask('cpu', ck['config']))
    cfg = json.loads(Path(a.config).read_text())
    cfg['source'] = str(Path(a.source).resolve())
    task = FullbodyTask('cpu', cfg)
    if task.robot.metadata() != ck['robot_metadata']:
        raise ValueError('Restart robot identity differs')
    if not cfg.get('stage_exploration') or task.weights.get('request_limit', 0.) >= 0:
        raise ValueError('Restart requires request penalty and stage exploration')
    curriculum = dict(ck['curriculum'], consecutive=0)
    exploration = StageExploration(cfg['stage_exploration'], curriculum['level'])
    migration = dict(source=str(Path(a.source).resolve()), source_sha256=digest,
                     source_iteration=ck['iteration'], source_actor_updates=ck['actor_updates'],
                     inherited=['pi', 'wm', 'curriculum.mastered', 'curriculum.level'],
                     reset=['critic', 'optimizers', 'replay', 'iteration', 'exploration', 'best'],
                     reason='H8 request penalty and stage-local exploration')
    init = dict(task=TASK_NAME, stage='initialization', smoke=False, config=cfg,
                task_config=task.metadata(), robot_metadata=task.robot.metadata(),
                pi=ck['pi'], wm=ck['wm'], curriculum=curriculum,
                exploration=exploration.state_dict(), migration=migration, code_identity=code_identity())
    save_checkpoint(out / 'init.pt', init)
    json_save(out / 'config.json', cfg); json_save(out / 'migration.json', migration)
    print(migration)


if __name__ == '__main__':
    main()
