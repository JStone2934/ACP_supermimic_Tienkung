"""Snapshot the selected actor AND WM, validate source semantics, produce a fresh v2 initialization."""
import argparse
import json
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import DEFAULT_CONFIG, TASK_NAME, FullbodyTask, migrate_policy
from supermimic.acp.aiming_fullbody_io import code_identity, json_save, save_checkpoint, snapshot
from supermimic.acp.networks import WorldModel
from supermimic.acp.standing import freeze_normalizers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--source')
    p.add_argument('--run-dir', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    source = Path(a.source or cfg['source']).resolve()
    out = Path(a.run_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8); torch.manual_seed(cfg['seed'])
    digest = snapshot(source, out / 'source' / 'checkpoint.pt')
    ck = torch.load(out / 'source' / 'checkpoint.pt', map_location='cpu', weights_only=False)
    task = FullbodyTask('cpu', cfg)
    pi, report = migrate_policy(ck, task)
    wm = WorldModel(task.mdp.scan_dim, robot=task.robot)
    wm.load_state_dict(ck['wm']); freeze_normalizers(wm)
    report.update(actor_source=str(source), wm_source=str(source), source_sha256=digest,
                  source_iteration=ck.get('iteration'), robot_metadata=task.robot.metadata(),
                  command_layout=task.metadata()['command_layout'])
    cfg['source'] = str((out / 'source' / 'checkpoint.pt').resolve())
    saved = dict(task=TASK_NAME, stage='initialization', pi=pi.state_dict(), wm=wm.state_dict(),
                 task_config=task.metadata(), robot_metadata=task.robot.metadata(), config=cfg,
                 migration=report, iteration=0, actor_updates=0, code_identity=code_identity())
    save_checkpoint(out / 'train' / 'init.pt', saved)
    json_save(out / 'source' / 'migration.json', report)
    json_save(out / 'config.json', cfg)
    json_save(out / 'code_identity.json', saved['code_identity'])
    print(json.dumps(dict(initialization=str(out / 'train' / 'init.pt'), source_sha256=digest)))


if __name__ == '__main__':
    main()
