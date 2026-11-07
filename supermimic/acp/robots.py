"""Instance-local robot contracts. Legacy calls default to G1; never switch module globals."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TIENKUNG_SHA = '4c2c061d72dcdab60d74e45d3efb17246a3b6f62'
CONTACT_NAMES = ('left_foot', 'right_foot', 'left_knee', 'right_knee', 'left_hand', 'right_hand', 'left_elbow', 'right_elbow', 'pelvis')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    mjcf_path: str
    joint_names: tuple[str, ...]
    tracked_bodies: tuple[str, ...]
    contact_bodies: dict
    default_q: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    effort: tuple[float, ...]
    velocity: tuple[float, ...]
    armature: tuple[float, ...]
    action_scales: tuple[float, ...]
    motion_dir: str
    torso_body: str = 'torso_link'
    root_body: str = 'pelvis'
    control_dt: float = .02
    decimation: int = 4
    stand_height: float = .76
    locked_joints: tuple[str, ...] = ()
    geometry_version: str = 'legacy-g1-spheres-v1'

    def __post_init__(self):
        if self.control_dt != .02:
            raise ValueError("ACP currently requires 50 Hz control; changing history/horizon timing needs a separate audit")
        if self.decimation < 1:
            raise ValueError("positive physics decimation required")
        if len(self.joint_names) != len(set(self.joint_names)):
            raise ValueError("duplicate canonical joint names")

    @property
    def num_joints(self): return len(self.joint_names)
    @property
    def state_dim(self): return 13 + 2 * self.num_joints
    @property
    def xi_dim(self): return 4 + self.num_joints
    @property
    def obs_dim(self): return 12 + 3 * self.num_joints
    @property
    def inv_dim(self): return 13 + 2 * self.num_joints
    @property
    def wm_out(self): return 12 + 2 * self.num_joints

    def pd_params(self): return self.kp, self.kd, self.effort, self.velocity

    def get_spec(self):
        if self.robot_id == 'g1':
            from .g1 import get_spec
            return get_spec()
        spec = mujoco.MjSpec.from_file(self.mjcf_path)
        if len(spec.actuators):
            raise ValueError('asset must not contain actuators')
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                geom.contype, geom.conaffinity = 0, 1
        return spec

    def entity_cfg(self):
        from mjlab.actuator import BuiltinPositionActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
        acts = tuple(BuiltinPositionActuatorCfg(target_names_expr=(n,), stiffness=k, damping=d,
                      effort_limit=e, armature=a) for n,k,d,e,a in zip(self.joint_names,self.kp,self.kd,self.effort,self.armature))
        return EntityCfg(init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,self.stand_height),
                         joint_pos=dict(zip(self.joint_names,self.default_q)), joint_vel={'.*':0.}),
                         spec_fn=self.get_spec,
                         articulation=EntityArticulationInfoCfg(actuators=acts,soft_joint_pos_limit_factor=.9))

    def metadata(self):
        config = asdict(self)
        config.pop('mjcf_path'); config.pop('motion_dir')
        model_hash = sha256(self.mjcf_path)
        manifest_path = Path(self.mjcf_path).parent.parent/'manifests/assets.json'
        if self.robot_id != 'g1':
            manifest = json.loads(manifest_path.read_text())
            root = manifest_path.parent.parent
            for rel, digest in manifest['files'].items():
                if sha256(root/rel) != digest:
                    raise ValueError(f'asset hash mismatch: {rel}')
            model_hash = hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()
        return dict(robot_id=self.robot_id,model_hash=model_hash,
                    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                    joint_order=list(self.joint_names),control_dt=self.control_dt)

    def validate_metadata(self, metadata, *, legacy=False):
        if metadata is None and legacy and self.robot_id == 'g1': return
        if not isinstance(metadata,dict): raise ValueError(f'{self.robot_id}: missing robot metadata')
        for k,v in self.metadata().items():
            if metadata.get(k) != v: raise ValueError(f'robot metadata mismatch: {k}: expected {v!r}, got {metadata.get(k)!r}')

    def joint_addresses(self, model):
        return (np.array([model.jnt_qposadr[model.joint(n).id] for n in self.joint_names]),
                np.array([model.jnt_dofadr[model.joint(n).id] for n in self.joint_names]))


def get_robot(robot: str | RobotSpec = 'g1') -> RobotSpec:
    if isinstance(robot,RobotSpec): return robot
    if robot == 'g1':
        from . import g1
        return RobotSpec('g1',g1.MJCF_PATH,tuple(g1.G1_JOINT_NAMES),tuple(g1.TRACKED_BODIES),g1.CONTACT_BODIES,
                         tuple(g1.default_joint_pos_list()),*(tuple(x) for x in g1.pd_params()),
                         tuple(g1.motor_of(n)[2] for n in g1.G1_JOINT_NAMES),
                         tuple(g1.action_scale(n) for n in g1.G1_JOINT_NAMES),g1.MOTION_DIR)
    if robot != 'tienkung3_29dof': raise ValueError(f'unknown robot {robot!r}')
    root = ROOT/'data/tienkung3_29dof'
    cfg = json.loads((root/'manifests/robot.json').read_text())
    for k in ('joint_names','tracked_bodies','default_q','kp','kd','effort','velocity','armature','action_scales','locked_joints'):
        cfg[k] = tuple(cfg[k])
    return RobotSpec(mjcf_path=str(root/'generated/robot.xml'),motion_dir=str(root/'motions/bootstrap'),**cfg)


def resolve_robot(explicit, container):
    """Checkpoint/trajectory identity takes precedence; old objects are G1 only."""
    meta = container.get('robot_metadata')
    if isinstance(meta,np.ndarray): meta = json.loads(str(meta.item()))
    name = meta['robot_id'] if meta else 'g1'
    if explicit is not None and explicit != name:
        raise ValueError(f'explicit robot {explicit!r} conflicts with stored robot {name!r}')
    robot = get_robot(name)
    robot.validate_metadata(meta,legacy=True)
    return robot
