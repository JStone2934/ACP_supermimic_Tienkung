"""Opt-in Stage1 reset/truncation curriculum. Does not change reward, policy inputs or failure termination.

Local episodes end after a fixed event window. Passing its tracking/contact stability gate
advances only its start, in seconds rather than clip-relative jumps. Failure keeps the hard
window in the sampling pool. State shared across ranks excludes transient simulator episodes.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import torch


@dataclasses.dataclass(frozen=True)
class TransitionCfg:
    local_fraction: float = .50
    full_fraction: float = .25
    retention_climb_fraction: float = .50  # fraction within the remaining 25%
    advance_s: float = .15
    stable_s: float = .20
    promotion_episodes: int = 20
    promotion_rate: float = .80
    stable_fraction: float = .80
    root_error_m: float = .20
    joint_error_rad: float = .50
    contact_recall: float = .80
    contact_distance_m: float = .02
    contact_slip_m_s: float = .20

    def __post_init__(self):
        if not (0 < self.local_fraction < 1 and 0 < self.full_fraction < 1
                and self.local_fraction + self.full_fraction < 1):
            raise ValueError('local/full/retention fractions must all be positive')
        if not 0 < self.retention_climb_fraction < 1:
            raise ValueError('retention must include climb suffixes and non-climb skills')
        for name in ('promotion_rate', 'stable_fraction', 'contact_recall'):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in (0, 1]')
        for name in ('advance_s', 'stable_s', 'root_error_m', 'joint_error_rad', 'contact_distance_m', 'contact_slip_m_s'):
            if not 0 < getattr(self, name) < float('inf'):
                raise ValueError(f'{name} must be finite and positive')
        if not isinstance(self.promotion_episodes, int) or self.promotion_episodes < 1:
            raise ValueError('promotion_episodes must be a positive integer')


class TransitionCurriculum:
    # Episode kinds: 0 local window, 1 full climb, 2 retained suffix/non-climb.
    def __init__(self, plan, motion, contacts, num_envs, device, runtime=None):
        if isinstance(plan, (str, Path)):
            plan = json.loads(Path(plan).read_text())
        self.plan = plan
        self.cfg = c = TransitionCfg(**plan.get('config', {}))
        self.plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        self.motion, self.contacts, self.runtime, self.device = motion, contacts, runtime, device
        if plan.get('version') != 1 or plan.get('data_hash') != motion.data_hash:
            raise ValueError('transition curriculum version/dataset mismatch')
        dt = motion.robot.control_dt
        self.advance = max(1, round(c.advance_s / dt))
        self.stable = max(1, round(c.stable_s / dt))
        if abs(plan['control_dt'] - dt) > 1e-9:
            raise ValueError('transition curriculum control rate mismatch')
        name_to_id = {n: i for i, n in enumerate(motion.names)}
        rows = []
        seen = set()
        climb_names = {n for n in motion.names if n.split('/')[0] in plan['groups']}
        if set(plan['clips']) != climb_names or not climb_names:
            raise ValueError('plan must cover every clip in its climb groups exactly')
        for name, entry in sorted(plan['clips'].items()):
            idx = name_to_id[name]
            length = int(motion.clip_len[idx])
            if entry['frames'] != length or not entry['events']:
                raise ValueError(f'invalid transition clip: {name}')
            for event in entry['events']:
                start, anchor, end = (event[k] for k in ('start', 'anchor', 'end'))
                if not all(isinstance(x, int) for x in (start, anchor, end)):
                    raise ValueError('window frames must be integers')
                if not 0 <= start <= anchor < end < length or end-start < self.stable:
                    raise ValueError(f'invalid transition window: {name} {event}')
                if (idx, anchor) in seen:
                    raise ValueError('duplicate transition anchor')
                seen.add((idx, anchor))
                rows.append((idx, start, end))
        r = torch.tensor(rows, dtype=torch.long, device=device)
        self.clip, self.initial_start, self.end = r.unbind(-1)
        self.start = self.initial_start.clone()
        self.climbs = torch.tensor(sorted({x[0] for x in rows}), device=device)
        self.nonclimbs = torch.tensor([i for i, n in enumerate(motion.names) if n not in climb_names], device=device)
        if not len(self.nonclimbs):
            raise ValueError('retention requires non-climb clips')
        # Sample clips uniformly first, so clips with more events do not dominate.
        self.events_by_clip = torch.full((len(motion.names), max(sum(x[0] == i for x in rows) for i in self.climbs.tolist())), -1,
                                        dtype=torch.long, device=device)
        self.event_counts = torch.zeros(len(motion.names), dtype=torch.long, device=device)
        for j, (idx, _, _) in enumerate(rows):
            self.events_by_clip[idx, self.event_counts[idx]] = j
            self.event_counts[idx] += 1
        self.attempts = torch.zeros(len(rows), dtype=torch.long, device=device)
        self.successes = torch.zeros_like(self.attempts)
        self.promotions = torch.zeros_like(self.attempts)
        self.totals = torch.zeros(3, 3, dtype=torch.long, device=device)  # started, ended, passed
        self.pending_starts = torch.zeros(3, dtype=torch.long, device=device)
        self.quality_counts = torch.zeros(7, dtype=torch.long, device=device)
        self.pending_quality = torch.zeros_like(self.quality_counts)
        # Fixed environment shares also fix the share of collected transitions. Drawing
        # a new role at every reset would underweight short local episodes in replay.
        quantile = (torch.arange(num_envs, device=device)+.5)/num_envs
        self.bucket = torch.where(quantile < c.local_fraction, 0,
                                 torch.where(quantile < c.local_fraction+c.full_fraction, 1, 2))
        retention_ids = (self.bucket == 2).nonzero().flatten()
        self.retained_nonclimb = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.retained_nonclimb[retention_ids] = ((torch.arange(len(retention_ids), device=device)+.5)
                                               / max(len(retention_ids),1) >= c.retention_climb_fraction)
        self.kind = torch.full((num_envs,), 2, dtype=torch.long, device=device)
        self.window = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.window_start = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode_end = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.good_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.checked_steps = torch.zeros_like(self.good_steps)

    def sample(self, env_ids, retained_phase):
        n = len(env_ids); c, m, dev = self.cfg, self.motion, self.device
        kind = self.bucket[env_ids]
        clips = self.climbs[torch.randint(len(self.climbs), (n,), device=dev)]
        retained_nonclimb = self.retained_nonclimb[env_ids]
        clips[retained_nonclimb] = self.nonclimbs[torch.randint(len(self.nonclimbs), (int(retained_nonclimb.sum()),), device=dev)]
        offset = (retained_phase[clips] * (m.clip_len[clips]-1)).long()
        # Retained climb episodes deliberately exercise the learned suffix.
        offset = torch.where((kind == 2) & ~retained_nonclimb,
                             torch.maximum(offset, ((m.clip_len[clips]-1)*.6).long()), offset)
        offset = torch.where(kind == 1, 0, offset)
        ends = m.clip_len[clips]-1
        windows = torch.full((n,), -1, dtype=torch.long, device=dev)
        local = kind == 0
        lc = clips[local]
        slot = (torch.rand(len(lc), device=dev)*self.event_counts[lc]).long()
        wi = self.events_by_clip[lc, slot]
        windows[local] = wi; offset[local] = self.start[wi]; ends[local] = self.end[wi]
        self.kind[env_ids], self.window[env_ids] = kind, windows
        self.window_start[env_ids] = offset
        self.episode_end[env_ids] = m.clip_start[clips] + ends
        self.good_steps[env_ids] = 0; self.checked_steps[env_ids] = 0
        self.pending_starts += torch.bincount(kind, minlength=3)
        frames = m.clip_start[clips] + offset
        return frames, m.frame_bin[frames]

    def explicit_reset(self, env_ids, frames):
        self.kind[env_ids] = 2; self.window[env_ids] = -1
        self.episode_end[env_ids] = self.motion.clip_end[self.motion.clip_id[frames]]
        self.good_steps[env_ids] = 0; self.checked_steps[env_ids] = 0
        self.pending_starts[2] += len(env_ids)

    @torch.no_grad()
    def observe(self, S0, S1, f1, variants):
        """Quality gates only for the final stable_s of local windows. No reward shaping."""
        ids = ((self.kind == 0) & (f1 > self.episode_end-self.stable)).nonzero().flatten()
        if not len(ids):
            return
        c, m, ct = self.cfg, self.motion, self.contacts
        f, v = f1[ids], variants[ids]
        prev, now = tuple(x[ids] for x in S0), tuple(x[ids] for x in S1)
        boxes = m.boxes(f, v)
        touched = ct.body_distance(now, boxes) <= c.contact_distance_m
        lab = ct.labels[f, ct.rv(v)]
        # Separate hands from other contacts: two feet cannot mask a missing hand.
        hand = torch.tensor([i for i,n in enumerate(m.robot.contact_bodies) if 'hand' in n], device=self.device)
        other = torch.tensor([i for i,n in enumerate(m.robot.contact_bodies) if 'hand' not in n], device=self.device)
        def coverage(mask, index):
            active = lab[:, index]; den = active.sum(-1)
            return ((mask[:, index] & active).sum(-1) >= c.contact_recall*den)
        points0, points1 = ct.contact_point(prev, boxes), ct.contact_point(now, boxes)
        slip = (points1-points0).norm(dim=-1) / m.robot.control_dt
        persistent = lab & ct.labels[(f-1).clamp(min=0), ct.rv(v)]
        slip_ok = (~persistent | (slip <= c.contact_slip_m_s)).all(-1)
        root_ok = ((now[0]-m.root_p[f]).norm(dim=-1) <= c.root_error_m)
        joint_ok = (now[4]-m.q[f]).square().mean(-1).sqrt() <= c.joint_error_rad
        hand_ok, other_ok = coverage(touched, hand), coverage(touched, other)
        good = root_ok & joint_ok & hand_ok & other_ok & slip_ok
        self.pending_quality += torch.stack((torch.ones_like(good), root_ok, joint_ok, hand_ok,
                                             other_ok, slip_ok, good)).sum(-1)
        self.good_steps[ids] += good.long(); self.checked_steps[ids] += 1

    def finish(self, f1, done, term):
        """All ranks call once per simulation step, even when no local episode ends."""
        passed = done & ~term & (f1 >= self.episode_end)
        local = done & (self.kind == 0)
        passed &= (self.kind != 0) | ((self.checked_steps >= self.stable)
                   & (self.good_steps >= self.cfg.stable_fraction*self.checked_steps))
        # Episodes already in flight when a window advances belong to the old difficulty.
        local &= self.window_start == self.start[self.window.clamp(min=0)]
        counts = torch.zeros(2, len(self.start), dtype=torch.long, device=self.device)
        counts[0].scatter_add_(0, self.window[local], torch.ones_like(self.window[local]))
        counts[1].scatter_add_(0, self.window[local], passed[local].long())
        totals = torch.stack((self.pending_starts, torch.bincount(self.kind[done], minlength=3),
                              torch.bincount(self.kind[passed], minlength=3)), -1)
        size = counts.numel()
        packed = torch.cat((counts.flatten(), totals.flatten(), self.pending_quality))
        if self.runtime is not None:
            packed = self.runtime.sum(packed)
        counts = packed[:size].view_as(counts); totals = packed[size:size+9].view(3,3)
        self.quality_counts += packed[size+9:]; self.pending_quality.zero_()
        self.pending_starts.zero_(); self.totals += totals
        self.update_windows(counts[0], counts[1])

    def update_windows(self, attempts, successes):
        self.attempts += attempts; self.successes += successes
        ready = self.attempts >= self.cfg.promotion_episodes
        promote = ready & (self.successes >= self.cfg.promotion_rate*self.attempts) & (self.start > 0)
        self.start[promote] = (self.start[promote]-self.advance).clamp(min=0)
        self.promotions[promote] += 1
        # Failed batches stay at the hard window, rather than retreating out of it.
        self.attempts[ready] = 0; self.successes[ready] = 0

    def state_dict(self):
        return dict(plan=self.plan, plan_hash=self.plan_hash, **{k:getattr(self,k).clone() for k in
                    ('start','attempts','successes','promotions','totals','quality_counts')})

    def load_state_dict(self, state):
        if state['plan_hash'] != self.plan_hash:
            raise ValueError('cannot resume a different transition plan; use an explicit new --init experiment')
        for k in ('start','attempts','successes','promotions','totals','quality_counts'):
            x = state[k].to(self.device)
            if x.shape != getattr(self,k).shape or x.dtype != torch.long or (x < 0).any():
                raise ValueError(f'invalid transition state: {k}')
            if k == 'start' and (x > self.initial_start).any():
                raise ValueError('transition start moved past its original hard window')
            getattr(self,k).copy_(x)
        if (self.successes > self.attempts).any():
            raise ValueError('transition successes exceed attempts')

    def summary(self):
        return dict(plan_hash=self.plan_hash, episode_kinds=['local','full_climb','retention'],
                    environments_per_rank=torch.bincount(self.bucket, minlength=3).cpu().tolist(),
                    counts=self.totals.cpu().tolist(), count_columns=['started','ended','passed'],
                    quality_columns=['checked','root_ok','joint_ok','hand_contact_ok','other_contact_ok','slip_proxy_ok','all_ok'],
                    quality_counts=self.quality_counts.cpu().tolist(),
                    windows=len(self.start), windows_at_start=int((self.start==0).sum()),
                    promotions=int(self.promotions.sum()), mean_start_frame=float(self.start.float().mean()),
                    note='local passed is a curriculum quality gate; full/retention passed only means no termination, not task acceptance')
