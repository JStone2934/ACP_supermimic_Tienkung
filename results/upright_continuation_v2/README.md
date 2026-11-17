# Take006 22-joint reference continuation (v2)

User objective: keep the original standing/aiming task rewards; imitate the repaired Take006 frame with all joints EXCEPT the entire right arm. The right arm remains trainable and aims at the target.

Source policy: `results/lafan_jwp_aiming_20261115/wide_translated_full_gain7/ckpt_000001.pt` (same immutable source as v1, not the failed v1 result and not the video's distinct switching checkpoint). Existing seed17/53 baseline reports in v1 are reused only after matching checkpoint SHA256. Source tensors, initial root pose and joint pose are exactly preserved.

Reward contract:
- Every original reward definition/weight is retained.
- Exactly one added term: `0.2 * dt / (1 + mean(((q-reference)/0.35)^2))`, averaged over22 named native joints.
- Reference: repaired Take006 frame452 in `take006_ready/ready_pose.npz`, with its hash recorded in init.json.
- Included:12 leg joints +3 waist joints +7 left-arm joints.
- Excluded:all7 right-arm joints (shoulder pitch/roll/yaw, elbow pitch/yaw, wrist pitch/roll).
- No added chest shaping, no teacher-output retention penalty, no extra request-limit term. Original action-rate/joint-limit rewards still apply to all relevant joints.
- Right-arm exclusion means zero DIRECT reference-reward derivative with respect to those joint angles; it does not freeze their policy parameters or remove physical coupling.

First bounded diagnostic:
- Original stage4 target distribution, radius10cm, old bent reset pose retained.
- Full policy trainable, LR1e-6, critic-tail actor gradient0. No old critic/optimizer reused.
- Two GPUs,512 total environments, WM batch256 / actor batch128, H32 and WM window32.
- 100 model-only iterations, then at most ONE policy update. Save iteration100 and101.
- Pre-update gate on8 fresh512-step episodes, with chest/pelvis P95<=2deg plus original root/feet/aim prediction checks.
- Fresh post-update model gate; regardless of its result, independently evaluate frozen candidate for30s on seeds17/53.
- Retention review: no previously passing scenario lost; no drop in pass count; aim median/P95 no worse by >0.1mm. Invalid-ray quantiles are explicit null, never float32max.
- Pose review: median per-frame22-joint RMS reference error must decrease by at least0.1deg. Chest tilt is reported separately, not used as the new shaping reward or a replacement for original task metrics.
- No automatic next block or promotion. Zero actor updates because a gate failed is not a learned improvement. Final posture,5cm precision,and switching acceptance remain uncertified.

Runtime: `upright-continuation-v2.service`; progress in experiment_status.json and train/status.json. Results: post_update_gate.json, evaluation/, comparison.json.

Validation:10 unit tests passed; preflight verifies exact source weights/reset/original reward weights. Smoke is separate and never reused for formal training. Frozen code identity is in runner_manifest.json.
