# Gradual upright continuation from the old aiming policy

Source: wide_translated_full_gain7/ckpt_000001.pt, SHA256 recorded in init.json.
The source policy tensors and old bent reset state are exactly preserved at initialization.
The Take006 repaired pose is a weak legs/waist reference, not a replacement reset pose and not an arm target.

First bounded experiment:
- Level4 original wide targets / 10cm radius; all original task rewards retained.
- Added chest upright +0.2, chest excess -0.05, bounded lower-body reference +0.05 (scale0.35rad), unclipped request excess -0.1.
- Same-input frozen teacher anchor: subtract 2 * mean((raw-teacher_raw)*action_scale)^2 * dt from imagined objective. This is training-only regularization; physical reward excludes it. Teacher never learns; input gradients through teacher are intentionally disabled.
- Full policy trainable, old normalizers frozen, learning rate1e-6; arms may compensate for torso movement.
- Actor critic-tail gradient0; new critic/optimizers, existing world model adapted first.
- 512 environments total on two GPUs, WM batch256 / actor batch128 globally, H32 / WM window32.
- 100 model-only iterations then at most5 policy updates. Model gate includes chest and pelvis orientation P95<=2deg at configured validation horizons. If gate fails no actor update is permitted.
- Afterwards evaluate seeds17/53 for30s under the same original stage4 protocol. No automatic promotion/retry.

Paired review criteria: no formerly successful scenario lost; successful count does not decrease; median and P95 aim error not worse by more than0.1mm; median chest tilt after3s improves at least0.5deg. Chest comparison is sampled every0.2s. These are diagnostic continuation criteria, not final acceptance or robustness proof. Final precision5cm and target-switching tests remain required later.

The original baseline is imperfect: seed17 159/175 full task passes, seed53 152/175; standing161/175 and154/175. Random success95%/89%; both fixed grids fail. The experiment must improve posture without degrading this measured baseline, not claim the old policy fully passed.

Strict chest acceptance remains enabled and unchanged. In this first transition stage the bent initial pose does NOT meet the final initial-upright rule; reports must retain that failure. Only after a reliable physical transition is learned may a separate experiment add upright resets. A later successful posture-only metric must not be called final acceptance.

Artifacts: init.json, preflight.json, baseline/, runner_manifest.json, train/, evaluation/, comparison.json, experiment_status.json.
Service: upright-continuation-v1.service. Bounded runner: run_experiment.py.
The smoke checkpoint is isolated and must never initialize formal training.
