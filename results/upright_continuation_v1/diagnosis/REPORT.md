# Upright continuation v1 failure diagnosis

No new training was launched. All physical work below used frozen policies. The original experiment/checkpoints/comparison are preserved.

## Findings

1. Identity and accounting: init -> checkpoint100 policy tensors identical; reset config unchanged; baseline/candidate evaluation initial root/quaternion/joints/targets exactly equal. Five policy updates occurred. Normalizer buffers unchanged. actor_value_gradient=0, so this is not the earlier critic-tail-gradient failure. The reported33/34 task passes exclude the newly added chest acceptance, so collapse is not merely stricter grading.

2. Actual balance deteriorated, not just aiming. Seed17 standing161->70/175 and terminations14->74; seed53 standing154->64 and terminations19->76. All recorded state-finite flags remain true.

3. Frozen output-swap experiment:16 selected seed17 scenarios, same saved initial states and target parameters,30s, no resets. Full passes: old15, new3, new body+old right arm6, old body+new right arm14. Standing passes15/8/9/14. Policies receive the same current observations within each world; outputs, not weights, are mixed. Body means22 outputs excluding the7 right-arm joints. Body changes explain much of the regression on this subset; arm-only change is not harmless. This is a diagnostic subset, not full acceptance or a claim of unique joint causality.

4. Same-input changes on16 recorded states: joint-target-request RMS0.193deg, max0.720deg. Right ankle pitch RMS0.603deg, left0.438deg, waist pitch0.289deg; right-arm per-joint RMS0.006–0.032deg. Small learning rate does not ensure small physical closed-loop impact. We have not isolated each ankle as a necessary cause.

5. Retention penalty is connected, but too weak in this probe. At the source, student==teacher so squared retention loss AND gradient are exactly0. After5 updates on fixed source-WM rollout, retention objective=-1.23e-5 and gradient norm0.0331, versus chest14.425, aim17.777, stability14.775. Mean joint-angle squared error averages29 outputs and poorly reflects sensitivity of balance joints. It is a soft penalty, not a constraint on30s task retention. These norms are all-policy gradients on16 starts, not a replay-wide average.

6. Model-space objective did not predict the observed damage: fixed checkpoint100 WM, same16 physical starts, H32 reward values: aim2.0811->2.0953, stability5.4844->5.5218, chest-0.3781->-0.3555, pose0.01719->0.01737. Parameter displacement has positive source-gradient dot for all these groups. Thus this diagnostic does not support a simple 'sacrificed imagined aiming to improve posture' explanation. It supports an imagined-vs-physical discrepancy; it is not a direct measurement of physics Jacobian error.

7. The approval gate was checked BEFORE updates, using8 episodes of128steps(2.56s), H32(0.64s) open-loop forecasts. No post-update validation occurred inside the5-update block before completion (validation interval5; loop ended). New read-only checks extend to512steps(10.24s), identical seed17, fixed PRE-update WM. Old-policy tip error P95=2.83cm, new-policy=6.24cm (>5cm gate); chest orientation P95=0.160->0.955deg, still below2deg. Old passes, new fails. This shows the model used to authorize the update does not retain its gate accuracy on the shifted policy distribution. It does not prove that merely tightening the chest gate would fix the controller.

8. Huge P95 numbers are a reporting defect: ray_metrics deliberately emits +inf when the forward ray/plane intersection is invalid. np.nan_to_num(error,nan=np.inf) leaves posinf unspecified and converts existing +inf to float32 max3.402823466e38. Invalid errors cover6.52%/7.58% of candidate post-grace frames; none is a numerical NaN. Invalid frames in these two candidate evaluations occur only in scenarios failing standing acceptance. The corrected report explicitly counts invalid rays and uses null for a percentile falling in the invalid tail. Finite-only P95 is5.53m/6.87m, conditional on valid intersections and NOT a replacement success metric. New error_summary utility has3 passing regression tests. Original experiment report is retained as evidence.

## Recommended next experiment (not launched)

- Roll back to the immutable old policy, not failed105. Keep the existing final physical criteria.
- Use single-update paired physical checks and reject/backtrack updates that damage balance/aiming; five updates and low scalar LR were not an adequate trust limit.
- Limit per-joint request changes and check ankle/leg sensitivity; a uniform mean-over29 penalty cannot guarantee retention. Do not simply multiply the chest reward.
- Preserve most of the original feedback controller initially; test small explicit posture residuals with arm compensation and physical retention checks, rather than moving every shared layer at once.
- Revalidate on current-policy trajectories AFTER each candidate update, including longer episodes/contact transitions. Add physical directional probes before assuming differentiable WM gradients are reliable.

Evidence: identity.json; trajectory_summary.json; gradients.json and gradients.py; body_arm.json and body_arm.py; model_policy100/105.json/.npz and check_model.py; corrected_error_statistics.json.
