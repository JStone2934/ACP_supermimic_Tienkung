# PHP conversion server ownership and resource protection

Read `/home/js/.local/share/php-guard/OPERATIONS.txt` before launching conversion work.

The user's original goal is owned by `php-codex.service`, independently of VS Code.
If `PHP_OVERNIGHT_OWNER` is not `1` and that service is active, only inspect status;
do not edit the algorithm, resume the goal, or launch a second autonomous executor.
Explain that the background owner is running. Explicit user instructions override
this rule; coordinate a single-owner handoff before making such a change.

Use the normal acp_supermimic Python environment and `python -m tools.convert_php`.
Its startup hook submits all controllers, including frozen runners, to the shared
memory queue. Keep requested worker counts and quality criteria visible. Never
bypass this hook or weaken the memory limits to accelerate a run. Queueing is not
a failed or hung conversion. Check `guard.py status` and the corresponding journal.

Before launching a batch, check existing job/output identity to avoid duplicate work.
Do not kill an adopted controller expecting it to stay stopped: use `guard.py cancel`
to disarm recovery first. Do not modify the guard or service configuration as part
of algorithm experiments. Preserve the original goal's acceptance criteria.
