from __future__ import annotations

import argparse
import time
import urllib.request

from inventory_transfer_env import InventoryTransferAction, InventoryTransferEnv


def _http_get_ok(url: str, timeout_s: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    args = parser.parse_args()

    start = time.time()

    health_url = args.base_url.rstrip("/") + "/health"
    if not _http_get_ok(health_url, timeout_s=args.timeout_s):
        raise SystemExit(f"FAIL: {health_url} did not return 200")

    tasks = [
        "easy", "medium", "hard", "hard_v1", "hard_v2", "hard_v3",
        "edge_case", "hub_spoke", "rolling_3day", "noisy_demand", "noisy_rolling",
    ]

    null_scores_run1 = {}
    null_scores_run2 = {}

    def run_null_action_once(out: dict) -> None:
        for task_id in tasks:
            # Some servers close WebSocket sessions after an episode.
            # Reconnect per task to be robust in automated validation.
            with InventoryTransferEnv(base_url=args.base_url).sync() as env:
                reset = env.reset(task_id=task_id)
                if reset.done is not False:
                    raise SystemExit(
                        f"FAIL: reset.done must be False for task {task_id}"
                    )

                st = env.state()
                if not getattr(st, "episode_id", None):
                    raise SystemExit(
                        f"FAIL: missing episode_id in state for task {task_id}"
                    )

                # Multi-step episodes need N steps before done=True.
                max_steps = getattr(reset.observation, "max_steps", 1)
                action = InventoryTransferAction(transfers=[])
                step = None
                for _s in range(max_steps):
                    step = env.step(action)
                    if _s < max_steps - 1:
                        if step.done is True:
                            raise SystemExit(
                                f"FAIL: step.done=True too early at step {_s+1}/{max_steps} for {task_id}"
                            )

                if step is None or step.done is not True:
                    raise SystemExit(
                        f"FAIL: step.done must be True on final step for task {task_id}"
                    )

                obs = step.observation
                if obs.score is None:
                    raise SystemExit(
                        f"FAIL: missing score in observation for task {task_id}"
                    )
                if not (0.0 <= float(obs.score) <= 1.0):
                    raise SystemExit(
                        f"FAIL: score out of range for task {task_id}: {obs.score}"
                    )

                if obs.disqualified and not obs.dq_reasons:
                    raise SystemExit(
                        f"FAIL: disqualified but dq_reasons empty for task {task_id}"
                    )

                out[task_id] = float(obs.score)

    # Run twice for determinism/variance sanity
    run_null_action_once(null_scores_run1)
    run_null_action_once(null_scores_run2)

    if set(null_scores_run1.keys()) != set(tasks):
        raise SystemExit("FAIL: did not evaluate all tasks")

    # Grader sanity: avoid constant-score grader disqualification
    if len(set(null_scores_run1.values())) == 1:
        raise SystemExit(
            "FAIL: null-action scores are constant across tasks; graders may be considered non-informative"
        )

    # Variance check across repeated runs
    for t in tasks:
        if abs(null_scores_run1[t] - null_scores_run2[t]) > 1e-9:
            raise SystemExit(
                f"FAIL: non-deterministic scoring for task {t}: {null_scores_run1[t]} vs {null_scores_run2[t]}"
            )

    elapsed = time.time() - start
    if elapsed > args.max_seconds:
        raise SystemExit(f"FAIL: validation runtime too slow: {elapsed:.2f}s > {args.max_seconds:.2f}s")

    print(f"PASS: endpoints + tasks OK (elapsed={elapsed:.2f}s)")


if __name__ == "__main__":
    main()
