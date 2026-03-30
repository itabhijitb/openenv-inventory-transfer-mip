from __future__ import annotations

import os
import math
from typing import Dict, List, Tuple

from openai import OpenAI

from inventory_transfer_env import InventoryTransferAction, InventoryTransferEnv, Transfer


def _greedy_plan(obs) -> InventoryTransferAction:
    wh = {w.id: w for w in obs.warehouses}
    products = obs.products

    transfers: List[Transfer] = []

    remaining_budget = float("inf") if obs.budget is None else float(obs.budget)
    remaining_out = None if obs.outbound_capacity is None else {k: int(v) for k, v in obs.outbound_capacity.items()}
    remaining_in = None if obs.inbound_capacity is None else {k: int(v) for k, v in obs.inbound_capacity.items()}

    lane_remaining = None
    if obs.lane_capacity is not None:
        lane_remaining = {}
        for i, row in obs.lane_capacity.items():
            for j, by_p in row.items():
                for p, cap in by_p.items():
                    lane_remaining[(i, j, p)] = int(cap)

    sku_remaining = None
    if obs.sku_capacity is not None:
        sku_remaining = {}
        for w in obs.warehouses:
            by_p = obs.sku_capacity.get(w.id, {})
            for p, cap in by_p.items():
                sku_remaining[(w.id, p)] = int(cap) - int(w.inventory.get(p, 0))

    lots = {} if obs.min_transfer_lot is None else {p: int(v) for p, v in obs.min_transfer_lot.items()}

    fixed_cost = {} if obs.lane_fixed_cost is None else {
        (i, j): float(c)
        for i, row in obs.lane_fixed_cost.items()
        for j, c in row.items()
    }
    activated_lanes: set[tuple[str, str]] = set()

    # Greedy: for each product, send from surplus to deficit by cheapest lane
    for p in products:
        surplus: Dict[str, int] = {}
        deficit: Dict[str, int] = {}
        for w in obs.warehouses:
            inv = int(w.inventory.get(p, 0))
            dem = int(w.demand.get(p, 0))
            if inv > dem:
                surplus[w.id] = inv - dem
            elif dem > inv:
                deficit[w.id] = dem - inv

        while surplus and deficit and remaining_budget > 1e-9:
            # pick best (i,j) lane cost
            best: Tuple[str, str, float] | None = None
            for i in surplus:
                for j in deficit:
                    c = float(obs.transfer_cost[i][j])
                    if not math.isfinite(c) or c < 0:
                        continue

                    if lane_remaining is not None and lane_remaining.get((i, j, p), 0) <= 0:
                        continue

                    # Prefer reusing an already activated lane to avoid paying fixed cost again.
                    effective_cost = c
                    if (i, j) not in activated_lanes:
                        effective_cost += float(fixed_cost.get((i, j), 0.0))

                    if best is None or effective_cost < best[2]:
                        best = (i, j, effective_cost)
            if best is None:
                break
            i, j, _ = best

            q = min(surplus[i], deficit[j])
            if remaining_out is not None:
                q = min(q, max(0, remaining_out.get(i, 0)))
            if remaining_in is not None:
                q = min(q, max(0, remaining_in.get(j, 0)))

            if lane_remaining is not None:
                q = min(q, max(0, lane_remaining.get((i, j, p), 0)))

            if sku_remaining is not None:
                q = min(q, max(0, sku_remaining.get((j, p), 0)))

            lot = int(lots.get(p, 0) or 0)
            if lot > 0:
                q = (int(q) // lot) * lot

            lane_cost = float(obs.transfer_cost[i][j])
            lane_fixed = 0.0
            if (i, j) not in activated_lanes:
                lane_fixed = float(fixed_cost.get((i, j), 0.0))

            if lane_cost > 0:
                if math.isfinite(remaining_budget):
                    q = min(q, int(remaining_budget // lane_cost))
            else:
                q = int(q)

            if lot > 0:
                q = (int(q) // lot) * lot

            # Ensure we can afford the fixed activation if this is a new lane.
            if lane_fixed > 0 and math.isfinite(remaining_budget) and remaining_budget <= lane_fixed + 1e-9:
                break

            if q <= 0:
                break
            transfers.append(Transfer(from_warehouse=i, to_warehouse=j, product=p, quantity=int(q)))

            if (i, j) not in activated_lanes and lane_fixed > 0:
                remaining_budget -= lane_fixed
                activated_lanes.add((i, j))

            remaining_budget -= lane_cost * q
            if remaining_out is not None:
                remaining_out[i] = int(remaining_out.get(i, 0) - q)
            if remaining_in is not None:
                remaining_in[j] = int(remaining_in.get(j, 0) - q)

            if lane_remaining is not None:
                lane_remaining[(i, j, p)] = int(lane_remaining.get((i, j, p), 0) - q)

            if sku_remaining is not None:
                sku_remaining[(j, p)] = int(sku_remaining.get((j, p), 0) - q)

            surplus[i] -= q
            deficit[j] -= q
            if surplus[i] == 0:
                surplus.pop(i)
            if deficit[j] == 0:
                deficit.pop(j)

    return InventoryTransferAction(transfers=transfers)


def _score_from_obs(obs) -> float:
    if obs.score is not None:
        return float(obs.score)

    optimal_cost = obs.optimal_cost
    optimal_feasible = obs.optimal_feasible

    if not obs.is_feasible:
        return 0.0

    if optimal_feasible is False:
        return 0.0

    if optimal_cost is None or optimal_cost <= 0 or optimal_cost == float("inf"):
        return 0.0

    return max(0.0, min(1.0, optimal_cost / max(obs.total_cost, optimal_cost)))


def main() -> None:
    base_url = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

    use_llm = os.environ.get("USE_LLM", "0") == "1"
    api_base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    hf_token = os.environ.get("HF_TOKEN")

    if use_llm:
        if not api_base_url or not model_name or not hf_token:
            raise RuntimeError(
                "USE_LLM=1 requires API_BASE_URL, MODEL_NAME, HF_TOKEN environment variables"
            )
        client = OpenAI(base_url=api_base_url, api_key=hf_token)
        _ = client

    tasks = ["easy", "medium", "hard", "hard_v1", "hard_v2", "hard_v3"]

    results = []
    for task_id in tasks:
        # Some servers close WebSocket sessions after an episode.
        # Reconnect per task to be robust in automated evaluation.
        with InventoryTransferEnv(base_url=base_url).sync() as env:
            r = env.reset(task_id=task_id)
            action = _greedy_plan(r.observation)
            step = env.step(action)
            score = _score_from_obs(step.observation)
            results.append((task_id, score, step.observation.total_cost, step.observation))

    for task_id, score, cost, obs in results:
        dq = "DQ" if obs.disqualified else "OK"
        dq_reasons = "; ".join(obs.dq_reasons) if obs.dq_reasons else ""
        print(
            f"{task_id}: score={score:.4f} cost={cost:.2f} {dq} "
            f"opt_cost={obs.optimal_cost} ratio={obs.optimality_ratio} gap={obs.cost_gap} "
            f"fill_rate={obs.fill_rate:.4f} dq_reasons={dq_reasons}"
        )

    avg = sum(s for _, s, _, _ in results) / len(results)
    print(f"avg_score={avg:.4f}")


if __name__ == "__main__":
    main()
