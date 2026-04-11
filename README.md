---
title: Openenv Inventory Transfer Mip
emoji: 📦
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
short_description: Multi-warehouse lateral transshipment benchmark with MIP grader
---

# Inventory Transfer Optimization — OpenEnv Benchmark

**Multi-warehouse lateral transshipment** is one of the most impactful levers in supply chain management: redistribute surplus inventory from warehouses that have too much to warehouses that are running out — before demand materialises into stockouts.

This environment turns that daily, high-stakes planning decision into a clean RL / agent benchmark:

- **11 tasks** spanning single-step and multi-step episodes, deterministic and stochastic demand, and every major operational constraint found in real deployments (lane caps, fixed activation costs, lot sizes, SKU storage limits, per-step budgets).
- **MIP-grounded score**: every episode is scored against an OR-Tools SCIP exact optimum, so `score = 0.85` means the agent recovered 85 % of the maximum achievable cost reduction — not just "beat a heuristic".
- **OpenEnv-native**: works out of the box with any OpenEnv-compatible agent via HTTP / WebSocket.

---

## Why this benchmark fills a gap

Lateral transshipment appears in **every** large-scale e-commerce, retail, spare-parts, and pharmaceutical distribution network. A typical Fortune-500 logistics team runs this optimisation hundreds of times per day. Yet no public RL benchmark captures its full constraint complexity:

| Complexity dimension | This benchmark |
|---|---|
| Mixed-integer feasibility | Min transfer lots, binary lane activation |
| Network coupling | Transfers affect both source and destination capacity |
| Multi-period state carry-forward | Rolling-horizon tasks with inventory persistence |
| Demand uncertainty | Stochastic tasks with ±20–25 % noise |
| Cost–service tradeoff | Explicit shortage penalty vs. transfer cost budget |

Existing RL benchmarks for logistics are either toy (1-warehouse newsvendor) or black-box simulators with no ground-truth optimum. The MIP grader here gives a **tight, reproducible upper bound**, making it possible to measure how much optimality gap a model-based planner, LLM, or RL policy leaves on the table.

---

## Quick start

```bash
pip install -e .
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

```python
from inventory_transfer_env import InventoryTransferEnv, InventoryTransferAction, Transfer

with InventoryTransferEnv(base_url="http://127.0.0.1:8000").sync() as env:
    obs = env.reset(task_id="rolling_3day").observation   # 3-step rolling horizon
    done = False
    while not done:
        result = env.step(InventoryTransferAction(transfers=[]))  # replace with your policy
        obs, done = result.observation, result.done
    print(f"score={obs.score:.3f}   fill_rate={obs.fill_rate:.3f}   optimal_cost={obs.optimal_cost}")
```

Run the MIP + greedy baseline agent across all tasks:

```bash
python inference.py
```

Validate the server (null-action determinism + score range check):

```bash
python validate_submission.py --base-url http://127.0.0.1:8000
```

---

## Tasks

| Task | Steps | Warehouses | SKUs | Key challenge |
|---|---|---|---|---|
| `easy` | 1 | 3 | 1 | Baseline: lane fixed costs, lot sizes |
| `medium` | 1 | 5 | 3 | Multi-SKU coordination under budget |
| `hard` | 1 | 8 | 2 | All constraints active, tight budget |
| `hard_v1` | 1 | 8 | 2 | Tighter budget than `hard` |
| `hard_v2` | 1 | 8 | 2 | Different transfer-cost topology |
| `hard_v3` | 1 | 8 | 2 | Relaxed budget, new cost structure |
| `edge_case` | 1 | 4 | 2 | Adversarial: fixed-cost traps, cross-supply chains |
| `hub_spoke` | 1 | 6 | 2 | Spoke-to-spoke blocked — must route through hub |
| `rolling_3day` | 3 | 5 | 2 | Lane caps reset daily; inventory carries across 3 steps |
| `noisy_demand` | 1 | 5 | 2 | Stochastic demand ± 25 % per episode |
| `noisy_rolling` | 3 | 5 | 2 | Stochastic demand ± 20 % **and** 3-step rolling horizon |

**Difficulty progression:** `easy` → `medium` → `hard*` adds warehouses, SKUs, and constraint layers. `edge_case` and `hub_spoke` test structural reasoning. `rolling_3day` and `noisy_*` test sequential / adaptive planning. Together they form a curriculum for training and evaluating increasingly capable agents.

### Stochastic demand

`noisy_demand` and `noisy_rolling` expose demand uncertainty. Pass `seed=` to `reset()` for reproducible sampling:

```python
obs = env.reset(task_id="noisy_demand", seed=42)   # reproducible perturbed demand
obs = env.reset(task_id="noisy_demand")             # deterministic mean demand (default)
```

Demand is drawn from `Normal(mean, mean × noise_pct)` clamped to `[0, 2 × mean]`. The MIP grader re-solves against the realised demand each episode, so the task measures whether the agent's policy adapts to demand variability — not just memorises a fixed instance.

---

## Scoring

Every final-step observation includes:

| Field | Description |
|---|---|
| `score` | Normalised optimality ratio in `(0, 1)` — how close the agent came to the MIP optimum |
| `optimal_cost` | Reference MIP minimum cost (OR-Tools SCIP) |
| `total_cost` | Agent's actual cost (transfers + lane activation + shortage penalty) |
| `fill_rate` | Service level — fraction of demand units fulfilled |
| `cost_gap` | `total_cost − optimal_cost` in absolute cost units |
| `disqualified` | `True` if any constraint was violated |
| `dq_reasons` | List of constraint-violation messages |

`score ≈ 1.0` (capped at `0.999`) means the agent matched the exact MIP optimum. `score = 0.5` means the agent's cost was twice the minimum achievable.

### Training signals per step

| Signal | Notes |
|---|---|
| `reward` | `−total_step_cost` (continuous; non-zero every step, not just final) |
| `fill_rate` | Shaped service-level reward for coverage-focused objectives |
| `dq_reasons` | Structured constraint messages for constraint-guided or CMDP training |

---

## Real-world analogs

| Industry | Decision | Maps to |
|---|---|---|
| E-commerce / retail | Rebalance fast-moving SKUs across fulfillment centres before peak | `hard` / `hard_v*` tasks |
| Spare parts logistics | Reposition parts across depots to meet SLA | `hub_spoke`, `edge_case` |
| Pharma distribution | Rolling replenishment under uncertain demand | `noisy_rolling` |
| Seasonal retail | Multi-day markdown + transfer under budget | `rolling_3day` |

---

## HTTP API

Interactive docs: `/docs` (Swagger) · `/redoc`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/reset` | POST | Start episode: `{"task_id": "easy"}` |
| `/step` | POST | Apply action: `{"transfers": [{"from_warehouse":"W1","to_warehouse":"W2","product":"P1","quantity":10}]}` |
| `/state` | GET | Current episode state |
| `/ws` | WS | WebSocket session (reset + step in one connection) |

---

## LLM / API mode

The baseline defaults to a deterministic MIP + greedy planner. To enable the LLM planner:

```bash
export USE_LLM=1
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
export HF_TOKEN=<your-token>
python inference.py
```

---

## Development

```bash
pip install -e ".[dev]"
ruff check .
pyright
bash pre_validate.sh      # server + inference + docker build + openenv validate
openenv validate          # OpenEnv spec compliance check
```

---

## Deployment

Hosted at **https://huggingface.co/spaces/itabhijitb/openenv-inventory-transfer-mip**

```bash
# GitHub (triggers Actions auto-sync to HF Space if HF_TOKEN secret is set)
git push origin main

# HF Space direct (SSH — reliable on macOS where LibreSSL HTTPS may fail)
git push hf main
```
