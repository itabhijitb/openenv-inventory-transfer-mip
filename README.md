---
title: Openenv Inventory Transfer Mip
emoji: 😻
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
short_description: OpenEnv SCM transfer env w/ MIP grader
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Inventory Transfer Optimization (MIP) — OpenEnv

This repository implements a real-world supply chain optimization environment:
**multi-warehouse inventory transfer** (a lateral transshipment / inventory balancing problem).

The agent proposes a set of transfers and is scored against an OR-Tools MIP optimal solution.

## Why this environment matters

**Episode structure:** Tasks range from single-step (one transfer plan decision) to multi-step (rolling horizon across N days). All episodes are scored against an OR-Tools MIP optimal reference.

This environment models a practical SCM lever (lateral transshipment / inventory balancing) with realistic operational constraints. It is useful for evaluating planning agents because decisions are coupled across a network (lane selection, capacity bottlenecks, and fixed lane activation tradeoffs), and the grader provides a deterministic, reproducible signal grounded in an OR-Tools MIP optimum.

### Real-world analogs (how this maps to practice)

- **Retail / e-commerce DC networks:** rebalance fast-moving SKUs across fulfillment centers to protect service levels.
- **Spare parts logistics:** reposition parts inventory across depots while respecting throughput and storage constraints.
- **Budgeted expedites:** decide when it is worth opening a lane (fixed activation cost) vs. accepting shortage penalty.

**Difficulty progression:** `easy` -> `medium` -> `hard` -> `hard_v1`/`hard_v2`/`hard_v3` add scale and tighter operational constraints, including lane fixed activation costs.

### Hard task variants: what each one is testing

| Task | What changes vs. baseline hard | What it tests |
| --- | --- | --- |
| `hard` | Baseline 8-warehouse, 2-SKU instance with budgets + inbound/outbound caps + lane caps + SKU caps + min lots + lane fixed activation costs | General planning under coupled constraints + fixed-charge lane selection |
| `hard_v1` | Tighter budget | Budget-constrained planning and prioritization |
| `hard_v2` | Different transfer-cost matrix pattern | Robustness to cost-structure changes; avoids overfitting to one lane topology |
| `hard_v3` | Higher budget but different transfer-cost matrix pattern | Tradeoff shift: more options become feasible; tests whether policies exploit expanded feasible set |

Two additional tasks stress-test specific structural challenges: `edge_case` (adversarial fixed-cost traps) and `hub_spoke` (topology-constrained routing). The `rolling_3day` task is a multi-step episode where lane capacity resets each day — the agent must spread transfers across 3 steps, with inventory carrying forward.

**Constraints + KPIs (consolidated):** budget (episode-level or per-step), per-warehouse inbound/outbound capacity, per-lane capacity (resets each step in multi-step tasks), per-warehouse per-SKU storage capacity, per-product minimum transfer lots; KPI fields include `total_cost`, `fill_rate` (service level), `lanes_activated` (operational complexity), `co2_kg` (deterministic emissions proxy), `optimal_cost`, `score` in `[0,1]`, and disqualification reporting (`disqualified`, `dq_reasons`).

## Using this environment for RL / agent training

The environment is a standard OpenEnv server — any OpenEnv-compatible agent can interact with it via the HTTP/WebSocket API. The score signal is dense and differentiates near-optimal from sub-optimal plans, making it well-suited for reward shaping.

```python
from inventory_transfer_env import InventoryTransferEnv, InventoryTransferAction, Transfer

with InventoryTransferEnv(base_url="http://127.0.0.1:8000").sync() as env:
    obs = env.reset(task_id="rolling_3day").observation  # multi-step episode
    done = False
    while not done:
        # Replace with your policy
        result = env.step(InventoryTransferAction(transfers=[]))
        obs = result.observation
        done = result.done
    print(f"score={obs.score:.3f}  optimal_cost={obs.optimal_cost}")
```

Key training signals available each episode:
- `reward` — negative cost at each step (transfer cost + eventual shortage penalty)
- `score` — normalised optimality ratio in `[0, 1]` on the final step
- `fill_rate` — service level (useful as a shaped reward for learning coverage)
- `dq_reasons` — structured constraint-violation messages for constraint-guided training

## Run locally

1. Install

```bash
pip install -e .
```

## Development (code quality)

Install dev tools:

```bash
pip install -e ".[dev]"
```

Run lint + typecheck:

```bash
ruff check .
pyright
```

## Versioning

- Code is versioned using SemVer-like tags (major/minor/patch).
- Task instances are defined in `inventory_transfer_env/tasks.json` and are treated as part of the public benchmark surface.
- Backward-compatible changes (docs, tooling, refactors) should not change task data or scoring.
- If task definitions or grading semantics change, bump the minor/major version and document it.

2. Start server

```bash
python -m uvicorn inventory_transfer_env.server.app:app --host 0.0.0.0 --port 8000
```

3. Run baseline

```bash
python inference.py
```

## Pre-submission checklist (must-pass)

### One-command pre-validation

```bash
bash pre_validate.sh
```

This runs:
- baseline `inference.py`
- `docker build` (requires Docker daemon running)
- `openenv validate`

### 1) OpenEnv spec compliance

Run the OpenEnv validator:

```bash
openenv validate .
```

Expected: `Ready for multi-mode deployment`.

## How judging works (and how this repo fits)

### Phase 1: Automated validation (pass/fail gate)

This repo is designed to pass the automated gate checks:

- **HF Space deploys**
  - Root `app.py` exports `app` and `openenv.yaml` points to `app: app:app`.
  - Use `python validate_submission.py --base-url https://YOUR-SPACE.hf.space` after deployment.
- **OpenEnv spec compliance**
  - `openenv validate` passes.
- **Dockerfile builds**
  - Root `Dockerfile` + `requirements.txt` provided.
- **Baseline reproduces**
  - Root `inference.py` finishes without error and prints scores.
- **10 tasks with graders**
  - Tasks: `easy`, `medium`, `hard`, `hard_v1`, `hard_v2`, `hard_v3`, `edge_case`, `hub_spoke`, `rolling_3day`, `noisy_demand`.
  - OR-Tools MIP optimal solver produces `optimal_cost` and normalized `score` in `[0,1]`.
  - Multi-step task (`rolling_3day`): 3-step episode with inventory carry-forward and per-step lane capacity reset.

### Phase 2: Agentic evaluation (scored)

- The environment is deterministic: `reset()` then one or more `step()` calls.
- Single-step tasks: `reset()` then one `step()` with a full transfer plan.
- Multi-step tasks (e.g. `rolling_3day`): `reset()` then N `step()` calls; inventory carries forward and lane capacity resets each step.
- The observation exposes clear numeric targets (`total_cost`, `optimal_cost`, `score`) so standard agents can optimize.
- `validate_submission.py` includes a **variance/determinism sanity check** (runs twice) and a **non-constant grader check**.

### Phase 3: Human review (utility + creativity + exploit checks)

- Real-world utility: lateral transshipment / inventory balancing with operational constraints.
- Constraints included to avoid toy behavior:
  - lane caps
  - warehouse SKU storage caps
  - min transfer lots
  - budgets and inbound/outbound caps

### Disqualification avoidance

- **Deploy + respond**: validated via `/health`, `reset`, `step`, `state`.
- **Non-trivial graders**: scores differ across tasks under the same null action; checked in `validate_submission.py`.
- **Baseline present**: `inference.py` at repo root.

### 2) Local server responds to reset/step/state

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Smoke test:

```bash
python validate_submission.py --base-url http://127.0.0.1:8000
```

### 3) Baseline reproduces scores

```bash
python inference.py
```

The script must finish without error and print per-task scores in `[0,1]`.

### 4) Docker build + run

```bash
docker build -t inventory-transfer-env:local .
docker run --rm -p 8000:8000 inventory-transfer-env:local
```

Then:

```bash
python validate_submission.py --base-url http://127.0.0.1:8000
```

### 5) HF Spaces deploy + automated ping

After pushing to a Hugging Face Space, verify:

- Space URL returns HTTP 200
- `/reset` works

Example:

```bash
python validate_submission.py --base-url https://YOUR-SPACE.hf.space
```

## HTTP API reference

The server exposes a FastAPI application. Interactive docs are available at:
- **`/docs`** — Swagger UI (try endpoints interactively)
- **`/redoc`** — ReDoc documentation

Core OpenEnv endpoints (under `/`):
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/reset` | POST | Start new episode; body: `{"task_id": "easy"}` |
| `/step` | POST | Apply action; body: `{"transfers": [...]}` |
| `/state` | GET | Current episode state |
| `/ws` | WS | WebSocket session (reset + step in one connection) |

Observation fields are documented in the `InventoryTransferObservation` schema visible at `/docs`.

## Required environment variables (LLM mode)

The baseline defaults to deterministic heuristic mode and does **not** call any LLM.

If you enable LLM mode, set:

```bash
export USE_LLM=1
export API_BASE_URL=...   # OpenAI-compatible endpoint
export MODEL_NAME=...     # model identifier
export HF_TOKEN=...       # HF / API key
```

## Tasks

| Task | Steps | Warehouses | SKUs | Key Challenge |
|---|---|---|---|---|
| `easy` | 1 | 3 | 1 | Baseline — lane fixed costs, lot sizes |
| `medium` | 1 | 5 | 3 | Multi-SKU coordination, budget |
| `hard` | 1 | 8 | 2 | All constraints active, tight budget |
| `hard_v1` | 1 | 8 | 2 | Tighter budget than `hard` |
| `hard_v2` | 1 | 8 | 2 | Different cost matrix |
| `hard_v3` | 1 | 8 | 2 | Relaxed budget, different costs |
| `edge_case` | 1 | 4 | 2 | Adversarial: asymmetric fixed costs, cross-supply trap |
| `hub_spoke` | 1 | 6 | 2 | Spoke-to-spoke transfers blocked — route through hub |
| `rolling_3day` | 3 | 5 | 2 | Lane capacity resets daily — plan across 3 days |
| `noisy_demand` | 1 | 5 | 2 | **Stochastic**: demand sampled ±25% per episode (pass `seed=` to `reset()`) |

### Stochastic demand task

The `noisy_demand` task introduces demand uncertainty. Mean demand is fixed in the task spec; each episode can sample perturbed demand by passing a seed:

```python
obs = env.reset(task_id="noisy_demand", seed=42)   # reproducible random demand
obs = env.reset(task_id="noisy_demand")              # deterministic mean demand
```

Demand is drawn from `Normal(mean, mean × 0.25)` and clamped to `[0, 2×mean]`. Because the grader re-solves the MIP against the realised demand, `score = 1.0` is still achievable — the challenge is learning a policy that adapts to demand variability rather than overfitting to fixed values.

## Environment interface

The environment follows the OpenEnv `reset()` / `step()` / `state()` interface.

- `reset(task_id=...)` returns the initial observation for that task.
- `step(action)` consumes a list of transfers.

The `step()` observation includes:
- `total_cost` (transfer cost + lane activation + stockout penalty)
- `fill_rate` (service level KPI)
- `optimal_cost` (MIP optimum, set on final step)
- `score` in `[0,1]` (set on final step)
- `disqualified` + `dq_reasons` for constraint violations
- `done` (True on final step)
- `max_steps` / `step_number` for multi-step episodes
- `lanes_activated`, `co2_kg` for operational KPIs
