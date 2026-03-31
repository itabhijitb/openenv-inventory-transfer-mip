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

**Episode structure:** Single-step episode: one transfer plan decision, scored against an optimal MIP reference.

This environment models a practical SCM lever (lateral transshipment / inventory balancing) with realistic operational constraints. It is useful for evaluating planning agents because decisions are coupled across a network (lane selection, capacity bottlenecks, and fixed lane activation tradeoffs), and the grader provides a deterministic, reproducible signal grounded in an OR-Tools MIP optimum.

**Difficulty progression:** `easy` -> `medium` -> `hard` -> `hard_v1`/`hard_v2`/`hard_v3` add scale and tighter operational constraints, including lane fixed activation costs.

### Hard task variants: what each one is testing

| Task | What changes vs. baseline hard | What it tests |
| --- | --- | --- |
| `hard` | Baseline 8-warehouse, 2-SKU instance with budgets + inbound/outbound caps + lane caps + SKU caps + min lots + lane fixed activation costs | General planning under coupled constraints + fixed-charge lane selection |
| `hard_v1` | Tighter budget | Budget-constrained planning and prioritization |
| `hard_v2` | Different transfer-cost matrix pattern | Robustness to cost-structure changes; avoids overfitting to one lane topology |
| `hard_v3` | Higher budget but different transfer-cost matrix pattern | Tradeoff shift: more options become feasible; tests whether policies exploit expanded feasible set |

An additional adversarial stress-test task is included as `edge_case`.

**Constraints + KPIs (consolidated):** budget, per-warehouse inbound/outbound capacity, per-lane capacity, per-warehouse per-SKU storage capacity, per-product minimum transfer lots; KPI fields include `total_cost`, `fill_rate` (service level), `optimal_cost`, `score` in `[0,1]`, and disqualification reporting (`disqualified`, `dq_reasons`).

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
- **3+ tasks with graders**
  - Tasks: `easy`, `medium`, `hard`.
  - OR-Tools MIP optimal solver produces `optimal_cost` and normalized `score` in `[0,1]`.

### Phase 2: Agentic evaluation (scored)

- The environment is deterministic and single-shot: `reset()` then one `step()`.
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

- `easy`: 3 warehouses, 1 SKU
- `medium`: 5 warehouses, 3 SKUs + budget + lane caps + min lots + SKU storage caps
- `hard` / `hard_v1` / `hard_v2` / `hard_v3`: 8 warehouses, 2 SKUs + budget + inbound/outbound caps + lane caps + min lots + SKU storage caps + fixed per-lane activation cost

## Environment interface

The environment follows the OpenEnv `reset()` / `step()` / `state()` interface.

- `reset(task_id=...)` returns the initial observation for that task.
- `step(action)` consumes a list of transfers.

The `step()` observation includes:
- `total_cost` (transfer cost + stockout penalty)
- `fill_rate` (service level KPI)
- `optimal_cost` (MIP optimum)
- `score` in `[0,1]`
- `disqualified` + `dq_reasons` for constraint violations
