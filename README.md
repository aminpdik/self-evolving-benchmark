# Self‑Evolving Benchmark Generator (Self‑Instruct + LLM‑as‑Judge + EMA)

This repository implements a **self‑evolving benchmark generator**. It continuously creates new evaluation items (a reusable **instruction template** + a concrete **user input**), solves them with an LLM, scores them with an LLM‑judge, and **keeps only items that are both novel and high‑quality**. Over time, the benchmark grows and the generation policy adapts using **Exponential Moving Average (EMA)** tracking.

The system is designed to work with **any OpenAI‑API‑compatible endpoint** (OpenAI, Azure OpenAI with compatible gateway, LM Studio, vLLM OpenAI server, etc.).


## High‑level algorithm

Each iteration of the loop does:

1. **Sample target category & difficulty**
   - Category sampling is biased toward **under‑represented categories**.
   - Difficulty is chosen from recent performance (EMA) to keep quality stable.

2. **Generate a new task (Self‑Instruct style)**
   - The generator sees a few example tasks (“demos”) from the current pool.
   - It must produce **exactly one new task** in **strict JSON** with fields:
     - `instruction` (reusable template; *no instance data*)
     - `user_input` (one concrete instance; *must be non‑empty*)
     - `category`, `difficulty`, `skills_tested`, `expected_answer_type`

3. **Heuristic novelty filter (fast)**
   - Compute Jaccard similarity over **word shingles** (k‑grams).
   - Reject if the candidate is too similar to an existing item.

4. **Answer the task**
   - Use the same endpoint (or a different one if you choose) to produce an answer.

5. **LLM‑as‑judge evaluation (quality + novelty)**
   - A judge model returns strict JSON:
     - `score_0_10`, `novel` (bool), `difficulty`, `tags`, `rationale`
   - Reject if `novel=false` or `score_0_10 < min_score`.

6. **Accept & store**
   - Append the accepted item to `data/benchmark.jsonl`.
   - Append an event record to `data/run_logs.jsonl`.

7. **Update EMAs and evolve the pool**
   - Update a global EMA (`ema_global`) and category EMA (`ema_category`).
   - Add the accepted task into the future generation pool.
   - Add the accepted text into the novelty memory.

> **Result:** the benchmark grows with items that are (1) *not duplicates* and (2) *good enough to be useful*,
while the system continuously adapts using EMA.


## What “Self‑Instruct” means here

The generator is prompted with several existing tasks and asked to produce **one new task** that is:
- **Novel** (not a rewording / trivial variation)
- **Solvable** (without external tools)
- Properly structured as:
  - `instruction`: *reusable template*
  - `user_input`: *one concrete instance*

This is implemented in:
- `prompts.py` → `self_instruct_new_task_prompt(...)`
- `generator.py` → `SelfInstructGenerator.generate_task(...)`


## Novelty: two‑stage filtering

### Stage A — Heuristic novelty (cheap)
Implemented in `novelty.py`:
- Normalize text → lowercase, remove punctuation, collapse whitespace
- Convert to **k‑shingles** (default `k=5` words)
- Use **Jaccard similarity** between shingles sets
- Reject if max similarity ≥ `--sim_threshold` (default 0.70)

This prevents wasting compute on obviously duplicated items.

### Stage B — LLM novelty (semantic)
Implemented in `evaluator.py` + `prompts.py`:
- The judge sees the candidate question + top similar examples.
- It returns `novel: true/false` based on semantic closeness.

This catches “same concept, different words” cases.


## EMA (Exponential Moving Average) and why it’s used

EMA smooths noisy scores over time:

\[ EMA_t = \alpha \cdot x_t + (1-\alpha) \cdot EMA_{t-1} \]

- `alpha` is `--ema_alpha` (default 0.1).
- `x_t` is the accepted item’s judge score (0–10).

EMAs are used for:
- **Global health**: how good are accepted items overall?
- **Per‑category health**: how well are items in each category doing?

Implemented in `ema.py` (`EMATracker.update`).


## Warm‑starting EMAs from previous runs

If you stop and restart, you typically **do not want EMAs to reset**. The warm‑start logic does:

- Load existing items from `data/benchmark.jsonl`
- Restore `ema_global` / `ema_category` from the most recent stored items when possible
- If those fields aren’t present, **replay historical scores** to rebuild EMAs

This lets your system **resume** rather than “forget”.

Implemented in `main.py`:
- `_extract_score(...)`
- `_warm_start_emas(...)`


## Repository structure

- `main.py`  
  Orchestrates the full loop; chooses category/difficulty; stores results; updates EMA.
- `client.py`  
  Minimal OpenAI‑compatible HTTP client using `requests` + `.env` configuration.
- `prompts.py`  
  All prompts: system prompt, task generation prompt, answer prompt, judge prompt.
- `generator.py`  
  Creates tasks (Self‑Instruct) and parses strict JSON.
- `answerer.py`  
  Answers tasks using the endpoint.
- `evaluator.py`  
  LLM‑judge scoring + novelty decision; parses strict JSON.
- `novelty.py`  
  Heuristic novelty filter (shingles + Jaccard).
- `ema.py`  
  EMA tracker.
- `store.py`  
  JSONL storage for benchmark items + run logs.
- `seeds.py`  
  Initial seed tasks to bootstrap the pool.


## Output files

By default, outputs go to:

- `data/benchmark.jsonl` — accepted benchmark items  
  Each line is one JSON object with fields like:
  - `instruction`, `user_input`, `category`, `difficulty`, `answer`, `judge`
  - `score_0_10`, `ema_global`, `ema_category`, `heuristic_max_similarity`, `created_at`

- `data/run_logs.jsonl` — event log  
  Includes generation errors, rejection reasons, acceptance events, etc.


## Setup

### 1) Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
```

### 2) Install dependencies

This repo uses only a few dependencies:

```bash
pip install requests python-dotenv
```


## Configure the endpoint (.env)

`client.py` loads `.env` from the repository folder (same directory as `client.py`).  
Your `.env` looks like this:

```env
BASE_URL=http://127.0.0.1:1234
API_KEY=dummy
MODEL=meta-llama-3.1-8b-instruct
TIMEOUT=120
```

Meaning:
- `BASE_URL`: the server base (the code will call `${BASE_URL}/v1/chat/completions`)
- `API_KEY`: optional; set `dummy` for local servers if they ignore auth
- `MODEL`: model name expected by your server
- `TIMEOUT`: request timeout (seconds)


### Option A — Use OpenAI (example)

```env
BASE_URL=https://api.openai.com
API_KEY=YOUR_OPENAI_KEY
MODEL=gpt-4o-mini
TIMEOUT=60
```

> Note: this project sends requests to `.../v1/chat/completions` using `requests`.
If you use a proxy/gateway, ensure it supports the **chat completions** endpoint.


### Option B — Use a local OpenAI‑compatible server

Examples of local servers that often work (depending on your setup):
- LM Studio “OpenAI Compatible” server
- vLLM OpenAI server
- any gateway that exposes `/v1/chat/completions`

Typical `.env`:

```env
BASE_URL=http://127.0.0.1:1234
API_KEY=dummy
MODEL=meta-llama-3.1-8b-instruct
TIMEOUT=120
```

If your server is hosted at a different port or host, change `BASE_URL` accordingly.


## Run

From the repo root:

```bash
python main.py   --accept_target 50   --max_attempts 400   --min_score 6.5   --ema_alpha 0.1   --sim_threshold 0.70   --demo_k 4   --gen_temp 0.9   --ans_temp 0.3   --judge_temp 0.2   --seed 7   --benchmark_path data/benchmark.jsonl   --runlog_path data/run_logs.jsonl
```

### Common knobs

- `--accept_target`: how many accepted items you want
- `--max_attempts`: includes rejected candidates; set higher for stricter novelty/score thresholds
- `--min_score`: minimum judge score to accept
- `--sim_threshold`: heuristic duplicate threshold (lower = stricter novelty)
- `--demo_k`: number of demo tasks shown to the generator
- `--gen_temp`, `--ans_temp`, `--judge_temp`: temperatures for each stage
- `--ema_alpha`: EMA smoothing; lower = smoother/slower adaptation


## Notes and best practices

- **Strict JSON outputs**: generation/judge both rely on JSON parsing. If your model often produces extra text,
  lower temperature and/or use a model that follows formatting well.
- **Novelty**: heuristic stage is cheap and should stay enabled. LLM novelty is slower but improves quality.
- **Scaling**: for large benchmarks, consider replacing shingles/Jaccard with MinHash/LSH (the code is structured
  so you can swap novelty methods cleanly later).


## Quick sanity check

After a run, inspect:

```bash
tail -n 3 data/benchmark.jsonl
tail -n 20 data/run_logs.jsonl
```

You should see accepted items with `score_0_10` and the EMA fields updating.


