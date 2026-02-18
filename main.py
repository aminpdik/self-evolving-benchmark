"""Entry point for the self-evolving benchmark generator.

This script orchestrates the loop:
1) generate a new task (Self-Instruct),
2) filter for heuristic novelty,
3) answer the task,
4) judge quality + novelty,
5) store accepted items and update exponential moving averages (EMA).
"""

# CLI argument parsing
import argparse

# random sampling for categories + demos
import random

# used to count how many tasks per category
from collections import defaultdict

# typing helpers
from typing import Any, Dict, List, Tuple

# Core modules of the system
from answerer import Answerer                 # generates answer for task
from client import load_client_from_env       # loads OpenAI-compatible client
from ema import EMATracker                    # exponential moving average tracker
from evaluator import Evaluator               # LLM judge/scorer
from generator import SelfInstructGenerator   # generates new tasks
from novelty import NoveltyFilter             # heuristic duplicate filter
from seeds import SEED_TASKS                  # initial seed tasks
from store import JSONLStore, StorePaths      # storage for benchmark + logs


# Supported task categories
CATEGORIES = ["reasoning", "math", "coding", "science", "ml", "systems", "planning"]


def pick_category(category_counts: Dict[str, int], rng: random.Random) -> str:
    """Pick a category, biasing toward underrepresented ones."""

    # Build sampling weights
    weights = []
    for c in CATEGORIES:
        # Less-used categories get higher weight
        w = 1.0 / (1.0 + category_counts.get(c, 0))
        weights.append(w)

    total = sum(weights)           # sum of weights
    r = rng.random() * total       # random sample in [0,total]
    acc = 0.0

    # Weighted sampling
    for c, w in zip(CATEGORIES, weights):
        acc += w
        if r <= acc:
            return c

    # fallback (should not happen)
    return CATEGORIES[-1]


def pick_difficulty(category_ema: Dict[str, float], category: str) -> str:
    """Choose difficulty based on EMA performance."""

    ema = category_ema.get(category)

    # If no EMA yet → start medium
    if ema is None:
        return "medium"

    # If model is doing very well → harder tasks
    if ema >= 8.2:
        return "hard"

    # If weak → stabilize with medium
    if ema <= 6.2:
        return "medium"

    return "medium"


def validate_task_shape(task: Dict[str, Any]) -> Tuple[bool, str]:
    """Ensure generated task has required structure."""

    required = [
        "instruction",
        "user_input",
        "category",
        "difficulty",
        "skills_tested",
        "expected_answer_type",
    ]

    # Check required keys exist
    for k in required:
        if k not in task:
            return False, f"Missing field: {k}"

    # instruction must be non-empty
    if not isinstance(task["instruction"], str) or not task["instruction"].strip():
        return False, "instruction must be a non-empty string"

    # user_input must be non-empty
    if not isinstance(task["user_input"], str) or not task["user_input"].strip():
        return False, "user_input must be a non-empty string"

    # difficulty must be valid
    if task["difficulty"] not in ["easy", "medium", "hard"]:
        return False, "Invalid difficulty"

    # skills must be list
    if not isinstance(task["skills_tested"], list):
        return False, "skills_tested must be a list"

    return True, "ok"


def _extract_score(item: Dict[str, Any]) -> float | None:
    """Extract numeric score from stored item."""

    # preferred field
    if "score_0_10" in item:
        try:
            return float(item["score_0_10"])
        except Exception:
            return None

    # fallback: judge output
    judge = item.get("judge", {})
    if isinstance(judge, dict) and "score_0_10" in judge:
        try:
            return float(judge["score_0_10"])
        except Exception:
            return None

    return None


def _warm_start_emas(
    existing_items: List[Dict[str, Any]],
    alpha: float,
    ema_init: float | None,
) -> tuple[EMATracker, Dict[str, EMATracker]]:
    """Warm-start EMA trackers from stored benchmark."""

    # Global EMA tracker
    ema_global = EMATracker(alpha=alpha, ema=ema_init)

    # Category EMAs
    ema_by_cat: Dict[str, EMATracker] = {
        c: EMATracker(alpha=alpha) for c in CATEGORIES
    }

    # If no previous data → return fresh EMAs
    if not existing_items:
        return ema_global, ema_by_cat

    # If user forces init → use that
    if ema_init is not None:
        for it in existing_items:
            cat = it.get("category", "unknown")
            if cat in ema_by_cat:
                s = _extract_score(it)
                if s is not None:
                    ema_by_cat[cat].update(s)
        return ema_global, ema_by_cat

    # Try to reuse last stored EMA
    last = existing_items[-1]
    try:
        if last.get("ema_global") is not None:
            ema_global.ema = float(last["ema_global"])
    except Exception:
        pass

    # Try reuse per-category EMA
    for it in reversed(existing_items):
        cat = it.get("category", "unknown")
        if cat in ema_by_cat:
            if it.get("ema_category") is not None:
                try:
                    ema_by_cat[cat].ema = float(it["ema_category"])
                    break
                except Exception:
                    pass

    # If still missing → rebuild by replaying scores
    if ema_global.ema is None:
        for it in existing_items:
            s = _extract_score(it)
            if s is not None:
                ema_global.update(s)

    for c in CATEGORIES:
        if ema_by_cat[c].ema is None:
            for it in existing_items:
                if it.get("category") == c:
                    s = _extract_score(it)
                    if s is not None:
                        ema_by_cat[c].update(s)

    return ema_global, ema_by_cat


def run(args: argparse.Namespace) -> None:
    """Main loop runner."""

    # Load OpenAI-compatible client from env
    client = load_client_from_env()

    # Initialize storage paths
    store = JSONLStore(
        StorePaths(
            benchmark_jsonl=args.benchmark_path,
            runlog_jsonl=args.runlog_path,
        )
    )

    # Start with seed tasks
    task_pool: List[Dict[str, Any]] = list(SEED_TASKS)

    # Novelty filter
    novelty = NoveltyFilter(k_shingle=5)

    # Load previous benchmark items
    existing_items = list(store.iter_benchmark_items())

    # Warm-start EMA trackers from stored items
    ema_global, ema_by_cat = _warm_start_emas(
        existing_items=existing_items,
        alpha=args.ema_alpha,
        ema_init=args.ema_init,
    )

    # Add existing tasks to pool
    for it in existing_items:
        task_pool.append(
            {
                "instruction": it["instruction"],
                "user_input": it["user_input"],
                "category": it.get("category", "unknown"),
                "difficulty": it.get("difficulty", "medium"),
                "skills_tested": it.get("skills_tested", []),
                "expected_answer_type": it.get("expected_answer_type", "text"),
            }
        )

        # Add to novelty memory
        novelty.add(it["instruction"] + "\n" + it["user_input"])

    # Initialize components
    generator = SelfInstructGenerator(client, rng_seed=args.seed)
    answerer = Answerer(client)
    evaluator = Evaluator(client)

    rng = random.Random(args.seed)

    # Track how many tasks per category
    category_counts = defaultdict(int)
    for it in existing_items:
        cat = it.get("category", "unknown")
        if cat in CATEGORIES:
            category_counts[cat] += 1

    accepted = 0
    attempts = 0

    # =============================
    # MAIN LOOP
    # =============================
    while accepted < args.accept_target and attempts < args.max_attempts:
        attempts += 1

        # choose category
        category = pick_category(category_counts, rng)

        # difficulty based on EMA
        cat_ema_val = {c: ema_by_cat[c].ema for c in CATEGORIES if ema_by_cat[c].ema is not None}
        difficulty = pick_difficulty(cat_ema_val, category)

        # ---- Generate task ----
        try:
            task = generator.generate_task(
                task_pool=task_pool,
                target_category=category,
                target_difficulty=difficulty,
                demo_k=args.demo_k,
                temperature=args.gen_temp,
            )
        except Exception as e:
            store.append_runlog({"event": "gen_error", "error": str(e)})
            continue

        # validate task
        ok, msg = validate_task_shape(task)
        if not ok:
            store.append_runlog({"event": "gen_invalid", "reason": msg})
            continue

        qtext = task["instruction"] + "\n" + task["user_input"]

        # ---- heuristic novelty ----
        is_novel_h, max_sim = novelty.is_novel_heuristic(qtext, threshold=args.sim_threshold)
        if not is_novel_h:
            store.append_runlog({"event": "reject_heuristic_duplicate"})
            continue

        sim_hits = novelty.top_k_similar(qtext, k=5)
        similar_texts = [h.text for h in sim_hits if h.score > 0.25][:5]

        # ---- answer ----
        try:
            ans = answerer.answer(task["instruction"], task["user_input"], temperature=args.ans_temp)
        except Exception as e:
            store.append_runlog({"event": "answer_error", "error": str(e)})
            continue

        # ---- judge ----
        try:
            judge = evaluator.evaluate(
                instruction=task["instruction"],
                user_input=task["user_input"],
                answer=ans,
                similar_examples=similar_texts,
                temperature=args.judge_temp,
            )
        except Exception as e:
            store.append_runlog({"event": "judge_error", "error": str(e)})
            continue

        score = float(judge.get("score_0_10", -1))
        novel_llm = bool(judge.get("novel", False))

        # reject if LLM says not novel
        if not novel_llm:
            continue

        # reject if low quality
        if score < args.min_score:
            continue

        # ---- ACCEPT ----
        accepted += 1
        category_counts[category] += 1

        # update EMAs
        global_ema = ema_global.update(score)
        ema_by_cat[category].update(score)

        # store item
        item = {
            "id": f"item_{len(existing_items) + accepted:06d}",
            "instruction": task["instruction"],
            "user_input": task["user_input"],
            "category": category,
            "difficulty": judge.get("difficulty", difficulty),
            "skills_tested": task.get("skills_tested", []),
            "expected_answer_type": task.get("expected_answer_type", "text"),
            "answer": ans,
            "judge": judge,
            "score_0_10": score,
            "ema_global": global_ema,
            "ema_category": ema_by_cat[category].ema,
            "heuristic_max_similarity": max_sim,
        }

        store.append_benchmark_item(item)

        # add to evolving pool
        task_pool.append(task)
        novelty.add(qtext)

        if accepted % 10 == 0:
            print(f"[accepted {accepted}/{args.accept_target}] score={score:.1f} ema={global_ema:.2f}")

    print("\nDone.")
    print(f"Accepted: {accepted}")
    print(f"Attempts: {attempts}")


def build_argparser() -> argparse.ArgumentParser:
    """CLI arguments."""
    p = argparse.ArgumentParser(description="Self-evolving benchmark generator")
    p.add_argument("--accept_target", type=int, default=50)
    p.add_argument("--max_attempts", type=int, default=400)
    p.add_argument("--min_score", type=float, default=6.5)
    p.add_argument("--ema_alpha", type=float, default=0.1)
    p.add_argument("--ema_init", type=float, default=None)
    p.add_argument("--sim_threshold", type=float, default=0.70)
    p.add_argument("--demo_k", type=int, default=4)
    p.add_argument("--gen_temp", type=float, default=0.9)
    p.add_argument("--ans_temp", type=float, default=0.3)
    p.add_argument("--judge_temp", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--benchmark_path", type=str, default="data/benchmark.jsonl")
    p.add_argument("--runlog_path", type=str, default="data/run_logs.jsonl")
    return p


def main() -> None:
    """Program entry."""
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
