"""Self-Instruct task generation module.

Samples a few existing tasks as demonstrations and asks an OpenAI-compatible endpoint to
produce one new, novel task in a structured JSON format."""


from __future__ import annotations

import json
import random
from typing import Any, Dict, List

from client import OpenAICompatClient
from prompts import system_prompt, self_instruct_new_task_prompt


class SelfInstructGenerator:
    """Self-Instruct task generator."""

    def __init__(self, client: OpenAICompatClient, rng_seed: int = 0):
        """Initialize the generator.

Args:
    client: OpenAI-compatible client used to call the remote endpoint.
    rng_seed: Seed for the internal RNG used when sampling demonstration tasks.

Returns:
    None.

How it works:
    Stores the client and creates a dedicated random.Random instance so task sampling is
    repeatable given the same seed."""

        self.client = client
        self.rng = random.Random(rng_seed)

    def generate_task(
        self,
        task_pool: List[Dict[str, Any]],
        target_category: str,
        target_difficulty: str,
        demo_k: int = 4,
        temperature: float = 0.9,
    ) -> Dict[str, Any]:
        """Generate a single new task using a Self-Instruct-style prompt.

Args:
    task_pool: List of existing tasks. Each task is a dict containing (at least) instruction and user_input.
    target_category: Category label to request from the generator (e.g., "math", "coding").
    target_difficulty: Difficulty label to request from the generator ("easy"|"medium"|"hard").
    demo_k: Number of demonstration tasks to include in the prompt (sampled from task_pool).
    temperature: Sampling temperature for the generation call.

Returns:
    A dict representing the newly generated task, parsed from JSON. Expected keys include:
    instruction, user_input, category, difficulty, skills_tested, expected_answer_type.

How it works:
    1) Randomly samples up to demo_k tasks from task_pool as demonstrations.
    2) Builds a generation prompt via self_instruct_new_task_prompt(...).
    3) Calls the chat completion endpoint with a system message + user prompt.
    4) Parses the model output into a Python dict (with a small robustness layer in _safe_parse_json)."""

        demos = self.rng.sample(task_pool, k=min(demo_k, len(task_pool)))
        prompt = self_instruct_new_task_prompt(demos, target_category, target_difficulty)
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ]
        raw = self.client.chat_completions(
            messages,
            temperature=temperature,
            max_tokens=700,
            response_format_json=True,
        )
        return self._safe_parse_json(raw)

    def _safe_parse_json(self, s: str) -> Dict[str, Any]:
        """Parse a JSON object from a model response string.

Args:
    s: Raw text returned by the model.

Returns:
    Parsed JSON as a Python dict.

How it works:
    Tries json.loads() directly. If that fails (e.g., extra text around the JSON),
    it extracts the substring between the first '{' and the last '}' and tries again.

Raises:
    ValueError: If no valid JSON object can be parsed."""

        s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
        raise ValueError("Failed to parse generator JSON.")