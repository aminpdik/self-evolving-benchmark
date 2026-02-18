"""LLM-as-judge evaluation module.

Uses an OpenAI-compatible endpoint to score answer quality and decide whether a proposed
question is novel relative to similar existing examples."""


from __future__ import annotations

import json
from typing import Any, Dict, List

from client import OpenAICompatClient
from prompts import system_prompt, judge_prompt


class Evaluator:
    """LLM-as-judge evaluation (quality score + novelty)."""

    def __init__(self, client: OpenAICompatClient):
        """Create an Evaluator (LLM-as-judge).

Args:
    client: OpenAI-compatible client used to score answers and judge novelty.

Returns:
    None."""

        self.client = client

    def evaluate(
        self,
        instruction: str,
        user_input: str,
        answer: str,
        similar_examples: List[str],
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """Evaluate an answer and judge question novelty using an LLM-as-judge prompt.

Args:
    instruction: Task template.
    user_input: Concrete task instance.
    answer: The model-produced answer to evaluate.
    similar_examples: Short list of the most similar existing question texts (for novelty comparison).
    temperature: Sampling temperature for the judge call.

Returns:
    A dict parsed from the judge JSON output with fields like:
    score_0_10, novel, difficulty, tags, rationale.

How it works:
    1) Builds the judge prompt with judge_prompt(...), including similar_examples.
    2) Calls the endpoint and requests JSON-shaped output.
    3) Parses the response into a Python dict using _safe_parse_json."""

        prompt = judge_prompt(instruction, user_input, answer, similar_examples)
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ]
        raw = self.client.chat_completions(
            messages,
            temperature=temperature,
            max_tokens=500,
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
        raise ValueError("Failed to parse judge JSON.")