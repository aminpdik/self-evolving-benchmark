"""Task answering module.

Given an instruction and a concrete user input, calls the configured endpoint to produce an answer."""


from __future__ import annotations

from client import OpenAICompatClient
from prompts import system_prompt, answer_prompt


class Answerer:
    """Uses the endpoint to answer generated tasks."""

    def __init__(self, client: OpenAICompatClient):
        """Create an Answerer.

Args:
    client: OpenAI-compatible client used to produce answers.

Returns:
    None."""

        self.client = client

    def answer(self, instruction: str, user_input: str, temperature: float = 0.3) -> str:
        """Answer a generated benchmark task using the configured endpoint.

Args:
    instruction: The reusable task template (what to do).
    user_input: A concrete instance of the task (the data/question to solve).
    temperature: Sampling temperature for the answering call (lower = more deterministic).

Returns:
    The assistant's answer text as returned by the endpoint.

How it works:
    Builds a formatted prompt with answer_prompt(...), wraps it into system+user messages,
    and calls client.chat_completions(...)."""

        prompt = answer_prompt(instruction, user_input)
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ]
        return self.client.chat_completions(messages, temperature=temperature, max_tokens=900)