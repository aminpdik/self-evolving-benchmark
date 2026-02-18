"""Prompt templates for generation, answering, and judging.

These functions build the exact text sent to the OpenAI-compatible endpoint for each stage."""


from __future__ import annotations

from typing import Dict, List,Any

def system_prompt() -> str:
    """Return the system prompt used for all endpoint calls.

Returns:
    A string describing general behavior constraints (follow instructions, emit strict JSON when requested)."""

    return (
        "You are a careful research assistant. "
        "Follow instructions precisely. "
        "When asked for JSON, output ONLY valid JSON with the required fields."
    )



def self_instruct_new_task_prompt(
    demos: List[Dict[str, Any]],
    target_category: str,
    target_difficulty: str,
) -> str:
    """Build the prompt for generating one new task (Self-Instruct).

Args:
    demos: Demonstration tasks sampled from the task pool.
    target_category: Required category for the new task.
    target_difficulty: Required difficulty for the new task.

Returns:
    A long prompt string instructing the model to output EXACTLY ONE task as strict JSON.

How it works:
    Formats the demos into a readable block, then injects them into a template that:
    - defines instruction vs user_input responsibilities,
    - enforces non-empty user_input,
    - enforces category/difficulty constraints,
    - specifies the JSON schema."""

    demo_lines = []
    for i, d in enumerate(demos, 1):
        demo_lines.append(
            f"Task {i}:\n"
            f"Instruction: {d.get('instruction')}\n"
            f"User input: {d.get('user_input')}\n"
            f"Category: {d.get('category')}\n"
            f"Difficulty: {d.get('difficulty')}\n"
        )

    demos_text = "\n".join(demo_lines)

    prompt = f"""
You are generating benchmark tasks. Create EXACTLY ONE NEW task.

Target category: {target_category}
Target difficulty: {target_difficulty}

CRITICAL DESIGN:
- "instruction" must be a reusable TASK TEMPLATE.
  It describes what to do in general.
  It should be abstract enough to apply to many inputs.
  It should NOT contain the specific instance data.

- "user_input" must contain ONE concrete instance of the task.
  It must include all data needed to solve it.
  It must explicitly ask for an answer.
  It MUST be a NON-EMPTY STRING.
  Never use null. Never use "".

Example:
instruction: "Write a Python function that computes the sum of numbers satisfying a condition."
user_input: "Write the function and compute the sum of all multiples of 3 between 1 and 50."

Another example:
instruction: "Explain how a classification model can be designed."
user_input: "Explain how deep learning can be used for image classification, including preprocessing, model choice, and evaluation."

BAD EXAMPLES (do NOT do this):
- instruction contains the full question or numbers/data
- user_input is null
- user_input is empty

CONSTRAINTS:
- The new task must be novel and not a minor rewording.
- Avoid trivial variations.
- The task must be solvable without external tools.
- category MUST be exactly "{target_category}".
- difficulty MUST be exactly "{target_difficulty}".
- Output STRICT JSON only.

JSON schema:
{{
  "instruction": string,
  "user_input": string,
  "category": string,
  "difficulty": "easy"|"medium"|"hard",
  "skills_tested": [string, ...],
  "expected_answer_type": string
}}

Here are example tasks (do NOT copy them):
{demos_text}

Now output ONE new task as STRICT JSON only.
""".strip()

    return prompt



def answer_prompt(instruction: str, user_input: str) -> str:
    """Build the prompt used by the Answerer.

Args:
    instruction: Task template.
    user_input: Concrete instance to solve.

Returns:
    Prompt string combining instruction and user input with a directive to answer clearly."""

    return (
        f"Instruction:\n{instruction}\n\n"
        f"User Input:\n{user_input}\n\n"
        "Answer clearly and correctly. If a structured format is implied, follow it."
    )


def judge_prompt(
    instruction: str,
    user_input: str,
    answer: str,
    similar_examples: List[str],
) -> str:
    """Build the prompt used by the Evaluator (LLM-as-judge).

Args:
    instruction: Task template.
    user_input: Concrete instance.
    answer: Produced answer to score.
    similar_examples: Most similar existing question texts, used to judge novelty.

Returns:
    Prompt string that instructs the model to return STRICT JSON with score, novelty flag, tags, and rationale."""
    sim_block = "\n\n".join([f"- {s}" for s in similar_examples]) if similar_examples else "- (none)"

    return (
        "You are an evaluator for a benchmark item. Score the ANSWER quality and also decide if the QUESTION is novel.\n\n"
        "You MUST output STRICT JSON only.\n\n"
        "Scoring rubric (0-10):\n"
        "- 9-10: correct, complete, clear, follows instruction perfectly.\n"
        "- 7-8: mostly correct, minor issues or small gaps.\n"
        "- 5-6: partially correct, notable mistakes or missing key parts.\n"
        "- 3-4: largely incorrect, confused, or ignores instruction.\n"
        "- 0-2: nonsense, refusal without reason, or totally wrong.\n\n"
        "Novelty rubric:\n"
        "- novel=true if this question is meaningfully different from existing ones.\n"
        "- novel=false if it's a rephrase or very close in concept/structure.\n\n"
        "Failure tags (choose any that apply):\n"
        "math_error, logical_gap, hallucination, incomplete, format_violation, ambiguity, shallow_reasoning\n\n"
        "Input:\n"
        f"Instruction: {instruction}\n\n"
        f"User Input: {user_input}\n\n"
        f"Answer: {answer}\n\n"
        "Most similar existing questions (for novelty comparison):\n"
        f"{sim_block}\n\n"
        "Output JSON schema:\n"
        "{\n"
        '  "score_0_10": number,\n'
        '  "novel": boolean,\n'
        '  "difficulty": "easy"|"medium"|"hard",\n'
        '  "tags": [string, ...],\n'
        '  "rationale": string\n'
        "}\n\n"
        "Return ONLY JSON:"
    )