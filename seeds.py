"""Seed tasks used to bootstrap the initial task pool for Self-Instruct generation."""


from __future__ import annotations

from typing import Dict, List

SEED_TASKS: List[Dict[str, str]] = [
    {
        "instruction": "Solve the logic puzzle and explain your reasoning briefly.",
        "user_input": "Three boxes are labeled Apples, Oranges, Apples+Oranges. All labels are wrong. "                      "You may pick one fruit from one box. How do you label all boxes correctly?",
        "category": "reasoning",
        "difficulty": "medium",
        "skills_tested": ["logic", "constraint reasoning"],
        "expected_answer_type": "short explanation",
    },
    {
        "instruction": "Write a Python function that returns True if a string is a palindrome ignoring non-letters.",
        "user_input": "Implement is_pal(s). Example: 'A man, a plan, a canal: Panama' -> True",
        "category": "coding",
        "difficulty": "easy",
        "skills_tested": ["python", "string processing"],
        "expected_answer_type": "python code + brief explanation",
    },
    {
        "instruction": "Given a short scenario, propose a simple experimental design and justify it.",
        "user_input": "You suspect a new assay buffer increases signal-to-noise ratio. Propose an experiment design "                      "to test it with minimal runs.",
        "category": "science",
        "difficulty": "medium",
        "skills_tested": ["experimental design", "statistics intuition"],
        "expected_answer_type": "structured plan",
    },
    {
        "instruction": "Explain the concept in simple terms and give a small example.",
        "user_input": "What is an exponential moving average (EMA) and why use it for noisy scores?",
        "category": "ml",
        "difficulty": "easy",
        "skills_tested": ["explanation", "math intuition"],
        "expected_answer_type": "short explanation + tiny example",
    },
    {
        "instruction": "Do the calculation carefully and show key steps.",
        "user_input": "A model scored 7, then 9, then 6. With EMA alpha=0.2 and initial EMA=7, compute the final EMA.",
        "category": "math",
        "difficulty": "easy",
        "skills_tested": ["arithmetic", "ema"],
        "expected_answer_type": "step-by-step computation",
    },
    {
        "instruction": "Propose an algorithm and discuss complexity.",
        "user_input": "You have 1 million short questions. How do you detect near-duplicates efficiently?",
        "category": "systems",
        "difficulty": "hard",
        "skills_tested": ["algorithms", "scalability", "similarity"],
        "expected_answer_type": "algorithm + complexity discussion",
    },
]