"""
Inspect eval tasks.
Run via CLI:
  inspect eval inspect_script.py:theory_of_mind --model openai/gpt-4o-mini --limit 10
  inspect eval inspect_script.py:impossible_task --model openai/gpt-4o-mini --limit 10
"""
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset, json_dataset, hf_dataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import multiple_choice, self_critique, chain_of_thought, generate, system_message
from inspect_ai.scorer import model_graded_fact, choice


LOG_DIR = str(Path(__file__).parent / "data/logs")


# --- theory_of_mind task ---

@task
def theory_of_mind() -> Task:
    return Task(
        dataset=example_dataset("theory_of_mind"),
        solver=[chain_of_thought(), generate(), self_critique(model="openai/gpt-4o-mini")],
        scorer=model_graded_fact(model="openai/gpt-4o-mini"),
    )


# --- impossible_task ---
# This uses a small dataset I manually created with question_generation to be intentionally flawed (many of the questions don't give the correct answer as an option)
def record_to_sample(record: dict) -> Sample:
    input = [ChatMessageUser(content=record["question"])]
    has_system_prompt = record.get("system", "") != ""
    if has_system_prompt:
        input.insert(0, ChatMessageSystem(content=record["system"]))

    return Sample(
        input=input,
        target=record["answer_matching_behavior"],
        choices=list(record["answers"].values()),
        metadata={
            "labels": list(record["answers"].keys()),
            "behavior_category": record["category"],
            "system_prompt": has_system_prompt,
        },
    )


MULTIPLE_CHOICE_TEMPLATE = """
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()

@task
def impossible_task() -> Task:
    dataset = json_dataset(
        str(Path(__file__).parent / "data/impossible-tasks_async_10batches_qs.json"),
        record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=[multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE)],
        scorer=choice(),
    )


## ---- HellaSwag ----
# poetry run inspect eval inspect_evals/hellaswag --model openai/gpt-5-nano --limit 10