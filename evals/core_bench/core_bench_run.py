"""
Script for running CORE-Bench via inspect evals.

CORE-Bench details:
- Tests computational reproducibility of real research papers
- 270 tasks derived from 90 papers across Computer Science, Medical Sciences, and Social Sciences
- Languages: Python and R
- 3 difficulty levels:
    - Easy:   given pre-run results; model extracts answers (no code execution, no GPU needed)
    - Medium: model follows REPRODUCING.md to run Docker commands, then extracts answers
    - Hard:   model reads README, installs dependencies, runs code from scratch
- Answer types: numeric, string, list, and vision (figure-based) questions
- A task is only correct if ALL its questions are answered correctly
- Storage: up to ~15 GB for the full test split (capsules downloaded as tar.gz)

GPU note: Medium/Hard tasks may require GPU. Avoid this by using 'easy' difficulty or setting filter_out_gpu=True.
Vision note: vision is only supported on openAI models. You can pass in a different vision model using the vllm_model option if running this on claude/gemini
"""
from pathlib import Path

from inspect_ai import eval
from inspect_ai.solver import basic_agent
from inspect_ai.tool import bash, python, text_editor
from inspect_evals.core_bench import core_bench

LOG_DIR = str(Path(__file__).parent / "eval-logs")

# Create an agent that uses bash, python, and a text editor
agent = basic_agent(tools=[bash(timeout=180), python(), text_editor()])
task = core_bench(
    #language="Python",
    limit=0, #limit = 0 to run all
    solver=agent,
    difficulty='easy', #tasks are either easy, medium, or hard. medium and hard tasks may use gpu resources
    #filter_out_gpu=False, #note that tasks will fail if there is no GPU on your machine
    token_limit=1000000, #original CORE-bench had a $4 limit per task, can use this to approximate that
    max_messages=50
)

# See how agent performs
# Note I set these sandboxes and subprocesses to lower levels because I was getting instability, but this makes it run more slowly
eval(
    task, 
    model="openai/gpt-5-mini-2025-08-07",
    max_sandboxes=8,
    max_subprocesses=8,
    max_connections=20, 
    log_dir=LOG_DIR) #gpt-5.2-2025-12-11


#run with python evals/core_bench/core_bench_run.py
#then use [inspect view --log-dir "evals/core_bench/eval-logs"] to see results


