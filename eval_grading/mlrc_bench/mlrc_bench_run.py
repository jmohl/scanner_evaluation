"""
Script for running MLRC-Bench via inspect evals.

MLRC-Bench details:
- The purpose of the benchmark is to evaluate the ability of LLM-based research agents to propose and implement novel methods. 
- Drawing on tasks from recent ML conference competitions, MLRC-BENCH enables the evaluation of research agents' ideas compared 
to a reliable baseline method and the top human solution.
- 6 tasks (ML conference competitions): llm-merging', 'backdoor-trigger', 'temporal-action-loc', 'machine-unlearning', 'meta-learning', 'product-rec'.
- GPU requirements: 
    - At least 16GB VRAM: 'temporal-action-loc', 'machine-unlearning', 'meta-learning', 'product-rec'.
    - At least 48GB VRAM: 'backdoor-trigger'.
    - At least ~49GB VRAM: 'llm-merging'.
"""

from inspect_ai import eval
from inspect_evals.mlrc_bench import mlrc_bench
from pathlib import Path


LOG_DIR = str(Path(__file__).parent / "eval-logs")


task = mlrc_bench()

eval(
    task, 
    model="openai/gpt-5-mini-2025-08-07",
    log_dir=LOG_DIR,
) 
#Model options
# gpt-5.4-2026-03-05
# gpt-5-mini-2025-08-07
# gpt-5-nano-2025-08-07

# meant to run inside the mlrc_bench folder as the working directory
#run with [uv run python mlrc_bench_run.py]
#then use [inspect view --log-dir "./eval-logs"] to see results

