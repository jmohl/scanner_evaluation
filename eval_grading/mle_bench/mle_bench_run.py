"""
Script for running MLE-Bench via inspect evals.

MLE-Bench details:
- The dataset is sourced from Kaggle competitions. MLE-bench includes:
    - 75 curated ML engineering competitions from Kaggle
    - Tasks covering tabular, vision, NLP, and other domains
    - Public and private test sets with standardized scoring.
- The full dataset is ~3.3TB in size and will take ~2 days to download and prepare from scratch. The lite dataset (split = "low.txt") is ~158GB, but then it requires >400GB with unzipped files.
- The scorer returns boolean values for three criteria:
    - valid_submission: Whether the agent produced a valid submission
    - above_median: Whether the submission scored above the median
    - any_medal: Whether the submission achieved any medal (bronze, silver, or gold)
"""

from inspect_ai import eval
from inspect_evals.mle_bench import mle_bench
from pathlib import Path


LOG_DIR = str(Path(__file__).parent / "eval-logs")


task = mle_bench(
    split = 'low.txt',
)

eval(
    task, 
    model="openai/gpt-5-mini-2025-08-07",
    log_dir=LOG_DIR,
) 
#Model options
# gpt-5.4-2026-03-05
# gpt-5-mini-2025-08-07
# gpt-5-nano-2025-08-07

# meant to run inside the mle_bench folder as the working directory
#run with [uv run python mle_bench_run.py]
#then use [inspect view --log-dir "./eval-logs"] to see results

