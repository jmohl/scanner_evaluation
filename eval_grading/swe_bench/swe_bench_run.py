"""
Script for running SWE-Bench via inspect evals.

SWE-Bench details:
- Tests ability to solve real world software engineering issues from 12 python github repositories
- Languages: Python
- There are two built in datasets, or can point to other swe-bench datasets.
    - SWE-bench defaults to 'princeton-nlp/SWE-bench_Verified', this is the 500 verified swe bench dataset
    - SWEBench-verified-mini is a subset of SWEBench-verified that uses 50 instead of 500 datapoints, requires 5GB instead of 130GB of storage and has approximately the same distribution of performance, test pass rates and difficulty as the original dataset.

"""
from pathlib import Path

from inspect_ai import eval
from inspect_ai.solver import basic_agent
from inspect_ai.tool import bash, python, text_editor
from inspect_evals.swe_bench import swe_bench, swe_bench_verified_mini

LOG_DIR = str(Path(__file__).parent / "eval-logs")


# Run the eval
# Note swe bench verified can be pretty slow and consumes ~130gb of storage for the whole thing
# can speed it up with more sandboxes, subprocesses, connections, etc. Recommended to not go over 32 docker containers, and this is system dependent

#note these parameters are mostly for cost control, and to minimize the size of the resulting transcript. They may impact pass rate though. 
task = swe_bench_verified_mini(
    token_limit=1500000, # there are not per-mode token limits, so best to use an average of input/output probably.
    max_messages=100 
)

eval(
    task, 
    model="openai/gpt-5.4-2026-03-05",
    log_dir=LOG_DIR,
    #limit=0, 
    max_sandboxes=8,
    max_subprocesses=16,
    max_connections=10
) 
#Model options
# gpt-5.4-2026-03-05
# gpt-5-mini-2025-08-07
# gpt-5-nano-2025-08-07

# meant to run inside the swe_bench folder as the working directory
#run with [uv run python swe_bench_run.py]
#then use [inspect view --log-dir "./eval-logs"] to see results


