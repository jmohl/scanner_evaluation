"""
Script for running GAIA via inspect evals
Note: 
"""
from pathlib import Path

from inspect_ai import eval, eval_set
from inspect_evals.gaia import gaia, gaia_level1, gaia_level2, gaia_level3
from inspect_ai.solver import use_tools, generate, system_message, basic_agent
from inspect_ai.tool import bash, web_search


LOG_DIR = str(Path(__file__).parent / "eval-logs")

# Create an agent that only uses bash
agent = basic_agent(tools=[bash()])
task = gaia_level1(
    language="Python",
    limit=5,
    solver=agent,
    difficulty='easy',
    filter_out_gpu=True,
)
# See how agent performs
eval(gaia)
eval_set([gaia_level1], log_dir=LOG_DIR)
eval(task, model="openai/gpt-5-nano", log_dir=LOG_DIR)

#then use [inspect view] to see results

## Running from command line
# this is another way to do this instead of running this script. Need to pass in your options
# inspect eval inspect_evals/core_bench --model openai/gpt-5-nano --limit 1 
