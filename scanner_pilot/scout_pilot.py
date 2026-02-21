from inspect_scout import Scanner, Transcript, llm_scanner, scanner

@scanner(messages="all")
def impossible_task() -> Scanner[Transcript]:
    return llm_scanner(
        question="In the transcript above do you detect "
        "instances of cases where the task contains errors "
        "preventing the agent from completing it's task? "
        "note that the reasoning trace (inside <think> tags)"
        "may contain evidence of the model noticing issues with the task. \n"
        "issues may include things like: 1. No available answer is correct " 
        "2. It is unclear how the model is expected to answer "
        "3. The task is confusing or poorly worded ",
        answer="boolean"
    )

# example for running on a single transcript log
# scout scan scanner_pilot/scout_pilot.py -T logs/2026-02-21T19-49-50+00-00_hellaswag_KYrRj2JczeE4cTDdVChdDk.eval --model openai/gpt-5-nano
# scout view
