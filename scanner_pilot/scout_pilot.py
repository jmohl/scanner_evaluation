from inspect_scout import Scanner, Transcript, llm_scanner, scanner

@scanner(messages="all")
def impossible_task() -> Scanner[Transcript]:
    return llm_scanner(
        question="In the transcript above do you detect "
        "instances of cases where the task constains errors "
        "preventing the agent from completing it's task?",
        answer="boolean"
    )