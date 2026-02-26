"""
Script for v.0 scanners. This is mostly testing the workflow.

Starting with T.2 (tool use failrues prevent success) and O.h.1 (answer format not clearly specified).
Initially running the scanners on all messages, then will try targeting specific messages
TODO: would like this to have access to the gold standard answers, particularly for o.h.1

"""

import re

from pydantic import BaseModel, Field
from shortuuid import uuid

from inspect_scout import (
    Reference, 
    Result, 
    Scanner, 
    Transcript, 
    llm_scanner,
    scanner, 
    tool_callers,
    transcripts_from,
    columns as c
)

# transcripts = transcripts_from("./eval-logs")

## ----------- Scanner implementations ---------

# ----Impossible_task----
# a pilot scanner to check for general impossibility of the task. This is intentionally quite open ended
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


# ---- answer_format -----
# check for correct specification of answer format that may lead to false negatives
# This is for criteria O.h.1
# To do this, the scanner will need access to:
# 1. The prompt (both system and user)
# 2. The final answer output
# 3. The gold standard/expected answer

# Custom template: omits the default "You are an expert in LLM transcript
# analysis..." preamble and {{ messages }} block so only our constructed
# context is sent to the model.
ANSWER_FORMAT_TEMPLATE = """\
{{ question }}

Your answer should include an explanation of your assessment. It should include the message id's (e.g. '[M2]') to clarify which message(s) you are referring to.

{{ answer_format }}"""

answer_format_prompt = """\
You are evaluating whether a scientific task's expected answer format is clearly \
and correctly specified in the prompts given to an AI agent.

Your job: Determine whether a format mismatch exists that could cause a correct \
answer to be scored as wrong. This includes issues such as:
  - Rounding or precision differences not addressed by the prompt
  - Units or scales not specified (e.g. percent vs decimal)
  - Ambiguous question wording that could elicit a differently-formatted answer
  - The prompt not specifying how many decimal places to report

Respond True if there IS a format or specification problem that could cause a \
false negative (correct answer marked wrong due to format).
Respond False if the format is clearly and unambiguously specified.

"""

@scanner(messages="all")
def answer_format() -> Scanner[Transcript]:

    async def build_question(transcript: Transcript) -> str:
        system_text = "\n".join(
            m.text for m in transcript.messages if m.role == "system"
        )
        user_text = "\n".join(
            m.text for m in transcript.messages if m.role == "user"
        )
        final_msg = transcript.messages[-1] if transcript.messages else None
        final_text = final_msg.text if final_msg else "(no final message)"
        sample_metadata = (transcript.metadata or {}).get("sample_metadata", {})
        gold_answers = sample_metadata.get("results", "(not available)")

        return (
            f"{answer_format_prompt}"
            f"--- SYSTEM PROMPT ---\n{system_text}\n\n"
            f"--- USER PROMPT ---\n{user_text}\n\n"
            f"--- FINAL SUBMISSION ---\n{final_text}\n\n"
            f"--- GOLD STANDARD ANSWERS ---\n{gold_answers}\n"
        )

    return llm_scanner(
        question=build_question,
        answer="boolean",
        template=ANSWER_FORMAT_TEMPLATE,
    )

# ---- Command_not_found -----
# This is an example scanner from the Scout documentation that looks for tool use failures
# This is currently standing in for T.2: tool errors

class CommandNotFound(BaseModel):
    message_id: str = Field(description="Message that made the tool call.")
    command: str = Field(description="The command that was not found.")
    tool: str | None = Field(description="Tool that produced the output.")

@scanner(messages="all")
def command_not_found() -> Scanner[Transcript]:

    async def scan(transcript: Transcript) -> list[Result]:

        results: list[Result] = []

        # Build a mapping from tool_call_id to assistant message
        tool_call_to_assistant = tool_callers(transcript)

        # Pattern to match "command not found" errors
        pattern = r"(\w+): line \d+: (\w+): command not found"

        # Iterate through all tool messages with tool call ids
        for message in (m for m in transcript.messages if m.role == "tool"):
         
            # skip messages with no tool_call_id
            if message.tool_call_id is None:
                continue

            # look for 'command not found'
            match = re.search(pattern, message.text)
            if match:
                # extract the command and tool name
                command = match.group(2)
                tool_name = message.function

                # find the assistant message that made this tool call
                # (skip messages with no correpsonding assistant message)
                assistant_msg, assistant_idx = tool_call_to_assistant.get(
                    message.tool_call_id, (None, 0)
                )
                if assistant_msg is None:
                    continue
                
                # append the result
                results.append(
                    Result(
                        value=CommandNotFound(
                            message_id=f"M{assistant_idx}",
                            command=command,
                            tool=tool_name,
                        ).model_dump(),
                        explanation=(
                            f"[M{assistant_idx}] Found 'command not found' "
                            f"for command {command}' in {tool_name} output"
                        ),
                        references=[Reference(
                            type="message",
                            cite=f"M{assistant_idx}",
                            id=assistant_msg.id or uuid()
                        )],
                    )
                )
               

        return results

    return scan

# example for running on a single transcript log
# scout scan scanner_pilot/scout_pilot.py -T logs/2026-02-21T19-49-50+00-00_hellaswag_KYrRj2JczeE4cTDdVChdDk.eval --model openai/gpt-5-nano
# scout view
