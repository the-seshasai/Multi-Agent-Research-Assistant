"""
Writer node.

Responsibility: synthesize the (approved) research notes into a final,
well-structured markdown report answering the original query. This is the
last node before END.
"""

from __future__ import annotations

from config import extract_text, get_llm
from state import ResearchState, latest_notes_by_subtask

WRITER_SYSTEM_PROMPT = """You are a technical writer. You will be given a \
research question and a set of research notes (one per subtask) that have \
already been reviewed and approved as adequate. Synthesize them into a \
single, well-structured markdown report that directly answers the \
original question.

Structure:
- A short introductory paragraph restating the question and summarizing \
the answer.
- One section per subtask/theme (use ## headings), synthesizing the \
notes in your own words -- don't just copy them verbatim.
- A brief concluding paragraph.

Output ONLY the markdown report, no preamble like "Here is the report"."""


def writer_node(state: ResearchState) -> dict:
    """Synthesize deduped, approved research notes into the final report."""
    notes = latest_notes_by_subtask(state["research_notes"])
    print(f"[writer] synthesizing final report from {len(notes)} notes")

    notes_text = "\n\n".join(
        f"## Subtask: {note['subtask']}\n{note['findings']}" for note in notes
    )

    llm = get_llm()
    response = llm.invoke(
        [
            ("system", WRITER_SYSTEM_PROMPT),
            (
                "human",
                f"Original question: {state['query']}\n\nApproved research notes:\n{notes_text}",
            ),
        ]
    )

    print("[writer] report generated")
    return {"final_report": extract_text(response)}
