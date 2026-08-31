"""
Critic node.

Responsibility: review the combined research notes against the original
query and either approve them or return specific feedback describing
gaps/contradictions/missing coverage. This is what drives the conditional
retry loop in graph.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config import get_llm
from state import ResearchState, latest_notes_by_subtask

CRITIC_SYSTEM_PROMPT = """You are a critical reviewer of research notes. \
You will be given the user's original research question and a set of \
research notes (one per subtask) intended to answer it.

Decide whether the notes, taken together, adequately answer the original \
question. Check for:
- Gaps: aspects of the question left uncovered.
- Contradictions: notes that disagree with each other.
- Shallow findings: notes that are vague or say little of substance.

If everything looks adequate, set approved=true and leave feedback empty.
If not, set approved=false and give concise, actionable feedback that a \
researcher could use to improve the notes on the next pass."""


class CritiqueResult(BaseModel):
    approved: bool = Field(description="True if the research is adequate")
    feedback: str = Field(
        default="",
        description="Specific, actionable feedback if not approved; empty if approved",
    )


def critic_node(state: ResearchState) -> dict:
    """Review deduped research notes and decide approve vs. request revisions.

    Returns partial updates to `critique` (None if approved, else feedback
    text) and `iteration_count` (always incremented, so the retry cap in
    graph.py's route_after_critic can be enforced regardless of outcome).
    """
    notes = latest_notes_by_subtask(state["research_notes"])
    print(f"[critic] reviewing {len(notes)} notes (iteration {state['iteration_count']})")

    notes_text = "\n\n".join(
        f"## Subtask: {note['subtask']}\n{note['findings']}" for note in notes
    )

    llm = get_llm().with_structured_output(CritiqueResult)
    result: CritiqueResult = llm.invoke(
        [
            ("system", CRITIC_SYSTEM_PROMPT),
            (
                "human",
                f"Original question: {state['query']}\n\nResearch notes:\n{notes_text}",
            ),
        ]
    )

    if result.approved:
        print("[critic] verdict: approved")
        return {"critique": None, "iteration_count": state["iteration_count"] + 1}

    print(f"[critic] verdict: needs revision -- {result.feedback!r}")
    return {
        "critique": result.feedback,
        "iteration_count": state["iteration_count"] + 1,
    }
