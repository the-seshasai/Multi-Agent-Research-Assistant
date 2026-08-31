"""
Planner node.

Responsibility: take the raw user query and break it into 2-4 concrete
research subtasks that, together, cover the query. This is the first node
in the graph -- everything downstream (parallel researchers, critic,
writer) operates on the `subtasks` list this node produces.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config import get_llm
from state import ResearchState

PLANNER_SYSTEM_PROMPT = """You are a research planner. Given a user's \
research question, break it down into 2 to 4 focused subtasks that \
together give complete coverage of the question.

Rules:
- Each subtask should be answerable somewhat independently of the others \
(they will be researched in parallel).
- Avoid overlap between subtasks -- each should cover a distinct angle.
- Keep each subtask to a single, clear sentence.
- Do not exceed 4 subtasks or use fewer than 2."""


class SubtaskList(BaseModel):
    """Structured output schema the LLM must fill in.

    Using `with_structured_output` (rather than parsing free-form text)
    means we get a validated Python list back directly, with no brittle
    string-splitting on the LLM's response.
    """

    subtasks: list[str] = Field(
        description="2 to 4 focused research subtasks covering the query"
    )


def planner_node(state: ResearchState) -> dict:
    """Decompose the user query into 2-4 research subtasks.

    Returns a partial state update (just `subtasks`) -- LangGraph merges
    this into the shared ResearchState. We don't touch other fields like
    research_notes or critique here; that's the Researcher/Critic's job.
    """
    print(f"[planner] decomposing query: {state['query']!r}")

    llm = get_llm().with_structured_output(SubtaskList)
    result: SubtaskList = llm.invoke(
        [
            ("system", PLANNER_SYSTEM_PROMPT),
            ("human", state["query"]),
        ]
    )

    print(f"[planner] produced {len(result.subtasks)} subtasks:")
    for i, subtask in enumerate(result.subtasks, start=1):
        print(f"[planner]   {i}. {subtask}")

    return {"subtasks": result.subtasks}
