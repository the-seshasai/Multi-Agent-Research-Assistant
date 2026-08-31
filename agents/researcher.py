"""
Researcher node.

Responsibility: research ONE subtask and produce a findings note for it.
This node is invoked once per subtask, in parallel, via LangGraph's `Send`
mechanism (see graph.py's `route_to_researchers` / `route_after_critic`).
Because of that, this node's input is NOT the full ResearchState -- it's
just the small payload each `Send` carries: {"subtask": ..., "critique": ...}.
"""

from __future__ import annotations

from config import extract_text, get_llm
from tools.search import web_search

RESEARCHER_SYSTEM_PROMPT = """You are a research assistant. You will be \
given ONE research subtask and some raw search results about it. Write a \
concise, factual summary of findings for this subtask, using only the \
information in the search results provided.

If critique/feedback from a previous review round is included, make sure \
your findings directly address that feedback (e.g. add missing detail, \
resolve a flagged contradiction)."""


def researcher_node(state: dict) -> dict:
    """Research a single subtask and return a partial state update.

    `state` here is the Send() payload, not the full ResearchState:
      - state["subtask"]: the one subtask this branch is responsible for
      - state["critique"]: prior critic feedback (None on the first pass)

    Returns {"research_notes": [note]} -- a *partial* update to the shared
    `research_notes` list. Because multiple researcher branches run in
    parallel and all write to this same key, LangGraph combines their
    individual single-item lists using the `merge_research_notes` reducer
    defined in state.py (this is the fan-in half of fan-out/fan-in).
    """
    subtask = state["subtask"]
    critique = state.get("critique")

    print(f"[researcher] researching subtask: {subtask!r}")
    if critique:
        print(f"[researcher]   incorporating critic feedback: {critique!r}")

    # Step 1: call the search tool (Tavily) to get raw source material.
    raw_results = web_search(subtask)

    # Step 2: ask the LLM to turn raw results into a concise findings note.
    llm = get_llm()
    human_message = f"Subtask: {subtask}\n\nSearch results:\n{raw_results}"
    if critique:
        human_message += f"\n\nPrevious reviewer feedback to address:\n{critique}"

    response = llm.invoke(
        [
            ("system", RESEARCHER_SYSTEM_PROMPT),
            ("human", human_message),
        ]
    )

    findings = extract_text(response)
    print(f"[researcher] done with subtask: {subtask!r} ({len(findings)} chars)")

    note = {"subtask": subtask, "findings": findings}
    return {"research_notes": [note]}
