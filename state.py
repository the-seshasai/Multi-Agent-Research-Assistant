"""
Shared state schema for the multi-agent research assistant graph.

In LangGraph, every node in a StateGraph reads from and writes to a single
shared state object. This is different from a normal function call chain:
nodes don't pass arguments to each other directly, they communicate only
through this shared state dict. Understanding *why* a field lives here
(vs. being a local variable inside a node function) is the key mental
model for this project, so each field below is commented with that in mind.
"""

from __future__ import annotations

from typing import Annotated, TypedDict


class ResearchNote(TypedDict):
    """One research result, tied to a specific subtask.

    Kept as a small typed record (rather than a raw string) so the Critic
    and Writer can see which subtask a note answers, not just a blob of
    text. This matters once you have multiple parallel researchers writing
    into the same list -- you need to know who wrote what.
    """

    subtask: str
    findings: str


def merge_research_notes(
    left: list[ResearchNote], right: list[ResearchNote]
) -> list[ResearchNote]:
    """Reducer used to combine research_notes from parallel researcher branches.

    LangGraph fans the graph out into N parallel "Researcher" node
    executions (one per subtask). Each parallel branch returns its own
    partial state update for `research_notes`. Without a reducer, LangGraph
    would not know how to combine N concurrent writes to the same key and
    would raise an "InvalidUpdateError". By annotating the field with this
    function, we tell LangGraph: "when multiple branches write to
    research_notes at the same superstep, concatenate their lists together."
    This is the fan-in step of the fan-out/fan-in pattern.
    """
    return left + right


class ResearchState(TypedDict):
    """The single shared state object threaded through the whole graph.

    Every field below is *shared* state (visible to all nodes) rather than
    a local variable inside one node function, because at least one other
    node downstream needs to read it. If a value were only ever used and
    discarded inside a single node, it would just be a local variable in
    that node's function body and wouldn't belong here.
    """

    # --- Set once, by the user / entry point ---
    # The original user question. Shared because almost every node
    # (Planner, Critic, Writer) needs to re-check its output against the
    # original intent, not just against the immediate subtask.
    query: str

    # --- Written by Planner, read by Researcher (fan-out) and Critic ---
    # The list of subtasks the Planner decomposed the query into. Shared
    # because the Researcher fan-out step needs to iterate over it to spawn
    # one parallel branch per subtask, and the Critic needs it to check
    # coverage (did research answer every subtask?).
    subtasks: list[str]

    # --- Written by Researcher (parallel, fan-in via reducer), read by Critic/Writer ---
    # Accumulated findings, one ResearchNote per subtask. This uses the
    # `merge_research_notes` reducer above (Annotated[...]) because it is
    # written concurrently by multiple parallel Researcher branches -- a
    # plain field would raise a concurrent-update error. Shared because the
    # Critic needs the full collection to review coverage, and the Writer
    # needs it to synthesize the final report.
    research_notes: Annotated[list[ResearchNote], merge_research_notes]

    # --- Written by Critic, read by Researcher (on retry) and the router ---
    # None/empty string means "approved". A non-empty string is specific
    # feedback describing gaps/contradictions. Shared because it must
    # survive the loop back to Researcher (the retry pass needs to know
    # *what* to fix) and is also what the conditional edge inspects to
    # decide whether to loop back or proceed to the Writer.
    critique: str | None

    # --- Incremented by Critic, read by the conditional router ---
    # Counts how many times we've gone through the Researcher->Critic loop.
    # Shared (not local to Critic) because the conditional-edge routing
    # function runs as a separate step and needs to see it to enforce the
    # max-iteration cap, preventing infinite critique/revise loops.
    iteration_count: int

    # --- Written by Writer, read by the entry point / caller ---
    # The final markdown report. Shared because it's the ultimate output of
    # the whole graph, returned to main.py after the graph finishes.
    final_report: str | None


def latest_notes_by_subtask(notes: list[ResearchNote]) -> list[ResearchNote]:
    """Collapse research_notes down to one (the most recent) note per subtask.

    Because `research_notes` uses the append-only `merge_research_notes`
    reducer above, it accumulates notes from every retry pass, not just the
    latest one -- e.g. two rounds over the same 2 subtasks produces 4
    entries. Both the Critic and the Writer need to judge/use only the most
    recent attempt at each subtask, so they both call this helper rather
    than duplicating the dedup logic. Keeps first-seen subtask ordering.
    """
    latest: dict[str, ResearchNote] = {}
    order: list[str] = []
    for note in notes:
        if note["subtask"] not in latest:
            order.append(note["subtask"])
        latest[note["subtask"]] = note
    return [latest[subtask] for subtask in order]
