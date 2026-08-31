"""
Wires the Planner / Researcher / Critic / Writer nodes into a LangGraph
StateGraph.

Graph shape:

    Planner --(fan-out via Send)--> Researcher (x N, parallel)
                                          |
                                    (fan-in via reducer)
                                          v
                                       Critic --(approved)--> Writer --> END
                                          |
                                     (feedback, under cap)
                                          |
                                          v
                                     Researcher (retry, parallel again)

Two LangGraph mechanisms make this work, and they're the main things worth
understanding here:

1. Fan-out with `Send`: the edge function `route_to_researchers` returns a
   list of `Send("researcher", {...})` objects instead of a plain string.
   Each `Send` schedules one independent execution of the researcher node
   with its own payload (one subtask each). LangGraph runs all of them in
   the same "superstep" (i.e. in parallel).

2. Fan-in with a reducer: all those parallel researcher branches write to
   the same `research_notes` state key. state.py annotates that field with
   a reducer function (`merge_research_notes`) so LangGraph knows to
   concatenate the parallel writes instead of raising a conflict error.
   Once every branch finishes, execution proceeds to the next node
   (`critic`) with the merged state -- that's the fan-in.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.critic import critic_node
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.writer import writer_node
from state import ResearchState

MAX_ITERATIONS = 3


def route_to_researchers(state: ResearchState) -> list[Send]:
    """Fan-out edge: turn N subtasks into N parallel researcher_node calls.

    This is a conditional edge function used with `add_conditional_edges`,
    but instead of returning the *name* of the next node, it returns a list
    of `Send` objects. Each `Send("researcher", payload)` tells LangGraph
    "invoke the node registered as 'researcher' with this exact payload",
    independently of the others. This is how LangGraph implements
    fan-out/fan-in: you don't manually spawn threads, you just describe
    the branches and the framework schedules them in parallel.

    We include `critique` in each payload so that on a retry loop, every
    researcher branch knows what the Critic wants fixed -- not just that
    something needs fixing.
    """
    print(f"[graph] fanning out to {len(state['subtasks'])} parallel researcher(s)")
    return [
        Send("researcher", {"subtask": subtask, "critique": state.get("critique")})
        for subtask in state["subtasks"]
    ]


def route_after_critic(state: ResearchState) -> str | list[Send]:
    """Conditional edge: decide whether to loop back to research or move on.

    This is the crux of the "critic loop": if the Critic approved the
    research (critique is falsy), go to the Writer. Otherwise, loop back
    to the Researcher -- but only if we haven't hit MAX_ITERATIONS, so a
    stubborn Critic can't cause an infinite loop. If the cap is hit, we
    proceed to the Writer anyway with whatever research we have, rather
    than looping forever or crashing.

    On retry we re-fan-out over ALL subtasks (not just the ones that were
    weak) with the critique attached to each payload. This is a
    simplification -- a more advanced version could have the Critic point
    at specific subtasks to redo -- but it keeps the loop easy to reason
    about for a learning project, and every researcher gets the same
    feedback about what the Critic found lacking.
    """
    if not state.get("critique"):
        print("[graph] critic approved -> writer")
        return "writer"

    if state["iteration_count"] >= MAX_ITERATIONS:
        print(
            f"[graph] hit MAX_ITERATIONS={MAX_ITERATIONS} without approval "
            "-> writer anyway"
        )
        return "writer"

    print(
        f"[graph] critic requested revisions (iteration {state['iteration_count']}) "
        "-> re-fanning out to researcher"
    )
    return [
        Send("researcher", {"subtask": subtask, "critique": state["critique"]})
        for subtask in state["subtasks"]
    ]


def build_graph():
    """Construct and compile the StateGraph. Called once by main.py."""
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "planner")

    # Fan-out: planner's subtasks -> N parallel researcher invocations.
    graph.add_conditional_edges("planner", route_to_researchers, ["researcher"])

    # Fan-in happens implicitly: every researcher branch's Send() targets
    # the *same* node ("researcher"), and once ALL of them finish, LangGraph
    # advances along researcher's normal outgoing edge below. The merging
    # of their `research_notes` writes is handled by the reducer in state.py.
    graph.add_edge("researcher", "critic")

    # Conditional loop: critic -> writer (approved/cap hit) or -> researcher (retry).
    graph.add_conditional_edges(
        "critic", route_after_critic, ["writer", "researcher"]
    )

    graph.add_edge("writer", END)

    return graph.compile()
