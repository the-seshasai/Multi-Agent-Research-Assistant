# Multi-Agent Research Assistant

A small LangGraph project built to learn multi-agent orchestration
concepts: shared state, fan-out/fan-in parallelism, and conditional
retry loops. It runs a 4-agent pipeline (Planner -> Researcher(s) ->
Critic -> Writer) over a single research question.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set:
#   GOOGLE_API_KEY (from https://aistudio.google.com/apikey)
#   TAVILY_API_KEY (from https://app.tavily.com, free tier: 1000 searches/month)

python main.py "What are the tradeoffs of microservices vs monoliths?"
```

If `GOOGLE_API_KEY` or `TAVILY_API_KEY` isn't set, `config.py` /
`tools/search.py` exit immediately with a clear message instead of a stack
trace.

To swap models, either set `RESEARCH_ASSISTANT_MODEL` in `.env`, or edit
`config.py:get_llm()` to construct a different LangChain chat model
entirely (e.g. `ChatOpenAI`, `ChatAnthropic`) -- every agent calls
`get_llm()` rather than constructing its own client, so that's the only
place you'd need to change.

## Graph structure

```
START -> planner --(fan-out: Send per subtask)--> researcher (parallel)
                                                        |
                                                (fan-in via reducer)
                                                        v
                                                     critic
                                                    /       \
                                          (approved /      \ (feedback,
                                        or cap hit)          under cap)
                                                /               \
                                             writer          researcher (retry, parallel)
                                               |                  ^
                                               v                  |
                                              END          (loops back to critic)
```

### Why a graph, not a linear chain?

A plain function-call chain (`planner() -> researcher() -> critic() ->
writer()`) can't express two things this project needed:

1. **Parallel branches that later merge** (fan-out/fan-in) -- running N
   researchers concurrently and then continuing only once all of them are
   done.
2. **A conditional loop with a cap** -- the Critic can send work backwards
   to the Researcher, and that decision has to be made at runtime based on
   state, not hardcoded in the call order.

LangGraph models the pipeline as a `StateGraph`: a set of nodes (plain
Python functions) connected by edges, where edges can be conditional
(a router function decides where to go next based on the current state).

## Shared state (`state.py`)

Every node reads and writes a single shared `ResearchState` (a
`TypedDict`), rather than nodes calling each other directly with
arguments. This is the core mental shift from normal function
composition: **nodes never call each other** -- they only read/write a
common state object, and the graph decides what runs next based on edges
and (for conditional edges) the current state.

| Field | Written by | Read by | Why it's shared, not local |
|---|---|---|---|
| `query` | entry point | planner, critic, writer | Every stage needs to check its output against the *original* intent, not just its immediate input. |
| `subtasks` | planner | researcher (fan-out), critic | The fan-out step iterates over this to spawn one parallel branch per subtask. |
| `research_notes` | researcher (parallel writes) | critic, writer | Written concurrently by parallel branches, so it needs a **reducer** (see below) to merge safely; downstream nodes need the full collection. |
| `critique` | critic | researcher (on retry), router | Must survive the loop back to researcher (which needs to know *what* to fix) and is inspected by the conditional edge to decide whether to loop. |
| `iteration_count` | critic | router | The conditional edge function runs as a separate step from any node and needs this to enforce the retry cap. |
| `final_report` | writer | caller (`main.py`) | The end product returned once the graph finishes. |

### The reducer: how parallel writes merge

When N researcher branches run in parallel, each returns its own partial
update like `{"research_notes": [one_note]}`. Without special handling,
LangGraph would see N concurrent writes to the same key in the same step
and raise an error -- it doesn't know whether you meant "replace" or
"combine". `state.py` annotates the field:

```python
research_notes: Annotated[list[ResearchNote], merge_research_notes]
```

`merge_research_notes` is just `lambda left, right: left + right`. This
tells LangGraph: "whenever multiple writes land on `research_notes` in the
same step, concatenate them." That's the fan-in half of fan-out/fan-in --
you don't manually collect results from parallel branches, the reducer
does it as part of the state update.

One consequence worth knowing: because the reducer is append-only,
`research_notes` accumulates across *every* pass through the researcher
node, including retries. A rejected-then-retried subtask ends up with two
entries, not a replacement. Rather than adding a more complex "replace"
reducer, the Critic and Writer both call `state.latest_notes_by_subtask()`
to collapse the list down to the most recent note per subtask before
using it. This is a deliberate simplification for a learning project --
worth knowing as a "what would you do differently" talking point.

## The nodes (`agents/`)

- **Planner** (`agents/planner.py`) -- takes `query`, asks the LLM (via
  `with_structured_output`, so we get a validated list back instead of
  parsing free text) to break it into 2-4 subtasks, writes `subtasks`.
- **Researcher** (`agents/researcher.py`) -- runs once per subtask,
  in parallel. Calls a research tool (`tools/search.py:web_search`, backed
  by the Tavily search API -- swap it for a different provider or your own
  RAG pipeline without touching any other file) and asks the LLM to
  summarize findings. If retried, incorporates the Critic's feedback into
  its summary. Appends one `ResearchNote` to `research_notes`.
- **Critic** (`agents/critic.py`) -- reviews the deduped notes against the
  original `query` for gaps, contradictions, or shallow findings. Returns
  structured `approved: bool` + `feedback: str`.
- **Writer** (`agents/writer.py`) -- once approved, synthesizes the
  deduped notes into a final markdown report.

## The fan-out/fan-in mechanism (`graph.py`)

Fan-out is implemented with LangGraph's `Send` primitive rather than a
normal conditional edge:

```python
def route_to_researchers(state: ResearchState) -> list[Send]:
    return [
        Send("researcher", {"subtask": subtask, "critique": state.get("critique")})
        for subtask in state["subtasks"]
    ]
```

Instead of returning the *name* of the next node (like a normal
conditional edge would), this returns a list of `Send` objects. Each
`Send(node_name, payload)` schedules one independent invocation of that
node with its own custom input -- LangGraph runs all of them within the
same "superstep" (i.e. concurrently). This is why `researcher_node`'s
input isn't the full `ResearchState` -- it's just the small payload
(`{"subtask": ..., "critique": ...}`) that `Send` handed it.

Fan-in isn't a separate node or function -- it falls out of the reducer.
Every `Send("researcher", ...)` targets the same node, and once *all* of
them finish, LangGraph proceeds along `researcher`'s normal outgoing edge
(`researcher -> critic`) with the merged state.

## The conditional critic loop

```python
def route_after_critic(state: ResearchState) -> str | list[Send]:
    if not state.get("critique"):
        return "writer"
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "writer"          # give up gracefully, don't loop forever
    return [Send("researcher", {...}) for subtask in state["subtasks"]]
```

This function is registered with `add_conditional_edges("critic", ...)`.
After the Critic node runs, LangGraph calls this router with the current
state to decide what happens next:

- **Approved** (`critique` is falsy) -> go straight to `writer`.
- **Rejected, but under the cap** -> re-fan-out to `researcher` with the
  critique attached to every subtask's payload, so each researcher knows
  what to fix. This goes back through `researcher -> critic` again.
- **Rejected, but `iteration_count >= MAX_ITERATIONS` (3)** -> proceed to
  `writer` anyway with whatever research exists, rather than looping
  forever. This is the safety valve that prevents an infinite
  critique/revise cycle if the Critic is never satisfied.

Note the retry re-runs *all* subtasks, not just weak ones -- a
simplification that keeps the loop easy to reason about (every researcher
sees the same feedback) at the cost of some redundant LLM calls on retry.
A more advanced version could have the Critic name specific subtasks to
redo.

## Tracing / visibility

Every node prints a `[node_name] ...` line as it runs (see `agents/*.py`
and `graph.py`), so running `main.py` shows the full sequence of
handoffs, including fan-out ("fanning out to N parallel researcher(s)")
and loop decisions ("critic requested revisions -> re-fanning out").
This is deliberately left as plain `print()` rather than a logging
framework, since the goal is watching state move through the graph, not
production-grade observability. For deeper inspection (e.g. viewing the
exact state at each step), LangGraph also supports `.stream()` instead of
`.invoke()`, and integrates with LangSmith tracing if you set
`LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` -- not set up here,
but a natural next step.

## Project layout

```
state.py              # ResearchState TypedDict + reducer + dedup helper
config.py              # get_llm(): swappable model, .env loading, missing-key error
graph.py               # StateGraph wiring: nodes, fan-out/fan-in, conditional loop
main.py                 # CLI entry point
agents/
  planner.py
  researcher.py
  critic.py
  writer.py
tools/
  search.py            # web_search(query) -> str, backed by Tavily; swap for another provider/RAG
```

## Extending this

- **Different search provider / RAG**: replace `tools/search.py:web_search`
  with a call to a different search API (SerpAPI, Bing, ...) or your own
  RAG retrieval function. Signature (`str -> str`) is all that matters.
- **Tool-calling researcher**: currently the Researcher calls the search
  tool directly in Python, not via LLM tool-calling. To make the LLM
  decide *when* and *how* to search, bind the tool with
  `llm.bind_tools([web_search])` and add a small tool-execution loop.
- **LangSmith tracing**: set `LANGCHAIN_TRACING_V2=true` +
  `LANGCHAIN_API_KEY` in `.env` to get a full visual trace of every node
  call, including exact prompts/responses, in the LangSmith UI.
- **Per-subtask retry**: have the Critic name which specific subtasks
  need rework, and only re-fan-out to those, instead of retrying
  everything.
