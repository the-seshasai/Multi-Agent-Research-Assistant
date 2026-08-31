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