"""
Research tool used by the Researcher node.

Calls the Tavily Search API (https://tavily.com) -- a search API purpose
-built for LLM/agent use: results come back as clean, already-summarized
snippets instead of raw HTML, so no scraping/parsing step is needed here.

Point `web_search` at your own RAG pipeline (vector store retrieval +
optional reranking) instead of a web search if you'd rather -- the
Researcher node only cares that this function takes a query string and
returns text, so nothing else in the project needs to change.

Keeping this as a plain Python function (not a LangChain @tool) is a
deliberate simplification: the Researcher node calls it directly rather
than going through an LLM tool-calling loop, since the goal here is to
learn the *multi-agent graph* pattern first. See the README for how you'd
upgrade this to a real bind_tools()-based tool-calling researcher later.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# Mirrors config.py: safe to call even with no .env file present, and makes
# TAVILY_API_KEY available whether this module is imported via main.py or
# directly (e.g. in a notebook or test).
load_dotenv()

MAX_RESULTS = int(os.environ.get("TAVILY_MAX_RESULTS", "5"))

_client = None


def _get_client():
    """Lazily construct and cache the Tavily client.

    Lazy + cached so `python main.py --help`-style usage doesn't require the
    package to be installed or the key to be set just to import this module,
    while still only constructing the client once per process.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: TAVILY_API_KEY environment variable is not set.\n"
            "This project uses Tavily (https://tavily.com) for web search.\n"
            "Get a free API key (1000 searches/month) from "
            "https://app.tavily.com and then run:\n"
            "  export TAVILY_API_KEY='your-key-here'\n"
            "(To use a different search provider, edit tools/search.py.)"
        )

    from tavily import TavilyClient

    _client = TavilyClient(api_key=api_key)
    return _client


def web_search(query: str) -> str:
    """Run a real web search for `query` and return formatted snippets.

    Returns Tavily's own generated answer (when available) followed by the
    top result snippets, each labeled with its source URL so the Researcher
    LLM can attribute findings and the Writer can cite sources.
    """
    client = _get_client()

    try:
        response = client.search(
            query=query,
            max_results=MAX_RESULTS,
            include_answer=True,
        )
    except Exception as exc:  # network error, invalid key, rate limit, etc.
        return f"[SEARCH ERROR for query {query!r}: {exc}]"

    results = response.get("results", [])
    if not results:
        return f"[No search results found for query: {query!r}]"

    parts = []
    answer = response.get("answer")
    if answer:
        parts.append(f"Summary: {answer}")

    for i, result in enumerate(results, start=1):
        title = result.get("title", "untitled")
        url = result.get("url", "")
        content = result.get("content", "")
        parts.append(f"[{i}] {title} ({url})\n{content}")

    return "\n\n".join(parts)
