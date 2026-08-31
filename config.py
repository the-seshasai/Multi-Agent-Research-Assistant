"""
Central place to configure which LLM backs the agents.

Keeping this in one module (instead of each agent constructing its own
ChatGoogleGenerativeAI) means swapping providers later -- e.g. to OpenAI or
Anthropic -- only requires editing this file, not every agents/*.py file.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# Loads variables from a .env file (if present) into os.environ. Safe to
# call even if no .env file exists -- it's then a no-op and normal
# `export`-ed environment variables still work as before. Doing this here
# (rather than in main.py) means it also takes effect for anyone importing
# config.py directly, e.g. in a notebook or test.
load_dotenv()

# Defaults are read from environment variables so you can override them
# without touching code, e.g.:
#   export RESEARCH_ASSISTANT_MODEL="gemini-1.5-pro"
MODEL_NAME = os.environ.get("RESEARCH_ASSISTANT_MODEL", "gemini-3.6-flash")
TEMPERATURE = float(os.environ.get("RESEARCH_ASSISTANT_TEMPERATURE", "0.2"))


def get_llm():
    """Construct and return the chat model used by every agent node.

    Centralizing construction here (rather than in each node) means all
    agents share the same model/temperature config, and there's exactly
    one place that needs to handle the "missing API key" case.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: GOOGLE_API_KEY environment variable is not set.\n"
            "This project uses Gemini (via langchain-google-genai) by default.\n"
            "Get a key from https://aistudio.google.com/apikey and then run:\n"
            "  export GOOGLE_API_KEY='your-key-here'\n"
            "(To use a different model/provider, edit config.py.)"
        )

    # Imported lazily so `python main.py --help`-style usage doesn't require
    # the package to be installed just to read this file, and so the error
    # above prints cleanly before we even try to import the provider SDK.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE)


def extract_text(response) -> str:
    """Pull plain text out of a chat model response.

    Newer Gemini models (e.g. gemini-3.6-flash) return `response.content` as
    a list of content blocks (e.g. `[{"type": "text", "text": "...", ...}]`)
    instead of a plain string, so any node that uses raw `.content` (i.e.
    doesn't go through `with_structured_output`) needs to go through this
    to get plain text back, regardless of which content shape the model
    used.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") == "text"
        )
    return str(content)
