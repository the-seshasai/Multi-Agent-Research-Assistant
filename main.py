"""CLI entry point: run the research assistant graph on a user query."""

from __future__ import annotations

import argparse

from graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent research assistant")
    parser.add_argument("query", help="The research question to investigate")
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to also save the final markdown report to",
    )
    args = parser.parse_args()

    app = build_graph()

    # Initial state: query is the only field with a real value at the
    # start. Everything else starts empty/zeroed and gets filled in as the
    # graph runs -- see state.py for why each field is shared.
    initial_state = {
        "query": args.query,
        "subtasks": [],
        "research_notes": [],
        "critique": None,
        "iteration_count": 0,
        "final_report": None,
    }

    print(f"[main] starting graph for query={args.query!r}\n")
    final_state = app.invoke(initial_state)

    print("\n[main] ===== FINAL REPORT =====\n")
    print(final_state["final_report"])

    if args.output:
        with open(args.output, "w") as f:
            f.write(final_state["final_report"])
        print(f"\n[main] report saved to {args.output}")


if __name__ == "__main__":
    main()
