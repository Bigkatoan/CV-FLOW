"""
cv_flow.cli — Command-line entry point for the cv-flow package.

Usage:
    cv-flow run <launch.py>           Run a pipeline launch script
    cv-flow validate <topics_dir>     Parse and validate all .topic files in a directory
    cv-flow list-nodes                Print the built-in node catalog
"""
from __future__ import annotations

import argparse
import runpy
import sys

from cv_flow.topic.parser import load_topics_dir, ParseError


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute a launch script as __main__ (so `if __name__ == '__main__'` blocks run)."""
    try:
        runpy.run_path(args.launch_file, run_name="__main__")
    except FileNotFoundError:
        print(f"cv-flow: launch file not found: {args.launch_file}", file=sys.stderr)
        return 1
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from pathlib import Path

    if not Path(args.topics_dir).is_dir():
        print(f"cv-flow: directory not found: {args.topics_dir}", file=sys.stderr)
        return 1

    try:
        topics = load_topics_dir(args.topics_dir)
    except ParseError as exc:
        print(f"cv-flow: {exc}", file=sys.stderr)
        return 1

    if not topics:
        print(f"cv-flow: no .topic files found in {args.topics_dir}")
        return 0

    print(f"cv-flow: {len(topics)} topic(s) validated OK")
    for name, td in sorted(topics.items()):
        in_desc  = "none" if td.input_port.is_none  else f"{len(td.input_port.fields)} field(s)"
        out_desc = "none" if td.output_port.is_none else f"{len(td.output_port.fields)} field(s)"
        elastic  = " [elastic]" if td.elastic else ""
        print(f"  - {name}: input={in_desc}, output={out_desc}{elastic}")
    return 0


def _cmd_list_nodes(args: argparse.Namespace) -> int:
    from cv_flow.nodes._catalog import NODE_CATALOG

    for node_type, meta in sorted(NODE_CATALOG.items()):
        print(f"{node_type} [{meta['category']}]")
        print(f"  {meta['description']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cv-flow", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a pipeline launch script")
    p_run.add_argument("launch_file", help="Path to a Python launch script")
    p_run.set_defaults(func=_cmd_run)

    p_validate = sub.add_parser("validate", help="Validate all .topic files in a directory")
    p_validate.add_argument("topics_dir", help="Path to a directory containing .topic files")
    p_validate.set_defaults(func=_cmd_validate)

    p_list = sub.add_parser("list-nodes", help="Print the built-in node catalog")
    p_list.set_defaults(func=_cmd_list_nodes)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
