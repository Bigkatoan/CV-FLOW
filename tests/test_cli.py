"""
Tests for cv_flow.cli and cv_flow package-level __init__ exports.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from cv_flow.cli import main, build_parser
from cv_flow.topic.topic import clear_topics


@pytest.fixture(autouse=True)
def clean_registry():
    clear_topics()
    yield
    clear_topics()


# ── package-level exports ─────────────────────────────────────────────────────

def test_package_exports():
    """cv_flow top-level exposes Node, Executor, Topic, etc."""
    import cv_flow
    assert hasattr(cv_flow, "Node")
    assert hasattr(cv_flow, "Executor")
    assert hasattr(cv_flow, "ElasticStage")
    assert hasattr(cv_flow, "Topic")
    assert hasattr(cv_flow, "load_topics")
    assert hasattr(cv_flow, "get_topic")
    assert cv_flow.__version__ == "0.4.0"


# ── cv-flow validate ──────────────────────────────────────────────────────────

def test_cli_validate_success(capsys):
    """validate <dir> prints OK summary for valid .topic files."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "a.topic").write_text("output: -> cpu\n   - x : uint8 shape=[4]\n")

    rc = main(["validate", str(tmp)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "1 topic(s) validated OK" in out
    assert "a:" in out


def test_cli_validate_bad_dtype(capsys):
    """validate <dir> with an invalid dtype exits nonzero and prints an error."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "bad.topic").write_text("output: -> cpu\n   - x : not_a_type shape=[4]\n")

    rc = main(["validate", str(tmp)])
    err = capsys.readouterr().err

    assert rc == 1
    assert "ParseError" in err


def test_cli_validate_missing_dir(capsys):
    """validate <missing_dir> exits nonzero with a clear error message."""
    rc = main(["validate", "/no/such/directory/at/all"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "not found" in err


# ── cv-flow list-nodes ────────────────────────────────────────────────────────

def test_cli_list_nodes(capsys):
    """list-nodes prints every catalog entry's type and description."""
    rc = main(["list-nodes"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "CameraSource" in out
    assert "[input]" in out


# ── cv-flow run ───────────────────────────────────────────────────────────────

def test_cli_run_executes_script(capsys):
    """run <script.py> executes the script as __main__."""
    tmp = Path(tempfile.mkdtemp())
    script = tmp / "launch.py"
    script.write_text(
        "print('hello from launch script')\n"
        "if __name__ == '__main__':\n"
        "    print('main block ran')\n"
    )

    rc = main(["run", str(script)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "hello from launch script" in out
    assert "main block ran" in out


def test_cli_run_missing_file(capsys):
    """run <missing.py> exits nonzero with a clear error."""
    rc = main(["run", "/no/such/launch.py"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "not found" in err


# ── argparse wiring ───────────────────────────────────────────────────────────

def test_build_parser_requires_subcommand():
    """Calling cv-flow with no subcommand is a parse error (required=True)."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
