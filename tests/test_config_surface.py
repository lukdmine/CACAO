"""Every attribute the codebase reads off `config` must actually be there.

The LLM stack moved from config.py into llm/, and config re-exports the public names.
Private module state did not come with it — `config._current_model` moved to
llm.providers and nothing re-exported it, so api/optimizer.py's run-metadata write
raised AttributeError inside a forked run. No test covers api/, so nothing caught it.

This is an import-and-getattr check rather than a call: it costs nothing and it fails
the moment a name that used to live on config stops living there.
"""

import ast
from pathlib import Path

import pytest

import config

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_PARTS = {
    "KTT",
    "node_modules",
    "dist",
    ".git",
    "__pycache__",
    "problems",
    "tests",
    ".claude",
    "archive",
    "output",
}
# config and llm own this state; they are allowed to touch their own privates.
SKIP_FILES = {"config.py"}


def _sources():
    for path in sorted(REPO_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        if rel.parts[0] == "llm" or path.name in SKIP_FILES:
            continue
        yield rel, path


def _config_aliases(tree):
    """Local names bound to the config module: `import config`, `import config as _cfg`."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "config":
                    names.add(alias.asname or "config")
    return names


def config_attribute_reads():
    """(file, lineno, attribute) for every `<config-alias>.<attr>` in the codebase."""
    for rel, path in _sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        aliases = _config_aliases(tree)
        if not aliases:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
            ):
                yield f"{rel}:{node.lineno}", node.attr


def test_every_attribute_read_off_config_exists():
    missing = [
        f"{where}: config.{attr}"
        for where, attr in config_attribute_reads()
        if not hasattr(config, attr)
    ]
    assert not missing, (
        "these read an attribute config does not have — the usual cause is that it "
        "moved to llm/ and was not re-exported:\n  " + "\n  ".join(missing)
    )


@pytest.mark.parametrize(
    "name",
    [
        # The public surface the rest of the codebase imports from config. Each of these
        # now lives in llm/ and is re-exported; dropping one breaks callers silently,
        # because `from config import X` fails only when that module is first imported.
        "get_model",
        "get_provider",
        "get_default_model",
        "set_model",
        "get_llm_creative",
        "get_llm_precise",
        "check_api_key",
        "global_tracker",
        "get_tracker_stats",
        "TrackedLLM",
        "TrackedStructuredLLM",
        "TokenTracker",
        "MODELS",
        "MODEL_CONFIGS",
        # Genuinely config's own.
        "get_output_dir",
        "get_problem_dir",
        "set_output_dir",
        "set_problem_dir",
        "OptimizerConfig",
        "init_from_config",
        "SCRIPT_DIR",
        "MAX_ITERATIONS",
        "MAX_BRANCH_DEPTH",
        "PATH_BUDGET",
        "MAX_STRATEGIES",
        "AGENTIC_STEPS",
        "STEP_TOOL_BUDGET",
        "STEP_TRUNCATION_RETRIES",
        "CROSS_BRANCH_ACCESS",
    ],
)
def test_config_still_exposes(name):
    assert hasattr(config, name), f"config.{name} disappeared"


def test_get_model_answers_without_a_model_having_been_set():
    """run_meta.json is written with config.get_model(). It used to read
    config._current_model, which is None until set_model runs — so the metadata could
    say the run had no model."""
    assert isinstance(config.get_model(), str)
    assert config.get_model()
