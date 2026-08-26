"""Staging and edit discipline.

The point of exact-match, unique-anchor edits is that an edit either lands where it
was meant to or fails loudly. A half-applied change to a kernel costs far more to
diagnose than a rejected tool call.
"""

import pytest

from agentic.workspace import ALL_FILES, ToolError, Workspace


def test_writes_land_in_staging_not_the_iteration_dir(workspace):
    workspace.write("kernels.cu", "// hello\n")
    assert (workspace.staging / "kernels.cu").exists()
    assert not (workspace.iter_dir / "kernels.cu").exists()


def test_commit_moves_files_and_clears_staging(filled_workspace):
    written = filled_workspace.commit()
    assert set(written) == set(ALL_FILES)
    for name in ALL_FILES:
        assert (filled_workspace.iter_dir / name).exists()
    assert not filled_workspace.staging.exists()


def test_reset_discards_a_previous_attempt(workspace):
    workspace.write("kernels.cu", "// stale\n")
    workspace.reset()
    assert not workspace.exists("kernels.cu")
    assert workspace.missing() == list(ALL_FILES)


def test_seed_loads_previous_files_for_editing(workspace):
    seeded = workspace.seed({"kernels.cu": "// previous\n", "ignored.txt": "x"})
    assert seeded == ["kernels.cu"]
    assert workspace.read("kernels.cu") == "// previous\n"


def test_seed_skips_empty_content(workspace):
    assert workspace.seed({"kernels.cu": ""}) == []


def test_unknown_file_is_rejected(workspace):
    with pytest.raises(ToolError, match="not a file this step owns"):
        workspace.write("main.cpp", "int main(){}")


def test_empty_write_is_rejected(workspace):
    with pytest.raises(ToolError, match="empty content"):
        workspace.write("kernels.cu", "   \n")


def test_reading_an_unwritten_file_is_rejected(workspace):
    with pytest.raises(ToolError, match="not been written"):
        workspace.read("kernels.cu")


def test_edit_before_write_is_rejected(workspace):
    with pytest.raises(ToolError, match="does not exist yet"):
        workspace.edit("kernels.cu", "a", "b")


def test_edit_requires_a_unique_anchor(workspace):
    workspace.write("kernels.cu", "int x = 1;\nint x = 1;\n")
    with pytest.raises(ToolError, match="occurs 2 times"):
        workspace.edit("kernels.cu", "int x = 1;", "int x = 2;")


def test_edit_reports_a_missing_anchor(workspace):
    workspace.write("kernels.cu", "int x = 1;\n")
    with pytest.raises(ToolError, match="was not found"):
        workspace.edit("kernels.cu", "int y = 1;", "int y = 2;")


def test_edit_rejects_a_no_op(workspace):
    workspace.write("kernels.cu", "int x = 1;\n")
    with pytest.raises(ToolError, match="identical"):
        workspace.edit("kernels.cu", "int x = 1;", "int x = 1;")


def test_edit_rejects_an_empty_anchor(workspace):
    workspace.write("kernels.cu", "int x = 1;\n")
    with pytest.raises(ToolError, match="must not be empty"):
        workspace.edit("kernels.cu", "", "x")


def test_edit_applies_exactly(workspace):
    workspace.write("kernels.cu", "#define TILE 16\nint x;\n")
    workspace.edit("kernels.cu", "TILE 16", "TILE 32")
    assert workspace.read("kernels.cu") == "#define TILE 32\nint x;\n"


def test_missing_lists_only_what_is_absent(workspace):
    workspace.write("kernels.cu", "// k\n")
    assert "kernels.cu" not in workspace.missing()
    assert workspace.complete() is False


def test_regions_are_keyed_for_the_assembler(filled_workspace):
    regions = filled_workspace.regions()
    assert set(regions) == {"kernels", "params", "launcher"}
    assert "AddParameter" in regions["params"]


def test_utf8_content_round_trips(workspace):
    # Kernel comments in this repo carry µ and mathematical symbols; a locale-default
    # encoding would corrupt them.
    workspace.write("kernels.cu", "// 305 µs — Σxᵢyⱼ\n")
    assert "µs" in workspace.read("kernels.cu")
