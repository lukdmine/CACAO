"""Cross-branch reads, including what must stay unreachable."""

import json

import pytest

from state import crossbranch


def test_index_is_sorted_fastest_first(cov_output):
    text = crossbranch.branch_index(cov_output)
    times = [
        int(part.split("best ")[1].split(" µs")[0].replace(",", ""))
        for part in text.splitlines()
        if "best " in part
    ]
    assert times == sorted(times)


def test_index_excludes_the_reading_branch(cov_output, cov_branch):
    text = crossbranch.branch_index(cov_output, exclude_path=cov_branch)
    assert "\n- register_coarsened_syrk " not in text
    assert "single_pass_raw_score" in text


def test_index_is_bounded(cov_output):
    text = crossbranch.branch_index(cov_output)
    assert len([l for l in text.splitlines() if l.startswith("- ")]) <= (
        crossbranch._MAX_INDEX_ROWS + 1  # +1 for the "and N more" line
    )


def test_index_is_empty_without_branches(tmp_path):
    assert crossbranch.branch_index(tmp_path) == ""


def test_nested_branches_are_addressable(cov_output):
    names = crossbranch.branch_names(cov_output)
    assert "register_coarsened_syrk/cpasync_direct_syrk" in names
    assert crossbranch.resolve_branch(cov_output, "cpasync_direct_syrk") is not None


def test_log_reports_what_was_tried(cov_output):
    text = crossbranch.branch_log(cov_output, "single_pass_raw_score")
    assert "iter 1:" in text
    assert text.count("\n- iter") >= 5


def test_log_carries_no_kernel_source(cov_output):
    # The pruning signal is what was tried and what happened. Handing over the code
    # would make four parallel branches converge into one.
    text = crossbranch.branch_log(cov_output, "single_pass_raw_score")
    assert "__global__" not in text
    assert "```" not in text


def test_log_names_available_branches_when_asked_for_a_bad_one(cov_output):
    text = crossbranch.branch_log(cov_output, "does_not_exist")
    assert "Unknown branch" in text and "single_pass_raw_score" in text


def test_errors_are_the_deepest_cross_branch_read_available():
    # Regression guard: read_branch_file gave a branch its sibling's kernels.cu, which
    # is exactly the contamination cross-branch sharing has to avoid.
    assert not hasattr(crossbranch, "read_branch_file")


def test_errors_returns_failure_analyses(cov_output):
    # The assertion used to be `"Failures recorded" in text or "no failure analyses"
    # in text` — the union of the function's two possible outcomes, so it could only
    # fail by raising. This branch has recorded analyses; say so.
    text = crossbranch.branch_errors(cov_output, "register_coarsened_syrk")
    assert "Failures recorded" in text
    assert "no failure analyses" not in text


def test_errors_reports_absence_rather_than_failing(tmp_path):
    root = tmp_path / "branches" / "b"
    root.mkdir(parents=True)
    (root / "branch.json").write_text(json.dumps({"current_iter": 1}), encoding="utf-8")
    assert "no failure analyses" in crossbranch.branch_errors(tmp_path, "b")


def test_in_flight_iteration_is_not_read(tmp_path):
    # A sibling's current iteration is mid-write and has no decision yet; reading it
    # would surface a half-written state as though it were a result.
    root = tmp_path / "branches" / "b"
    (root / "iter1").mkdir(parents=True)
    (root / "iter2").mkdir(parents=True)
    (root / "branch.json").write_text(json.dumps({"current_iter": 2}), encoding="utf-8")
    for n, summary in ((1, "done"), (2, "in flight")):
        (root / f"iter{n}" / "state.json").write_text(
            json.dumps({"iter_num": n, "decision": {"iteration_summary": summary}}),
            encoding="utf-8",
        )
    text = crossbranch.branch_log(tmp_path, "b")
    assert "done" in text and "in flight" not in text


def test_corrupt_sibling_state_is_skipped(tmp_path):
    root = tmp_path / "branches" / "b"
    (root / "iter1").mkdir(parents=True)
    (root / "branch.json").write_text(json.dumps({"current_iter": 2}), encoding="utf-8")
    (root / "iter1" / "state.json").write_text("{ truncated", encoding="utf-8")
    # A worker rewriting this file concurrently is normal, not exceptional.
    assert "no completed iterations" in crossbranch.branch_log(tmp_path, "b").lower()


def test_list_iterations_distinguishes_ran_from_failed(cov_branch):
    text = crossbranch.list_iterations(cov_branch)
    assert "| 8 | ran |" in text
    # iter9 compiled nothing: 0/218. This is the "last iteration that worked" lookup.
    assert "| 9 | failed |" in text
    assert "0/218" in text


def _finished_branch(tmp_path, status: str, last_iter: int = 3):
    """A branch that terminated at ``last_iter``.

    The worker stops advancing current_iter on a terminal decision, so it names that
    iteration rather than the next one.
    """
    root = tmp_path / "branches" / "b"
    for n in range(1, last_iter + 1):
        (root / f"iter{n}").mkdir(parents=True, exist_ok=True)
        (root / f"iter{n}" / "state.json").write_text(
            json.dumps(
                {
                    "iter_num": n,
                    "decision": {
                        "iteration_summary": f"summary {n}",
                        "error_analysis": {"error_type": f"failure {n}"},
                    },
                }
            ),
            encoding="utf-8",
        )
    (root / "branch.json").write_text(
        json.dumps({"current_iter": last_iter, "status": status}), encoding="utf-8"
    )
    return root


@pytest.mark.parametrize("status", ["success", "failed", "branching"])
def test_a_finished_branch_shows_its_final_iteration(tmp_path, status):
    """current_iter is an exclusive bound only while a branch is running.

    A branch that stopped at iteration 3 keeps current_iter == 3, so treating it as
    exclusive hid iteration 3 from every cross-branch read — including the terminal
    error_analysis, which is the one CROSS_BRANCH_ACCESS="errors" exists to share.
    """
    root = _finished_branch(tmp_path, status)

    assert "summary 3" in crossbranch.branch_log(tmp_path, "b")
    assert "| 3 |" in crossbranch.list_iterations(root)
    assert "failure 3" in crossbranch.branch_errors(tmp_path, "b")


def test_a_running_branch_still_hides_its_in_flight_iteration(tmp_path):
    """The fix must not leak the iteration a live worker is mid-write on."""
    _finished_branch(tmp_path, "running")

    text = crossbranch.branch_log(tmp_path, "b")
    assert "summary 2" in text
    assert "summary 3" not in text


def test_index_counts_a_finished_branch_final_iteration(tmp_path):
    _finished_branch(tmp_path, "success")
    assert "3 iters" in crossbranch.branch_index(tmp_path)


def test_index_does_not_count_a_running_branch_in_flight_iteration(tmp_path):
    _finished_branch(tmp_path, "running")
    assert "2 iters" in crossbranch.branch_index(tmp_path)
