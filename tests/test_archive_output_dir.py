"""Starting a fresh run must not destroy the previous one.

This function is what both the CLI (without ``--resume``) and the frontend's Run
button call, and neither asks first. When it was an unconditional ``rmtree`` it took
a 23-hour run with it — four branches of kernels, gone on one misclick, with no undo.
It now renames the tree to ``archive/output_N`` instead, which costs nothing: a rename
is O(1) whatever the size, and the archive sits on the same filesystem.

It also must not destroy a hand-seeded ``reference_time.json``. The baseline for a
python or cpu_c reference cannot be measured by timing the reference — a GPU kernel
"beats" a single-threaded C loop by two orders of magnitude — so the speedup
denominator comes from measuring the real incumbent externally and seeding the file.
Everything downstream already respects that seed (``save_reference_time`` is
O_CREAT|O_EXCL first-write-wins), so this is the one place it could vanish.
"""

import json
import shutil

import pytest

import config
from utils.files import archive_output_dir, prune_archives
from utils.results import _REFERENCE_TIME_FILE, load_reference_time


@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    """Point config's global output dir at a temp dir for the duration of one test."""
    out = tmp_path / "output"
    monkeypatch.setattr(config, "_base_output_dir", out, raising=False)
    assert config.get_output_dir() == out
    return out


def _seed(out, us=1772.56):
    out.mkdir(parents=True, exist_ok=True)
    (out / _REFERENCE_TIME_FILE).write_text(json.dumps({"reference_time_us": us}))


def test_seeded_reference_time_survives(output_dir):
    _seed(output_dir)
    archive_output_dir()
    assert load_reference_time(output_dir) == 1772.56


def test_the_new_output_dir_starts_empty(output_dir):
    _seed(output_dir)
    _run(output_dir)

    archive_output_dir()

    assert sorted(p.name for p in output_dir.iterdir()) == [_REFERENCE_TIME_FILE]


def test_no_seed_leaves_a_clean_empty_dir(output_dir):
    output_dir.mkdir(parents=True)
    (output_dir / "context.json").write_text("{}")

    archive_output_dir()

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []
    assert load_reference_time(output_dir) is None


def test_absent_output_dir_is_created(output_dir):
    assert not output_dir.exists()
    archive_output_dir()
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_seed_survives_repeated_runs(output_dir):
    """A re-run of a re-run must not erode the baseline."""
    _seed(output_dir, 999.0)
    for _ in range(3):
        archive_output_dir()
    assert load_reference_time(output_dir) == 999.0


# -- archiving --------------------------------------------------------------


def _run(out, *, branch="some_branch", kernel="// v1\n", best=226.816):
    """A minimal output tree that looks like a run that produced work."""
    iter_dir = out / "branches" / branch / "iter1"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "kernels.cu").write_text(kernel)
    (iter_dir / "state.json").write_text(json.dumps({"iter_num": 1}))
    (iter_dir / "cacao_in_q.bin").write_bytes(b"\x00" * 4096)
    (iter_dir / "driver").write_bytes(b"\x7fELF" + b"\x00" * 1020)
    (out / "branches" / branch / "branch.json").write_text(
        json.dumps({"best_time_us": best})
    )
    (out / "context.json").write_text("{}")
    return iter_dir


def _archive_root(output_dir):
    return output_dir.parent / "archive"


def test_a_previous_run_is_archived_not_deleted(output_dir):
    """The whole point. A misclick must cost disk, not work."""
    _run(output_dir, kernel="// the 226 microsecond kernel\n")

    archive_output_dir()

    kept = _archive_root(output_dir) / "output_1" / "branches" / "some_branch" / "iter1"
    assert kept.is_dir()
    assert (kept / "kernels.cu").read_text() == "// the 226 microsecond kernel\n"


def test_archives_are_numbered_upwards(output_dir):
    for n in range(1, 4):
        _run(output_dir, kernel=f"// run {n}\n")
        archive_output_dir()

    root = _archive_root(output_dir)
    assert sorted(p.name for p in root.iterdir()) == ["output_1", "output_2", "output_3"]
    third = root / "output_3" / "branches" / "some_branch" / "iter1" / "kernels.cu"
    assert third.read_text() == "// run 3\n"


def test_a_number_is_never_reused(output_dir):
    """Deleting output_2 by hand must not make the next run overwrite output_3."""
    for _ in range(3):
        _run(output_dir)
        archive_output_dir()
    shutil.rmtree(_archive_root(output_dir) / "output_2")

    _run(output_dir, kernel="// fourth\n")
    archive_output_dir()

    assert (_archive_root(output_dir) / "output_4" / "branches").is_dir()
    assert (_archive_root(output_dir) / "output_3" / "branches").is_dir()


def test_the_archive_call_reports_where_it_went(output_dir):
    _run(output_dir)
    assert archive_output_dir() == _archive_root(output_dir) / "output_1"


def test_a_run_with_no_branches_is_not_archived(output_dir):
    """A run aborted during analysis produced nothing worth a numbered directory."""
    _seed(output_dir)
    (output_dir / "run.log").write_text("started, then stopped")

    assert archive_output_dir() is None
    assert not _archive_root(output_dir).exists()
    assert load_reference_time(output_dir) == 1772.56


def test_archiving_leaves_the_seed_in_the_new_dir_and_the_old_one(output_dir):
    _seed(output_dir)
    _run(output_dir)

    archive_output_dir()

    assert load_reference_time(output_dir) == 1772.56
    archived = _archive_root(output_dir) / "output_1"
    assert json.loads((archived / _REFERENCE_TIME_FILE).read_text())[
        "reference_time_us"
    ] == 1772.56


# -- pruning ----------------------------------------------------------------


def test_pruning_drops_regenerable_files_and_keeps_the_rest(output_dir):
    _run(output_dir)
    archive_output_dir()

    removed, freed = prune_archives(output_dir.parent)

    kept = _archive_root(output_dir) / "output_1" / "branches" / "some_branch" / "iter1"
    assert removed == 2 and freed == 4096 + 1024
    assert not (kept / "cacao_in_q.bin").exists()
    assert not (kept / "driver").exists()
    # The irreplaceable half is untouched.
    assert (kept / "kernels.cu").exists()
    assert (kept / "state.json").exists()


def test_pruning_never_touches_the_live_output_dir(output_dir):
    _run(output_dir)
    archive_output_dir()
    live = _run(output_dir, kernel="// current run\n")

    prune_archives(output_dir.parent)

    assert (live / "cacao_in_q.bin").exists()
    assert (live / "driver").exists()


def test_pruning_with_no_archive_is_a_no_op(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    assert prune_archives(output_dir.parent) == (0, 0)


def test_pruning_twice_frees_nothing_the_second_time(output_dir):
    _run(output_dir)
    archive_output_dir()
    prune_archives(output_dir.parent)
    assert prune_archives(output_dir.parent) == (0, 0)
