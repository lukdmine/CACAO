"""A branch's budget, and how a raised one reaches the children it spawns.

Sizing children from config rather than the parent's effective max_iter meant a grant
died on the branch it was made on.
"""

import pytest

import config as _cfg
from engine.master import _init_branch
from state import load_branch_config, load_branch_manifest, save_branch_config
from state.types import BranchConfig


@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setattr(_cfg, "_base_output_dir", out, raising=False)
    monkeypatch.setattr(_cfg, "get_output_dir", lambda: out)
    import engine.master as master

    monkeypatch.setattr(master, "get_output_dir", lambda: out)
    return out


STRATEGY = {"name": "tiled", "description": "d", "hypothesis": "h", "key_parameters": []}
CHILD = {"name": "vectorized", "description": "d", "hypothesis": "h", "key_parameters": []}


# -- depth mode -------------------------------------------------------------


def test_root_branch_takes_max_iter_from_config(output_dir, monkeypatch):
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 0)
    monkeypatch.setattr(_cfg, "MAX_ITERATIONS", 5)

    path = _init_branch(STRATEGY)

    assert load_branch_config(path).max_iter == 5


def test_child_inherits_a_raised_parent_budget(output_dir, monkeypatch):
    """The reported case: raise a branch to 12, and its children should start from 12."""
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 0)
    monkeypatch.setattr(_cfg, "MAX_ITERATIONS", 5)

    parent = _init_branch(STRATEGY)
    save_branch_config(parent, BranchConfig(max_iter=12))
    effective = load_branch_config(parent).max_iter

    child = _init_branch(
        CHILD, parent_branch=str(parent), current_depth=1, inherited_max_iter=effective
    )

    assert load_branch_config(child).max_iter == 12


def test_an_untouched_parent_still_hands_down_the_config_value(output_dir, monkeypatch):
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 0)
    monkeypatch.setattr(_cfg, "MAX_ITERATIONS", 5)

    parent = _init_branch(STRATEGY)
    child = _init_branch(
        CHILD,
        parent_branch=str(parent),
        current_depth=1,
        inherited_max_iter=load_branch_config(parent).max_iter,
    )

    assert load_branch_config(child).max_iter == 5


# -- path-budget mode -------------------------------------------------------


def test_root_branch_gets_the_whole_path_budget(output_dir, monkeypatch):
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 30)

    path = _init_branch(STRATEGY)

    assert load_branch_config(path).max_iter == 30
    assert load_branch_manifest(path).path_budget_total == 30


def test_child_gets_the_remaining_path_budget(output_dir, monkeypatch):
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 30)

    parent = _init_branch(STRATEGY)
    child = _init_branch(
        CHILD,
        parent_branch=str(parent),
        current_depth=1,
        path_iters_consumed=8,
        path_budget_total=30,
    )

    assert load_branch_config(child).max_iter == 22


def test_a_grant_extends_the_path_budget_rather_than_the_childrens_share(
    output_dir, monkeypatch
):
    """Allocated 30, raised to 34: children see a 34-iteration path, not 30 with 4
    already spent."""
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 30)

    parent = _init_branch(STRATEGY)
    save_branch_config(parent, BranchConfig(max_iter=34))

    manifest = load_branch_manifest(parent)
    budget_total = manifest.path_budget_total or _cfg.PATH_BUDGET
    allocated = max(budget_total - manifest.path_iters_consumed, 1)
    granted = max(load_branch_config(parent).max_iter - allocated, 0)
    assert granted == 4

    child = _init_branch(
        CHILD,
        parent_branch=str(parent),
        current_depth=1,
        path_iters_consumed=10,
        path_budget_total=budget_total + granted,
    )

    # 34 total on the path, 10 consumed by the parent.
    assert load_branch_config(child).max_iter == 24
    # And the extension travels further down.
    assert load_branch_manifest(child).path_budget_total == 34


def test_a_grant_compounds_down_the_path(output_dir, monkeypatch):
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 30)

    parent = _init_branch(STRATEGY)
    child = _init_branch(
        CHILD,
        parent_branch=str(parent),
        current_depth=2,
        path_iters_consumed=10,
        path_budget_total=34,
    )
    save_branch_config(child, BranchConfig(max_iter=30))  # allocated 24, granted 6

    manifest = load_branch_manifest(child)
    allocated = max(manifest.path_budget_total - manifest.path_iters_consumed, 1)
    granted = max(load_branch_config(child).max_iter - allocated, 0)
    assert granted == 6

    grandchild = _init_branch(
        {"name": "async_pipeline", "description": "", "hypothesis": "", "key_parameters": []},
        parent_branch=str(child),
        current_depth=1,
        path_iters_consumed=18,
        path_budget_total=manifest.path_budget_total + granted,
    )

    assert load_branch_manifest(grandchild).path_budget_total == 40
    assert load_branch_config(grandchild).max_iter == 22


def test_an_exhausted_path_still_gives_a_child_one_iteration(output_dir, monkeypatch):
    """A spawned branch that cannot run at all is worse than one that runs once."""
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 30)

    parent = _init_branch(STRATEGY)
    child = _init_branch(
        CHILD,
        parent_branch=str(parent),
        current_depth=1,
        path_iters_consumed=30,
        path_budget_total=30,
    )

    assert load_branch_config(child).max_iter == 1


def test_a_legacy_manifest_without_a_total_falls_back_to_config(output_dir, monkeypatch):
    """branch.json files written before path_budget_total existed carry 0."""
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 30)

    path = _init_branch(STRATEGY, path_budget_total=None)
    manifest = load_branch_manifest(path)
    manifest.path_budget_total = 0

    assert (manifest.path_budget_total or _cfg.PATH_BUDGET) == 30


# -- the spawn path itself --------------------------------------------------
#
# The tests above drive _init_branch with the numbers the spawn site computes. These
# drive _spawn_sub_branches, which is what computes them.


from engine.master import _spawn_sub_branches
from state import save_branch_manifest


def _parent(output_dir, monkeypatch, *, depth=2, current_iter=4, consumed=0, total=0):
    path = _init_branch(STRATEGY, current_depth=depth)
    manifest = load_branch_manifest(path)
    manifest.current_iter = current_iter
    manifest.path_iters_consumed = consumed
    manifest.path_budget_total = total or manifest.path_budget_total
    manifest.status = "branching"
    save_branch_manifest(path, manifest)
    return path


def test_spawn_hands_children_the_raised_parent_budget(output_dir, monkeypatch):
    """The reported case, through the real spawn path."""
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 0)
    monkeypatch.setattr(_cfg, "MAX_ITERATIONS", 5)

    parent = _parent(output_dir, monkeypatch)
    save_branch_config(parent, BranchConfig(max_iter=12))

    children = _spawn_sub_branches(parent, [CHILD])

    assert len(children) == 1
    assert load_branch_config(children[0]).max_iter == 12


def test_spawn_uses_config_when_nothing_was_raised(output_dir, monkeypatch):
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 0)
    monkeypatch.setattr(_cfg, "MAX_ITERATIONS", 5)

    parent = _parent(output_dir, monkeypatch)

    children = _spawn_sub_branches(parent, [CHILD])

    assert load_branch_config(children[0]).max_iter == 5


def test_spawn_extends_the_path_budget_by_what_was_granted(output_dir, monkeypatch):
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 30)

    parent = _parent(output_dir, monkeypatch, current_iter=10, total=30)
    save_branch_config(parent, BranchConfig(max_iter=34))  # allocated 30, granted 4

    children = _spawn_sub_branches(parent, [CHILD])

    # 34 on the path, 10 consumed by the parent.
    assert load_branch_config(children[0]).max_iter == 24
    assert load_branch_manifest(children[0]).path_budget_total == 34


def test_spawn_without_a_grant_leaves_the_path_budget_alone(output_dir, monkeypatch):
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 30)

    parent = _parent(output_dir, monkeypatch, current_iter=10, total=30)

    children = _spawn_sub_branches(parent, [CHILD])

    assert load_branch_config(children[0]).max_iter == 20
    assert load_branch_manifest(children[0]).path_budget_total == 30


def test_spawn_decrements_depth_and_accumulates_consumption(output_dir, monkeypatch):
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 0)

    parent = _parent(output_dir, monkeypatch, depth=3, current_iter=4, consumed=6)

    children = _spawn_sub_branches(parent, [CHILD])
    manifest = load_branch_manifest(children[0])

    assert manifest.branch_depth == 2
    assert manifest.path_iters_consumed == 10


def test_spawn_discards_children_at_max_depth(output_dir, monkeypatch):
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 0)

    parent = _parent(output_dir, monkeypatch, depth=1)

    assert _spawn_sub_branches(parent, [CHILD]) == []


def test_spawn_discards_children_on_an_exhausted_path_budget(output_dir, monkeypatch):
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 30)

    parent = _parent(output_dir, monkeypatch, current_iter=30, total=30)

    assert _spawn_sub_branches(parent, [CHILD]) == []


def test_a_grant_can_revive_an_otherwise_exhausted_path(output_dir, monkeypatch):
    """Raising the budget is how a user says "keep going", so it must now spawn."""
    monkeypatch.setattr(_cfg, "PATH_BUDGET", 30)

    parent = _parent(output_dir, monkeypatch, current_iter=30, total=30)
    save_branch_config(parent, BranchConfig(max_iter=40))

    children = _spawn_sub_branches(parent, [CHILD])

    assert len(children) == 1
    assert load_branch_config(children[0]).max_iter == 10
