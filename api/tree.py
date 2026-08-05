"""Tree scanning, results, status, and GPU endpoints."""

import hashlib
import threading
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Response

from api.helpers import (
    get_problem_dir,
    load_yaml,
    load_json,
    _running_problems,
    _running_lock,
    is_problem_running,
)
from utils.files import get_parent_branch_dir

# Fields of IterState the tree carries inline. Everything else is fetched per
# iteration by the detail panel — see get_iteration() in api/branches.py.
_LIGHT_ITER_FIELDS = ("iter_num", "status", "decision", "results_summary")

# Fields stripped from the tree. The panel only renders these inside collapsed
# accordion sections, one iteration at a time, but they dominate the payload:
# on a long run ncu_metrics alone is ~16 MB across all iterations.
_HEAVY_ITER_FIELDS = (
    "kernel_code",
    "framework_cpp",
    "run_output",
    "ncu_metrics",
    "proposal",
)

# problem name -> {state.json path: (mtime_ns, size, light iteration dict)}
# The tree is polled every few seconds and almost nothing changes between polls,
# so unchanged state.json files are projected once and reused. Pruned each scan
# to the files actually seen, so reverted/deleted branches don't linger.
_iter_cache: dict[str, dict[str, tuple[int, int, dict]]] = {}
_iter_cache_lock = threading.Lock()


def _load_token_usage(output_dir: Path) -> dict | None:
    """Read token_usage.json from the output directory, or None if absent."""
    path = output_dir / "token_usage.json"
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def _load_run_meta(output_dir: Path) -> dict:
    """Read run_meta.json (model/provider) written by the optimizer subprocess."""
    path = output_dir / "run_meta.json"
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


router = APIRouter()


def _project_iter(snap: dict) -> dict:
    """Reduce a full IterState to what the tree needs, plus presence flags.

    The panel gates each accordion section on the field being non-empty, so the
    flags have to survive even though the content does not. ``bool({})`` is
    False, which matches the frontend's ``Object.keys(...).length > 0`` check on
    ncu_metrics.
    """
    light = {k: snap.get(k) for k in _LIGHT_ITER_FIELDS}
    light["has"] = {f: bool(snap.get(f)) for f in _HEAVY_ITER_FIELDS}
    return light


def _fingerprint(problem_dir: Path, name: str) -> str:
    """Hash everything the tree response is derived from, for the ETag.

    Stat-only: it never opens a file. ``running`` is process state rather than
    file state, so it is folded in explicitly — otherwise a run starting or
    dying would not invalidate the cached tree.
    """
    output_dir = problem_dir / "output"
    h = hashlib.blake2b(digest_size=16)
    h.update(b"1|")  # bump when the response shape changes
    h.update(b"running" if is_problem_running(name) else b"idle")

    paths = [
        problem_dir / "problem.yaml",
        output_dir / "run_meta.json",
        output_dir / "token_usage.json",
        output_dir / "strategies.json",
    ]
    branches_dir = output_dir / "branches"
    if branches_dir.is_dir():
        paths.extend(sorted(branches_dir.rglob("branch.json")))
        # branch_config.json is the only file the API itself writes, so leaving
        # it out would answer 304 to the very poll that should have shown the
        # user their own edit.
        paths.extend(sorted(branches_dir.rglob("branch_config.json")))
        paths.extend(sorted(branches_dir.rglob("iter*/state.json")))

    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        h.update(f"|{p}|{st.st_mtime_ns}|{st.st_size}".encode())
    return f'W/"{h.hexdigest()}"'


def _scan_branches(branches_dir: Path, name: str, seen: set[str]) -> list[dict]:
    """Recursively scan branch directories and collect branch.json + iter states."""
    from state import load_branch_config
    from utils.results import get_results_summary, load_reference_time

    output_dir = branches_dir.parent
    ref_time = load_reference_time(output_dir)
    cache = _iter_cache.setdefault(name, {})

    results = []
    if not branches_dir.is_dir():
        return results

    for entry in sorted(branches_dir.iterdir()):
        if not entry.is_dir():
            continue
        branch_file = entry / "branch.json"
        if not branch_file.exists():
            continue

        manifest = load_json(branch_file)
        manifest["branch_path"] = str(entry)  # derive from filesystem, not persisted
        # max_iter lives in branch_config.json, not the manifest — see BranchConfig.
        manifest["max_iter"] = load_branch_config(entry).max_iter

        # Load all iteration states
        iters = []
        all_msgs = []
        for iter_dir in sorted(entry.iterdir()):
            if not iter_dir.is_dir() or not iter_dir.name.startswith("iter"):
                continue
            state_file = iter_dir / "state.json"
            try:
                info = state_file.stat()
            except OSError:
                continue

            key = str(state_file)
            seen.add(key)
            cached = cache.get(key)
            if cached and cached[0] == info.st_mtime_ns and cached[1] == info.st_size:
                light, msgs = cached[2]
            else:
                snap = load_json(state_file)
                if not snap.get("results_summary") and snap.get("iter_num") is not None:
                    results_path = iter_dir / "results.json"
                    if results_path.exists():
                        snap["results_summary"] = get_results_summary(
                            results_path, ref_time
                        )
                light = _project_iter(snap)
                msgs = [
                    {**m, "iter_num": snap.get("iter_num", 0)}
                    for m in snap.get("user_messages", [])
                ]
                cache[key] = (info.st_mtime_ns, info.st_size, (light, msgs))

            iters.append(light)
            all_msgs.append(msgs)

        # Directory order is lexicographic (iter10 before iter2), so sort by
        # iter_num — the fallback and latest-status logic below both depend on it.
        order = sorted(range(len(iters)), key=lambda i: iters[i].get("iter_num", 0))
        iters = [iters[i] for i in order]
        manifest["_iteration_snapshots"] = iters
        manifest["_all_user_messages"] = [m for i in order for m in all_msgs[i]]

        # Fallback: pull best_time/speedup from iterations if missing from manifest
        best_time = manifest.get("best_time_us")
        speedup = manifest.get("speedup")
        for snap in iters:
            rs = snap.get("results_summary") or {}
            st, ss = rs.get("best_time_us"), rs.get("speedup")
            if st is not None and (best_time is None or st < best_time):
                best_time = st
            if ss is not None and (speedup is None or ss > speedup):
                speedup = ss
        manifest["best_time_us"] = best_time
        manifest["speedup"] = speedup

        # Surface iteration status for running branches
        if manifest.get("status") == "running" and iters:
            latest = iters[-1]
            latest_status = latest.get("status", "running")
            # Don't surface "decided" — it's a transient internal status between iterations
            if latest_status not in ("success", "failed", "decided"):
                manifest["status"] = latest_status

        results.append(manifest)

        # Recurse into sub-branches
        sub = entry / "branches"
        if sub.is_dir():
            results.extend(_scan_branches(sub, name, seen))

    return results


@router.get("/api/problems/{name}/tree")
def get_tree(
    name: str,
    response: Response,
    if_none_match: str | None = Header(default=None),
):
    """Get the optimization tree for a problem.

    Iterations are returned in light form — heavy work products come from
    /branches/{id}/iterations/{n} when the panel opens one. The response carries
    an ETag; an unchanged tree answers 304 and the frontend skips its store
    write entirely, which also keeps React Flow from re-rendering every node.
    """
    problem_dir = get_problem_dir(name)
    output_dir = problem_dir / "output"

    etag = _fingerprint(problem_dir, name)
    if if_none_match == etag:
        return Response(status_code=304, headers={"ETag": etag})
    response.headers["ETag"] = etag

    if not output_dir.is_dir():
        return {"nodes": [], "running": is_problem_running(name)}

    # Only the count is used (in the root node's hypothesis line), so the
    # strategy bodies never leave the server.
    num_strategies = 0
    strategies_path = output_dir / "strategies.json"
    if strategies_path.exists():
        num_strategies = len(load_json(strategies_path).get("strategies", []))

    seen: set[str] = set()
    with _iter_cache_lock:
        branch_states = _scan_branches(output_dir / "branches", name, seen)
        # Drop entries for iterations that no longer exist (reverted or deleted
        # branches), so the cache tracks the tree rather than growing with it.
        cache = _iter_cache.get(name)
        if cache is not None:
            for stale in cache.keys() - seen:
                del cache[stale]

    # Root node aggregation
    best_time = best_speedup = None
    overall_status = "idle"

    if branch_states:
        times = [
            s["best_time_us"]
            for s in branch_states
            if s.get("best_time_us") is not None
        ]
        if times:
            best_time = min(times)
        speedups = [s["speedup"] for s in branch_states if s.get("speedup") is not None]
        if speedups:
            best_speedup = max(speedups)

        statuses = [s.get("status", "") for s in branch_states]
        if all(s in ("success", "failed") for s in statuses):
            overall_status = (
                "success" if any(s == "success" for s in statuses) else "failed"
            )
        elif any(s not in ("success", "failed", "initialized") for s in statuses):
            overall_status = "running"
        else:
            overall_status = "initialized"

    config = {}
    py = problem_dir / "problem.yaml"
    if py.exists():
        config = load_yaml(py)

    nodes = [
        {
            "id": "root",
            "parentId": None,
            "strategy": {
                "name": config.get("name", name),
                "description": config.get("description", ""),
                "hypothesis": f"Optimizing with {num_strategies} strategies",
                "key_parameters": [],
            },
            "status": overall_status,
            "plan": "",  # branch-level; the root is not a branch
            "iter_num": 0,
            "max_iter": 0,
            "best_time_us": best_time,
            "speedup": best_speedup,
            "iterations": [],
        }
    ]

    # Branch nodes
    for state in branch_states:
        bp = state.get("branch_path", "")

        try:
            node_id = "branch/" + str(Path(bp).relative_to(output_dir / "branches"))
        except (ValueError, TypeError):
            node_id = f"branch/{state.get('strategy', {}).get('name', 'unknown')}"

        # Parent is derived from directory nesting, never from persisted paths,
        # so cloned/moved problem dirs keep their tree structure intact.
        parent_dir = get_parent_branch_dir(bp)
        if parent_dir is not None:
            parent_id = "branch/" + str(
                parent_dir.relative_to(output_dir / "branches")
            )
        else:
            parent_id = "root"

        strategy = state.get("strategy", {})
        if isinstance(strategy, str):
            strategy = {
                "name": strategy,
                "description": "",
                "hypothesis": "",
                "key_parameters": [],
            }

        nodes.append(
            {
                "id": node_id,
                "parentId": parent_id,
                "strategy": strategy,
                "status": state.get("status", "initialized"),
                "plan": state.get("plan", ""),
                "iter_num": state.get("current_iter", 0),
                "max_iter": state.get("max_iter", 5),
                "best_time_us": state.get("best_time_us"),
                "speedup": state.get("speedup"),
                "iterations": state.get("_iteration_snapshots", []),
                "user_messages": state.get("_all_user_messages", []),
            }
        )

    # Read model/provider from run_meta.json (written by optimizer subprocess)
    run_meta = _load_run_meta(output_dir)

    tuning_duration_s = (config.get("tuning") or {}).get("duration_s")

    return {
        "nodes": nodes,
        "running": is_problem_running(name),
        "llm_model": run_meta.get("model"),
        "llm_provider": run_meta.get("provider"),
        "token_usage": _load_token_usage(output_dir),
        "tuning_duration_s": tuning_duration_s,
    }


@router.get("/api/problems/{name}/results")
def get_results(name: str):
    """Get aggregated final results."""
    problem_dir = get_problem_dir(name)
    results_path = problem_dir / "output" / "final_results.json"
    if not results_path.exists():
        return {"results": None}
    return {"results": load_json(results_path)}


@router.get("/api/status")
def get_status():
    """Get global engine status."""
    with _running_lock:
        for n in [n for n, e in _running_problems.items() if not e.process.is_alive()]:
            del _running_problems[n]
        running = {
            name: {"gpu_index": e.gpu_index} for name, e in _running_problems.items()
        }
    return {"running_problems": running, "active_count": len(running)}


@router.get("/api/gpu/devices")
def get_gpu_devices():
    """List all detected GPUs."""
    from utils.gpu_info import detect_gpus

    try:
        return {"devices": detect_gpus()}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to detect GPUs: {e}"
        ) from e


@router.get("/api/gpu/selected")
def get_selected_gpu(index: int = 0):
    """Get GPU specs for a given device index."""
    from utils.gpu_info import get_gpu_details

    info = get_gpu_details(index)
    return {"index": index, "info": info}


@router.get("/api/models")
def get_models():
    """Return available LLM providers and their models."""
    from config import MODELS

    return {
        provider: {
            "default": cfg["default"],
            "available": cfg.get("available", [cfg["default"]]),
        }
        for provider, cfg in MODELS.items()
    }
