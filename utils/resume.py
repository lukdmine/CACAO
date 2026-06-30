"""
Resume utilities - scan output directory and reconstruct state for resuming.

Scans ``branch.json`` manifests to find branches that were interrupted
mid-execution and returns their paths for re-queuing.
"""

from pathlib import Path
from typing import List

from config import get_output_dir
from utils.log import log
from state import load_branch_manifest, save_branch_manifest


def get_resume_states() -> List[Path]:
    """
    Scan all branches (including sub-branches) and return paths
    for any branch that was interrupted mid-execution.

    Returns:
        List of branch directory paths to resume.
    """
    branches_dir = get_output_dir() / "branches"

    if not branches_dir.exists():
        log("No branches directory found - nothing to resume.", "WARN")
        return []

    resume_paths = []

    # Recursively find every branch.json in the branches directory tree
    for branch_file in branches_dir.rglob("branch.json"):
        try:
            branch_path = branch_file.parent
            manifest = load_branch_manifest(branch_path)
            status = manifest.status
            branch_name = manifest.strategy.name

            # Terminal states — skip
            if status in ["success", "failed"]:
                log(
                    f"  Branch '{branch_name}': {status} (skipping)",
                    "SUCCESS" if status == "success" else "ERROR",
                )
                continue

            # Already branched — skip
            if status == "branching":
                log(
                    f"  Branch '{branch_name}': branching complete (skipping)",
                    "SUCCESS",
                )
                continue

            # Restore stopped branches
            if status == "stopped":
                pre = manifest.pre_stop_status
                manifest.pre_stop_status = None
                if pre:
                    manifest.status = pre
                else:
                    manifest.status = "running"
                save_branch_manifest(branch_path, manifest)
                status = manifest.status

            log(
                f"  Branch '{branch_name}': resuming at status '{status}' (iter {manifest.current_iter})",
                "INFO",
            )
            resume_paths.append(branch_path)

        except Exception as e:
            log(f"Failed to read/parse branch file {branch_file}: {e}", "ERROR")

    return resume_paths
