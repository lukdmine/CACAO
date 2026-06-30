#!/usr/bin/env bash
# Archive student-problem outputs to experiments/<name>/.
# Usage: bash archive_experiment.sh <experiment_name>
#
# Moves problems/student_*/output/ trees into experiments/<name>/<problem>/output/
# (NOT a copy — the working tree is left clean for the next run_eval.sh).
# eval_results.log is copied alongside if present. Non-student problems
# (e.g. mmul_*) are left untouched.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -lt 1 ]; then
    echo "Usage: bash archive_experiment.sh <experiment_name>"
    echo "Example: bash archive_experiment.sh baseline_run_1"
    exit 1
fi

NAME="$1"
DEST="$SCRIPT_DIR/experiments/$NAME"

if [ -d "$DEST" ]; then
    echo "ERROR: experiment '$NAME' already exists at $DEST"
    exit 1
fi

mkdir -p "$DEST"

# Copy eval results log
if [ -f "$SCRIPT_DIR/eval_results.log" ]; then
    cp "$SCRIPT_DIR/eval_results.log" "$DEST/"
    echo "Copied eval_results.log"
fi

# Move each problem's output directory
count=0
for problem_dir in "$SCRIPT_DIR"/problems/student_*/; do
    output_dir="$problem_dir/output"
    if [ -d "$output_dir" ]; then
        problem_name=$(basename "$problem_dir")
        dest_problem="$DEST/$problem_name"
        mkdir -p "$dest_problem"
        mv "$output_dir" "$dest_problem/"
        count=$((count + 1))
        echo "Archived: $problem_name/output/"
    fi
done

echo ""
echo "Done. Moved $count problem outputs to: $DEST"
echo "Total size: $(du -sh "$DEST" | cut -f1)"
echo ""
echo "Problem output dirs have been removed — run_eval.sh will start fresh."
