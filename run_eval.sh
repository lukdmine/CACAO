#!/usr/bin/env bash
# Experiment runner — runs all problem/model combinations sequentially.
# Usage: bash run_eval.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

PATH_BUDGET=30

PROBLEMS=(student_2024_savings student_2025_moving_average)
#PROBLEMS=(student_2022_galaxies student_2024_savings student_2025_moving_average)

# model -> provider
declare -A MODELS=(
    [kimi-k2.6]=cerit
    [qwen3.5]=cerit
    [glm-5]=cerit
    #[gpt-5.4]=openai
)

# model -> dir suffix
declare -A SUFFIXES=(
    [kimi-k2.6]=kimi
    [qwen3.5]=qwen
    [glm-5]=glm
    #[gpt-5.4]=gpt
)

RESULTS_LOG="$SCRIPT_DIR/eval_results.log"
: > "$RESULTS_LOG"

total=0
passed=0
failed=0
skipped=0

echo "========================================"
echo "  CACAO Evaluation Run"
echo "  $(date)"
echo "  Path budget: $PATH_BUDGET"
echo "  Problems: ${PROBLEMS[*]}"
echo "  Models: ${!MODELS[*]}"
echo "========================================"
echo ""

for problem in "${PROBLEMS[@]}"; do
    for model in "${!MODELS[@]}"; do
        provider="${MODELS[$model]}"
        suffix="${SUFFIXES[$model]}"
        dir="problems/${problem}_${suffix}"
        total=$((total + 1))

        echo "────────────────────────────────────────"
        echo "[$total] $problem + $model ($provider)"
        echo "  dir: $dir"

        if [ ! -d "$dir" ]; then
            echo "  SKIP — directory not found"
            skipped=$((skipped + 1))
            echo "SKIP $problem $model — dir not found" >> "$RESULTS_LOG"
            continue
        fi

        # Skip if already has results
        if [ -f "$dir/output/final_results.json" ]; then
            echo "  SKIP — already has results"
            skipped=$((skipped + 1))
            echo "SKIP $problem $model — already completed" >> "$RESULTS_LOG"
            continue
        fi

        if $DRY_RUN; then
            echo "  DRY RUN — would run: python cli.py --dir $dir --provider $provider --model $model --path-budget $PATH_BUDGET"
            continue
        fi

        start_time=$(date +%s)
        echo "  Started: $(date)"

        mkdir -p "$dir/output"
        if python cli.py --dir "$dir" --provider "$provider" --model "$model" --path-budget "$PATH_BUDGET" 2>&1 | tee "$dir/output/run.log"; then
            exit_code=0
        else
            exit_code=$?
        fi

        end_time=$(date +%s)
        duration=$(( end_time - start_time ))
        minutes=$(( duration / 60 ))
        seconds=$(( duration % 60 ))

        if [ $exit_code -eq 0 ]; then
            echo "  PASS (${minutes}m ${seconds}s)"
            passed=$((passed + 1))
            echo "PASS $problem $model ${minutes}m${seconds}s" >> "$RESULTS_LOG"
        else
            echo "  FAIL (exit $exit_code, ${minutes}m ${seconds}s)"
            failed=$((failed + 1))
            echo "FAIL $problem $model exit=$exit_code ${minutes}m${seconds}s" >> "$RESULTS_LOG"
        fi
        echo ""
    done
done

echo ""
echo "========================================"
echo "  Summary"
echo "  Total: $total  Passed: $passed  Failed: $failed  Skipped: $skipped"
echo "========================================"
echo ""
cat "$RESULTS_LOG"
