#!/usr/bin/env bash
# Launches backend (conda 'ktt' env) and frontend in a tmux session 'cuda-opt'.
# Usage:
#   ./start.sh           # frontend on 0.0.0.0  (npm run host)
#   ./start.sh --no-host # frontend on localhost (npm run dev)
#
# Backend stdout+stderr go to ./server.log; tail -f to follow.
# Switch panes: Ctrl-b n / Ctrl-b p. Detach: Ctrl-b d. Reattach: tmux attach -t cuda-opt.
set -euo pipefail

NPM_SCRIPT="host"
[[ "${1:-}" == "--no-host" ]] && NPM_SCRIPT="dev"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="cuda-opt"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n backend -c "$SCRIPT_DIR" \
  'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ktt && python server.py 2>&1 | tee server.log'

tmux new-window -t "$SESSION" -n frontend -c "$SCRIPT_DIR/frontend" \
  "npm run $NPM_SCRIPT"

tmux select-window -t "$SESSION":backend
tmux attach -t "$SESSION"
