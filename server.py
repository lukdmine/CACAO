#!/usr/bin/env python3
"""
FastAPI server for the CUDA Agentic Optimizer frontend.

Usage:
    python server.py              # starts on http://localhost:8003 (or $PORT)
    python server.py --port 8080  # custom port
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import uvicorn
from uvicorn.config import LOGGING_CONFIG

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from api import create_app
from api.helpers import PROBLEMS_DIR

app = create_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUDA Agentic Optimizer API Server")
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind to (env: HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8003")),
        help="Port to listen on (env: PORT)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    frontend_port = os.getenv("FRONTEND_PORT", "5003")
    print("\n  CUDA Agentic Optimizer API")
    print(f"  Serving problems from: {PROBLEMS_DIR}")
    print(f"  Frontend: http://localhost:{frontend_port}")
    print(f"  API: http://{args.host}:{args.port}\n")

    reload_dirs = [str(SCRIPT_DIR)] if args.reload else None
    reload_excludes = ["frontend", "KTT", "problems", "*.so"] if args.reload else None

    log_config = copy.deepcopy(LOGGING_CONFIG)
    log_config["formatters"]["default"]["fmt"] = (
        "%(asctime)s %(levelprefix)s %(message)s"
    )
    log_config["formatters"]["access"]["fmt"] = (
        '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    )
    for fmt in log_config["formatters"].values():
        fmt["datefmt"] = "%Y-%m-%d %H:%M:%S"

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=reload_dirs,
        reload_excludes=reload_excludes,
        log_config=log_config,
    )
