# Known Issues and Limitations

## Security and trust model

CACAO is a research tool intended for use within a trusted private network. The HTTP API (`server.py`) does not authenticate requests — anyone who can reach port 8003 can create problems, run optimizations, and read or delete files inside `problems/`. The engine compiles and executes LLM-generated CUDA code under the server user's identity.

**Do not expose port 8003 to the public internet, and do not run CACAO on `problem.yaml` files from untrusted sources.** The `eval()`-based size and constraint expressions in `tuner.py` are sandboxed only with `{"__builtins__": {}}`, which is not a strong sandbox; a crafted expression can escape and execute arbitrary Python in the engine process.

## Functional limitations

- **Python 3.10 required.** The `pyktt` bindings are pinned to the 3.10 ABI; newer Python versions break pybind11 in KTT.
- **Single-host only.** Concurrent tuning on multiple GPUs of the same host is supported; cross-host scheduling is not implemented.
- **Cross-process branch config writes are not serialized.** If the API server and the engine subprocess both write to the same `branch.json` (e.g. user changes `max_iter` via the UI while the worker is mid-iteration), the later writer wins and the earlier change can be silently lost. Workaround: retry the config change. A proper fix requires cross-process file locking on `branch.json`.
