#!/usr/bin/env python3
"""
Generate TypeScript interfaces from the backend Python models.

Reads:
  - state/types.py     → Pydantic models (Context, BranchManifest, IterState, MainState, BranchResult, SubStrategyDict)
  - models/strategy.py  → Pydantic models (Strategy, StrategizeOutput)
  - models/decision.py  → Pydantic models (OptimizationDecision, ErrorAnalysis, SubStrategy)

Writes:
  - frontend/src/api/types.generated.ts

Usage:
    python scripts/generate_types.py          # from frontend/ directory
    python frontend/scripts/generate_types.py # from project root
"""

import sys
from pathlib import Path
from typing import get_origin, get_args, Union

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = FRONTEND_DIR.parent
OUTPUT_FILE = FRONTEND_DIR / "src" / "api" / "types.generated.ts"

# Add project root to path so we can import the models
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Python type → TypeScript type mapping
# ============================================================

def python_type_to_ts(annotation) -> str:
    """Convert a Python type annotation to a TypeScript type string."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Optional[X] → X | null
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return f"{python_type_to_ts(non_none[0])} | null"
        else:
            return " | ".join(python_type_to_ts(a) for a in args)

    # Handle List[X] → X[]
    if origin is list:
        if args:
            return f"{python_type_to_ts(args[0])}[]"
        return "unknown[]"

    # Handle Dict/dict → Record<K, V>
    if origin is dict:
        if args and len(args) == 2:
            k = python_type_to_ts(args[0])
            v = python_type_to_ts(args[1])
            return f"Record<{k}, {v}>"
        return "Record<string, unknown>"

    if origin is type(None):
        return "null"

    # Check for Literal
    try:
        from typing import Literal
        if origin is Literal or (hasattr(annotation, '__origin__') and str(annotation).startswith('typing.Literal')):
            return " | ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in args)
    except ImportError:
        pass

    # Primitives
    TYPE_MAP = {
        str: "string",
        int: "number",
        float: "number",
        bool: "boolean",
        type(None): "null",
        dict: "Record<string, unknown>",
        list: "unknown[]",
        object: "unknown",
    }

    if annotation in TYPE_MAP:
        return TYPE_MAP[annotation]

    # If it's a class we know about, reference by name
    if isinstance(annotation, type):
        return annotation.__name__

    return "unknown"


# ============================================================
# Extract interfaces from Pydantic BaseModel classes
# ============================================================

def pydantic_to_ts(cls) -> str:
    """Convert a Pydantic BaseModel to a TypeScript interface string."""
    lines = [f"export interface {cls.__name__} {{"]

    for field_name, field_info in cls.model_fields.items():
        annotation = field_info.annotation
        ts_type = python_type_to_ts(annotation)

        origin = get_origin(annotation)
        args = get_args(annotation)
        is_nullable = origin is Union and type(None) in args
        optional_marker = "?" if is_nullable else ""

        lines.append(f"  {field_name}{optional_marker}: {ts_type};")

    lines.append("}")
    return "\n".join(lines)


# ============================================================
# Main generation
# ============================================================

def main():
    print(f"Generating TypeScript types from backend models...")
    print(f"  Backend dir: {PROJECT_ROOT}")
    print(f"  Output file: {OUTPUT_FILE}")

    # Import backend models
    from state.types import Context, BranchManifest, IterState, MainState, BranchResult, SubStrategyDict, StrategyInfo
    from models.strategy import Strategy, StrategizeOutput
    from models.decision import OptimizationDecision, ErrorAnalysis, SubStrategy as SubStrategyPydantic

    # Build the output
    sections = []

    sections.append("// =============================================================")
    sections.append("// AUTO-GENERATED — DO NOT EDIT MANUALLY")
    sections.append("// Generated from backend Python models by scripts/generate_types.py")
    sections.append("// =============================================================")
    sections.append("//")
    sections.append("// Source files:")
    sections.append("//   - state/types.py    (Pydantic: Context, BranchManifest, IterState, MainState, BranchResult, SubStrategyDict)")
    sections.append("//   - models/strategy.py (Pydantic: Strategy, StrategizeOutput)")
    sections.append("//   - models/decision.py (Pydantic: OptimizationDecision, ErrorAnalysis, SubStrategy)")
    sections.append("//")
    sections.append("")

    # BranchStatus — all statuses from the iteration FSM + branch-level statuses
    sections.append("// All status values from the iteration FSM and branch lifecycle")
    sections.append("export type BranchStatus =")
    sections.append('  | "initialized"')
    sections.append('  | "planning"')
    sections.append('  | "implementing"')
    sections.append('  | "configuring"')
    sections.append('  | "running"')
    sections.append('  | "profiling"')
    sections.append('  | "proposing"')
    sections.append('  | "deciding"')
    sections.append('  | "decided"')
    sections.append('  | "success"')
    sections.append('  | "failed"')
    sections.append('  | "branching"')
    sections.append('  | "stopped";')
    sections.append("")

    # Pydantic models from models/
    sections.append("// --- From models/strategy.py ---")
    sections.append("")
    sections.append(pydantic_to_ts(Strategy))
    sections.append("")
    sections.append(pydantic_to_ts(StrategizeOutput))
    sections.append("")

    sections.append("// --- From models/decision.py ---")
    sections.append("")
    sections.append(pydantic_to_ts(ErrorAnalysis))
    sections.append("")
    sections.append(pydantic_to_ts(SubStrategyPydantic))
    sections.append("")
    sections.append(pydantic_to_ts(OptimizationDecision))
    sections.append("")

    # Pydantic models from state/types.py
    sections.append("// --- From state/types.py ---")
    sections.append("")
    sections.append(pydantic_to_ts(Context))
    sections.append("")
    sections.append(pydantic_to_ts(StrategyInfo))
    sections.append("")
    sections.append(pydantic_to_ts(BranchManifest))
    sections.append("")
    sections.append(pydantic_to_ts(IterState))
    sections.append("")
    sections.append(pydantic_to_ts(MainState))
    sections.append("")
    sections.append(pydantic_to_ts(BranchResult))
    sections.append("")
    sections.append(pydantic_to_ts(SubStrategyDict))
    sections.append("")

    # Frontend-specific types (not from backend, but needed by components)
    sections.append("// --- Frontend-specific types ---")
    sections.append("")
    sections.append("export interface Problem {")
    sections.append("  name: string;")
    sections.append("  status: string;")
    sections.append("  description?: string;")
    sections.append("}")
    sections.append("")
    sections.append("export interface TokenUsage {")
    sections.append("  api_calls: number;")
    sections.append("  prompt_tokens: number;")
    sections.append("  completion_tokens: number;")
    sections.append("  total_tokens: number;")
    sections.append("}")
    sections.append("")
    sections.append("export interface ResultsSummary {")
    sections.append("  has_success: boolean;")
    sections.append("  num_successful: number;")
    sections.append("  num_total: number;")
    sections.append("  best_config: Record<string, number> | null;")
    sections.append("  best_time_us: number | null;")
    sections.append("  reference_time_us: number | null;")
    sections.append("  speedup: number | null;")
    sections.append("}")
    sections.append("")
    sections.append("export interface UserMessage {")
    sections.append("  content: string;")
    sections.append("  timestamp: string;")
    sections.append("  iter_num?: number;")
    sections.append("}")
    sections.append("")
    sections.append("export interface IterationSnapshot {")
    sections.append("  iter_num: number;")
    sections.append("  status: string;")
    sections.append("  plan: string;")
    sections.append("  kernel_code: string;")
    sections.append("  framework_cpp: string;")
    sections.append("  run_output: string;")
    sections.append("  ncu_metrics: Record<string, unknown> | null;")
    sections.append("  proposal?: string;")
    sections.append("  decision: OptimizationDecision | null;")
    sections.append("  feedback: string;")
    sections.append("  results_summary?: ResultsSummary | null;")
    sections.append("  user_messages?: UserMessage[];")
    sections.append("}")
    sections.append("")
    sections.append("export interface TreeNode {")
    sections.append("  id: string;")
    sections.append("  parentId: string | null;")
    sections.append("  strategy: Strategy;")
    sections.append("  status: string;")
    sections.append("  iter_num: number;")
    sections.append("  max_iter: number;")
    sections.append("  best_time_us: number | null;")
    sections.append("  speedup: number | null;")
    sections.append("  iterations: IterationSnapshot[];")
    sections.append("  user_messages?: UserMessage[];")
    sections.append("}")
    sections.append("")

    output = "\n".join(sections) + "\n"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output)

    print(f"  Generated {OUTPUT_FILE.name} ({len(output)} bytes)")


if __name__ == "__main__":
    main()
