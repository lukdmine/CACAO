// =============================================================
// AUTO-GENERATED — DO NOT EDIT MANUALLY
// Generated from backend Python models by scripts/generate_types.py
// =============================================================
//
// Source files:
//   - state/types.py    (Pydantic: Context, BranchManifest, IterState, MainState, BranchResult, SubStrategyDict)
//   - models/strategy.py (Pydantic: Strategy, StrategizeOutput)
//   - models/decision.py (Pydantic: OptimizationDecision, ErrorAnalysis, SubStrategy)
//   - models/finalize.py (Pydantic: KernelTags, ArchReport)
//

// All status values from the iteration FSM and branch lifecycle
export type BranchStatus =
  | "idle"
  | "initialized"
  | "planning"
  | "authoring"
  | "implementing"
  | "configuring"
  | "running"
  | "profiling"
  | "proposing"
  | "deciding"
  | "decided"
  | "success"
  | "failed"
  | "branching"
  | "stopped";

// --- From models/strategy.py ---

export interface Strategy {
  name: string;
  description: string;
  hypothesis: string;
  key_parameters: string[];
}

export interface StrategizeOutput {
  strategies: Strategy[];
  reasoning: string;
}

// --- From models/decision.py ---

export interface ErrorAnalysis {
  error_type: "compilation" | "runtime" | "validation" | "timeout" | "none";
  root_cause: string;
  suggested_fix: string;
}

export interface SubStrategy {
  name: string;
  description: string;
  hypothesis: string;
  key_parameters: string[];
}

export interface OptimizationDecision {
  action: "continue" | "retry" | "stop" | "branch";
  reasoning: string;
  feedback: string;
  error_analysis?: ErrorAnalysis | null;
  sub_strategies?: SubStrategy[] | null;
  skip_implement: boolean;
  iteration_summary: string;
}

// --- From state/types.py ---

export interface Context {
  analysis: string;
  gpu_info?: Record<string, unknown> | null;
}

export interface StrategyInfo {
  name: string;
  description: string;
  hypothesis: string;
  key_parameters: string[];
}

export interface BranchConfig {
  max_iter: number;
}

export interface BranchManifest {
  strategy: StrategyInfo;
  plan: string;
  branch_depth: number;
  path_iters_consumed: number;
  path_budget_total: number;
  current_iter: number;
  status: string;
  best_time_us?: number | null;
  speedup?: number | null;
  sub_strategies_cache?: Record<string, unknown>[] | null;
  pre_stop_status?: string | null;
}

export interface IterState {
  iter_num: number;
  status: string;
  next_status?: string | null;
  authoring_scope: "full" | "config_only";
  mode: "fresh" | "retry" | "followup";
  kernel_code: string;
  framework_cpp: string;
  run_output: string;
  ncu_metrics?: Record<string, unknown> | null;
  decision?: Record<string, unknown> | null;
  feedback: string;
  results_summary?: Record<string, unknown> | null;
  user_messages: Record<string, unknown>[];
  proposal: string;
}

export interface MainState {
  problem_yaml: string;
  ref_kernel: string;
  analysis: string;
  strategies: Record<string, unknown>[];
}

export interface BranchResult {
  branch_name: string;
  strategy: Record<string, unknown>;
  best_config?: Record<string, unknown> | null;
  best_time_us?: number | null;
  speedup?: number | null;
  iterations: number;
  status: string;
}

export interface SubStrategyDict {
  name: string;
  description: string;
  hypothesis: string;
  key_parameters: string[];
}

// --- From models/finalize.py and state/types.py (finalize) ---

export type FinalizeStatus =
  | "selecting"
  | "widening"
  | "tuning"
  | "verifying"
  | "overview"
  | "done"
  | "failed";

export interface KernelTags {
  tensor_cores: boolean;
  async_copy: boolean;
  precision: string;
  min_arch?: number | null;
  smem_bytes?: number | null;
  sm_scaled: boolean;
  note: string;
}

export interface ArchReport {
  archs: number[];
  base_min_arch?: number | null;
  winner_min_arch?: number | null;
  per_toggle: Record<string, number | null>;
  skipped: string;
}

export interface Variant {
  dir: string;
  summary?: Record<string, unknown> | null;
  tags?: KernelTags | null;
  arch?: ArchReport | null;
  ncu_metrics?: Record<string, unknown> | null;
  searcher: string;
}

export interface FinalistState {
  name: string;
  source_branch: string;
  source_iter: number;
  reason: string;
  forced: boolean;
  stages: Record<string, string>;
  source: Variant;
  finalized?: Variant | null;
  reported: "source" | "finalized";
  reported_dir: string;
  kernel_diff_lines?: number | null;
  error: string;
}

export interface OverviewFact {
  label: string;
  value: string;
}

export interface Overview {
  headline: string;
  findings: string[];
  dead_ends: string[];
  caveats: string[];
  facts: OverviewFact[];
  markdown: string;
}

export interface FinalizeManifest {
  status: "selecting" | "widening" | "tuning" | "verifying" | "overview" | "done" | "failed";
  attempt: number;
  trigger: "auto" | "manual";
  started_at: string;
  finished_at: string;
  error: string;
  pid?: number | null;
  source_fingerprint: string;
  finalists: FinalistState[];
  category_bests: Record<string, string>;
  overview?: Overview | null;
}

// --- Frontend-specific types ---

export interface Problem {
  name: string;
  status: string;
  description?: string;
}

export interface TokenUsage {
  api_calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ResultsSummary {
  has_success: boolean;
  num_successful: number;
  num_total: number;
  best_config: Record<string, number> | null;
  best_time_us: number | null;
  reference_time_us: number | null;
  speedup: number | null;
  cases?: Record<string, ResultsSummary | null>;
  geomean_speedup?: number | null;
  worst_case?: string | null;
  failed_case?: string | null;
}

export interface UserMessage {
  content: string;
  timestamp: string;
  iter_num?: number;
}

export interface IterationSnapshot {
  iter_num: number;
  status: string;
  decision: OptimizationDecision | null;
  results_summary?: ResultsSummary | null;
  has: {
    kernel_code: boolean;
    framework_cpp: boolean;
    run_output: boolean;
    ncu_metrics: boolean;
    proposal: boolean;
  };
}

export interface IterationDetail {
  iter_num: number;
  status: string;
  mode?: string;
  kernel_code?: string;
  framework_cpp?: string;
  run_output?: string;
  ncu_metrics?: Record<string, unknown> | null;
  proposal?: string;
  decision?: OptimizationDecision | null;
  feedback?: string;
  results_summary?: ResultsSummary | null;
  user_messages?: UserMessage[];
}

export interface TreeNode {
  id: string;
  parentId: string | null;
  strategy: Strategy;
  status: string;
  plan: string;
  iter_num: number;
  max_iter: number;
  best_time_us: number | null;
  speedup: number | null;
  iterations: IterationSnapshot[];
  user_messages?: UserMessage[];
}

