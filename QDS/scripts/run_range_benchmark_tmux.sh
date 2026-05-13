#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_range_benchmark_tmux.sh [launcher options] [benchmark_runner args...]

Launch the range workload-aware diagnostic benchmark in tmux with a second pane
logging lightweight system/GPU telemetry.

Launcher options:
  --session NAME       tmux session name. Default: qds-range-benchmark.
  --no-attach          Start the tmux session but do not attach/switch to it.
  -h, --help           Show this help.

Environment overrides:
  PYTHON                       Python executable. Default: repo-root .venv absolute path.
  PROFILE                      benchmark_runner profile. Default: range_workload_aware_diagnostic.
  CSV_PATH                     Cleaned CSV file/directory. Default: ../AISDATA/cleaned.
  CACHE_DIR                    Cache directory.
  ARTIFACT_ROOT                Benchmark family directory. Default:
                               artifacts/benchmarks/range_workload_aware_diagnostic.
  RUN_ID                       Run directory name. Default: timestamped slug.
  RESULTS_DIR                  Exact benchmark run directory. Overrides
                               ARTIFACT_ROOT/RUN_ID when set.
  MAX_POINTS_PER_SEGMENT       Optional per-segment point cap. Default: unset.
  MAX_SEGMENTS                 Optional segment cap. Default: unset.
  MAX_TRAJECTORIES             Optional post-load trajectory cap. Default: unset.
  MONITOR_INTERVAL             Monitor sample interval in seconds. Default: 10.
  ATTACH                       Attach to tmux after start. Default: 1.

Any remaining arguments are appended to the benchmark_runner command.
EOF
}

q() {
  printf '%q' "$1"
}

join_shell() {
  printf '%q ' "$@"
}

display_path() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "$QDS_ROOT" "$1" ;;
  esac
}

QDS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$(cd "$QDS_ROOT/.." && pwd)/.venv/bin/python"
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
PROFILE="${PROFILE:-range_workload_aware_diagnostic}"
CSV_PATH="${CSV_PATH:-../AISDATA/cleaned}"
CACHE_DIR="${CACHE_DIR:-artifacts/cache/range_workload_aware_diagnostic}"
MAX_POINTS_PER_SEGMENT="${MAX_POINTS_PER_SEGMENT:-}"
MAX_SEGMENTS="${MAX_SEGMENTS:-}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-10}"
SESSION="${SESSION:-qds-range-benchmark}"
ATTACH="${ATTACH:-1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/benchmarks/range_workload_aware_diagnostic}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)_range_${PROFILE}_3day_full}"
RESULTS_DIR="${RESULTS_DIR:-$ARTIFACT_ROOT/runs/$RUN_ID}"

extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION="$2"
      shift 2
      ;;
    --no-attach)
      ATTACH=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found on PATH." >&2
  exit 127
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable is not executable: $PYTHON" >&2
  exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 2
fi

results_abs="$(display_path "$RESULTS_DIR")"
mkdir -p "$results_abs"
artifact_root_abs="$(display_path "$ARTIFACT_ROOT")"
mkdir -p "$artifact_root_abs"
printf '%s\n' "$RESULTS_DIR" > "$artifact_root_abs/latest_run.txt"

logs_dir="$RESULTS_DIR/logs"
console_log="$logs_dir/console.log"
monitor_log="$logs_dir/system_monitor.log"
status_file="$logs_dir/tmux_status.txt"
done_file="$logs_dir/.benchmark.done"
mkdir -p "$(display_path "$logs_dir")"
rm -f "$(display_path "$monitor_log")" "$(display_path "$status_file")" "$(display_path "$done_file")"

benchmark_cmd=(
  "$PYTHON"
  -m src.experiments.benchmark_runner
  --profile "$PROFILE"
  --workloads range
  --csv_path "$CSV_PATH"
  --cache_dir "$CACHE_DIR"
  --results_dir "$RESULTS_DIR"
  --run_id "$RUN_ID"
)

if [[ -n "$MAX_POINTS_PER_SEGMENT" ]]; then
  benchmark_cmd+=(--max_points_per_segment "$MAX_POINTS_PER_SEGMENT")
fi

if [[ -n "$MAX_SEGMENTS" ]]; then
  benchmark_cmd+=(--max_segments "$MAX_SEGMENTS")
fi

if [[ -n "$MAX_TRAJECTORIES" ]]; then
  benchmark_cmd+=(--max_trajectories "$MAX_TRAJECTORIES")
fi

if [[ "${#extra_args[@]}" -gt 0 ]]; then
  benchmark_cmd+=("${extra_args[@]}")
fi

monitor_cmd=(
  "$QDS_ROOT/scripts/monitor_system.sh"
  --interval "$MONITOR_INTERVAL"
  --output "$monitor_log"
  --stop-file "$done_file"
)

benchmark_shell=$(
  cat <<EOF
set -o pipefail
cd $(q "$QDS_ROOT")
mkdir -p $(q "$RESULTS_DIR")
rm -f $(q "$done_file")
trap 'touch $(q "$done_file")' EXIT
{
  echo "[tmux] session=$(q "$SESSION")"
  echo "[tmux] started_at=\$(date -Is)"
  echo "[tmux] command=$(join_shell "${benchmark_cmd[@]}")"
} | tee $(q "$status_file")
$(join_shell "${benchmark_cmd[@]}") 2>&1 | tee $(q "$console_log")
status=\${PIPESTATUS[0]}
if [ "\$status" -ne 0 ]; then
  $(q "$PYTHON") scripts/mark_benchmark_failed.py \
    --status-file $(q "$RESULTS_DIR/run_status.json") \
    --exit-status "\$status" \
    --message "tmux launcher observed non-zero benchmark exit status \$status"
fi
{
  echo "[tmux] exit_status=\$status"
  echo "[tmux] finished_at=\$(date -Is)"
} | tee -a $(q "$status_file")
touch $(q "$done_file")
exit "\$status"
EOF
)

monitor_shell="cd $(q "$QDS_ROOT"); $(join_shell "${monitor_cmd[@]}")"

tmux new-session -d -s "$SESSION" -n benchmark -c "$QDS_ROOT" "$benchmark_shell"
tmux split-window -h -t "$SESSION:benchmark" -c "$QDS_ROOT" "$monitor_shell"
tmux select-layout -t "$SESSION:benchmark" even-horizontal >/dev/null
tmux select-pane -t "$SESSION:benchmark.0"

echo "Started tmux session: $SESSION"
echo "Run ID:        $RUN_ID"
echo "Run directory: $(display_path "$RESULTS_DIR")"
echo "Benchmark log: $(display_path "$console_log")"
echo "Monitor log:   $(display_path "$monitor_log")"
echo "Status file:   $(display_path "$status_file")"

if [[ "$ATTACH" == "1" ]]; then
  if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$SESSION"
  else
    tmux attach -t "$SESSION"
  fi
else
  echo "Attach later with: tmux attach -t $SESSION"
fi
