#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_benchmark_queue_tmux.sh [launcher options] [benchmark_runner args...]

Launch a sequential benchmark_runner queue in tmux with a second pane logging
lightweight system/GPU telemetry. Use this for multi-seed batches where runs
must not compete for the same GPU/CPU/RAM.

Launcher options:
  --session NAME       tmux session name. Default: qds-benchmark-queue.
  --no-attach          Start the tmux session but do not attach/switch to it.
  -h, --help           Show this help.

Environment overrides:
  UV                           uv executable. Default: uv.
  UV_GROUP                     uv dependency group. Default: dev.
  PROFILE                      benchmark_runner profile. Default: range_workload_aware_diagnostic.
  WORKLOADS                    benchmark_runner --workloads value. Default: range.
  CSV_PATH                     Cleaned CSV file/directory. Default: ../AISDATA/cleaned.
  CACHE_DIR                    Cache directory.
  ARTIFACT_ROOT                Benchmark family directory. Default:
                               artifacts/benchmarks/range_workload_aware_diagnostic.
  SEEDS                        Comma-separated seeds for default plan. Default: 42,43,44.
  CHILD_EXTRA_ARGS             String passed as benchmark_runner --extra_args for
                               every default-plan run. Example:
                               "--ranking_top_quantile 0.70".
  PLAN_FILE                    Optional tab-separated plan file with rows:
                               run_id<TAB>seed<TAB>child_extra_args
                               Blank lines and # comments are ignored.
  QUEUE_ID                     Queue artifact directory name. Default: timestamped.
  QUEUE_DIR                    Exact queue artifact directory. Overrides
                               ARTIFACT_ROOT/queues/QUEUE_ID when set.
  RUN_PREFIX                   Default run_id prefix when PLAN_FILE is unset.
                               Default: QUEUE_ID.
  MAX_POINTS_PER_SEGMENT       Optional per-segment point cap. Default: unset.
  MAX_SEGMENTS                 Optional segment cap. Default: unset.
  MAX_TRAJECTORIES             Optional post-load trajectory cap. Default: unset.
  MONITOR_INTERVAL             Monitor sample interval in seconds. Default: 10.
  CONTINUE_ON_FAILURE          Continue later queue rows after a failed run.
                               Default: 0.
  ATTACH                       Attach to tmux after start. Default: 1.

Any remaining arguments are appended to every benchmark_runner command.
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

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

QDS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV:-uv}"
UV_GROUP="${UV_GROUP:-dev}"
PROFILE="${PROFILE:-range_workload_aware_diagnostic}"
WORKLOADS="${WORKLOADS:-range}"
CSV_PATH="${CSV_PATH:-../AISDATA/cleaned}"
CACHE_DIR="${CACHE_DIR:-artifacts/cache/range_workload_aware_diagnostic}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/benchmarks/range_workload_aware_diagnostic}"
SEEDS="${SEEDS:-42,43,44}"
CHILD_EXTRA_ARGS="${CHILD_EXTRA_ARGS:-}"
PLAN_FILE="${PLAN_FILE:-}"
QUEUE_ID="${QUEUE_ID:-$(date +%Y%m%d-%H%M%S)_${PROFILE}_queue}"
RUN_PREFIX="${RUN_PREFIX:-$QUEUE_ID}"
QUEUE_DIR="${QUEUE_DIR:-$ARTIFACT_ROOT/queues/$QUEUE_ID}"
MAX_POINTS_PER_SEGMENT="${MAX_POINTS_PER_SEGMENT:-}"
MAX_SEGMENTS="${MAX_SEGMENTS:-}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-10}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
SESSION="${SESSION:-qds-benchmark-queue}"
ATTACH="${ATTACH:-1}"

extra_benchmark_args=()
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
      extra_benchmark_args+=("$1")
      shift
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found on PATH." >&2
  exit 127
fi

if ! command -v "$UV" >/dev/null 2>&1; then
  echo "uv executable was not found on PATH: $UV" >&2
  exit 127
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 2
fi

queue_abs="$(display_path "$QUEUE_DIR")"
artifact_root_abs="$(display_path "$ARTIFACT_ROOT")"
mkdir -p "$queue_abs/logs" "$artifact_root_abs"
printf '%s\n' "$QUEUE_DIR" > "$artifact_root_abs/latest_queue.txt"

plan_abs="$queue_abs/queue_plan.tsv"
if [[ -n "$PLAN_FILE" ]]; then
  plan_source="$(display_path "$PLAN_FILE")"
  if [[ ! -f "$plan_source" ]]; then
    echo "PLAN_FILE does not exist: $plan_source" >&2
    exit 2
  fi
  cp "$plan_source" "$plan_abs"
else
  : > "$plan_abs"
  IFS=',' read -r -a seed_values <<< "$SEEDS"
  for raw_seed in "${seed_values[@]}"; do
    seed="$(trim "$raw_seed")"
    if [[ -z "$seed" ]]; then
      continue
    fi
    printf '%s\t%s\t%s\n' "${RUN_PREFIX}_seed${seed}" "$seed" "$CHILD_EXTRA_ARGS" >> "$plan_abs"
  done
fi

if ! awk -F '\t' 'NF && $1 !~ /^#/ { found=1 } END { exit(found ? 0 : 1) }' "$plan_abs"; then
  echo "Queue plan has no runnable rows: $plan_abs" >&2
  exit 2
fi

"$UV" run --group "$UV_GROUP" -- python "$QDS_ROOT/scripts/validate_benchmark_queue_plan.py" "$plan_abs"

console_log="$QUEUE_DIR/logs/console.log"
monitor_log="$QUEUE_DIR/logs/system_monitor.log"
status_file="$QUEUE_DIR/queue_status.jsonl"
summary_file="$QUEUE_DIR/queue_summary.json"
manifest_file="$QUEUE_DIR/queue_manifest.json"
done_file="$QUEUE_DIR/logs/.queue.done"
runner_file="$queue_abs/queue_runner.sh"
rm -f "$(display_path "$console_log")" "$(display_path "$monitor_log")" "$(display_path "$status_file")" \
  "$(display_path "$summary_file")" "$(display_path "$done_file")"

"$UV" run --group "$UV_GROUP" -- python - "$(display_path "$manifest_file")" \
  "$QUEUE_ID" "$SESSION" "$PROFILE" "$WORKLOADS" "$ARTIFACT_ROOT" "$QUEUE_DIR" \
  "$QUEUE_DIR/queue_plan.tsv" "$CONTINUE_ON_FAILURE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "queue_id": sys.argv[2],
    "session": sys.argv[3],
    "profile": sys.argv[4],
    "workloads": sys.argv[5],
    "artifact_root": sys.argv[6],
    "queue_dir": sys.argv[7],
    "plan_file": sys.argv[8],
    "continue_on_failure": bool(int(sys.argv[9])),
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -uo pipefail\n'
  printf 'QDS_ROOT=%q\n' "$QDS_ROOT"
  printf 'UV=%q\n' "$UV"
  printf 'UV_GROUP=%q\n' "$UV_GROUP"
  printf 'PROFILE=%q\n' "$PROFILE"
  printf 'WORKLOADS=%q\n' "$WORKLOADS"
  printf 'CSV_PATH=%q\n' "$CSV_PATH"
  printf 'CACHE_DIR=%q\n' "$CACHE_DIR"
  printf 'ARTIFACT_ROOT=%q\n' "$ARTIFACT_ROOT"
  printf 'PLAN_FILE=%q\n' "$QUEUE_DIR/queue_plan.tsv"
  printf 'CONSOLE_LOG=%q\n' "$console_log"
  printf 'STATUS_FILE=%q\n' "$status_file"
  printf 'SUMMARY_FILE=%q\n' "$summary_file"
  printf 'DONE_FILE=%q\n' "$done_file"
  printf 'MAX_POINTS_PER_SEGMENT=%q\n' "$MAX_POINTS_PER_SEGMENT"
  printf 'MAX_SEGMENTS=%q\n' "$MAX_SEGMENTS"
  printf 'MAX_TRAJECTORIES=%q\n' "$MAX_TRAJECTORIES"
  printf 'CONTINUE_ON_FAILURE=%q\n' "$CONTINUE_ON_FAILURE"
  if [[ "${#extra_benchmark_args[@]}" -gt 0 ]]; then
    printf 'EXTRA_BENCHMARK_ARGS=(%s)\n' "$(join_shell "${extra_benchmark_args[@]}")"
  else
    printf 'EXTRA_BENCHMARK_ARGS=()\n'
  fi
  cat <<'RUNNER'
cd "$QDS_ROOT"
mkdir -p "$(dirname "$CONSOLE_LOG")"
rm -f "$DONE_FILE"
trap 'touch "$DONE_FILE"' EXIT

json_event() {
  "$UV" run --group "$UV_GROUP" -- python - "$STATUS_FILE" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {}
for item in sys.argv[2:]:
    key, value = item.split("=", 1)
    if key in {"seed", "exit_status"}:
        payload[key] = int(value)
    else:
        payload[key] = value
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")
PY
}

run_one() {
  local run_id="$1"
  local seed="$2"
  local child_extra_args="${3:-}"
  local results_dir="$ARTIFACT_ROOT/runs/$run_id"
  local started finished status
  started="$(date -Is)"
  json_event event=started run_id="$run_id" seed="$seed" started_at="$started" child_extra_args="$child_extra_args"
  echo "[queue] starting run_id=$run_id seed=$seed child_extra_args=$child_extra_args at $started" | tee -a "$CONSOLE_LOG"

  local cmd=(
    "$UV"
    run
    --group "$UV_GROUP"
    --
    python
    -m experiments.benchmark_runner
    --profile "$PROFILE"
    --workloads "$WORKLOADS"
    --csv_path "$CSV_PATH"
    --cache_dir "$CACHE_DIR"
    --results_dir "$results_dir"
    --run_id "$run_id"
    --run_label "$PROFILE"
    --seed "$seed"
  )
  if [[ -n "$MAX_POINTS_PER_SEGMENT" ]]; then
    cmd+=(--max_points_per_segment "$MAX_POINTS_PER_SEGMENT")
  fi
  if [[ -n "$MAX_SEGMENTS" ]]; then
    cmd+=(--max_segments "$MAX_SEGMENTS")
  fi
  if [[ -n "$MAX_TRAJECTORIES" ]]; then
    cmd+=(--max_trajectories "$MAX_TRAJECTORIES")
  fi
  if [[ -n "$child_extra_args" ]]; then
    cmd+=(--extra_args "$child_extra_args")
  fi
  if [[ "${#EXTRA_BENCHMARK_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${EXTRA_BENCHMARK_ARGS[@]}")
  fi

  printf '[queue] command:' | tee -a "$CONSOLE_LOG"
  printf ' %q' "${cmd[@]}" | tee -a "$CONSOLE_LOG"
  printf '\n' | tee -a "$CONSOLE_LOG"

  "${cmd[@]}" 2>&1 | tee -a "$CONSOLE_LOG"
  status=${PIPESTATUS[0]}
  finished="$(date -Is)"
  if [[ "$status" -ne 0 ]]; then
    "$UV" run --group "$UV_GROUP" -- python scripts/mark_benchmark_failed.py \
      --status-file "$results_dir/run_status.json" \
      --exit-status "$status" \
      --message "queue launcher observed non-zero benchmark exit status $status"
  fi
  json_event event=finished run_id="$run_id" seed="$seed" exit_status="$status" finished_at="$finished"
  echo "[queue] finished run_id=$run_id status=$status at $finished" | tee -a "$CONSOLE_LOG"
  return "$status"
}

started_at="$(date -Is)"
echo "[queue] started_at=$started_at plan=$PLAN_FILE" | tee -a "$CONSOLE_LOG"
json_event event=queue_started started_at="$started_at" plan_file="$PLAN_FILE"

overall=0
run_count=0
failure_count=0
while IFS=$'\t' read -r run_id seed child_extra_args || [[ -n "${run_id:-}" ]]; do
  [[ -z "${run_id//[[:space:]]/}" ]] && continue
  [[ "$run_id" =~ ^[[:space:]]*# ]] && continue
  run_count=$((run_count + 1))
  if ! run_one "$run_id" "$seed" "${child_extra_args:-}"; then
    overall=1
    failure_count=$((failure_count + 1))
    if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
      echo "[queue] stopping after failed run because CONTINUE_ON_FAILURE=$CONTINUE_ON_FAILURE" | tee -a "$CONSOLE_LOG"
      break
    fi
  fi
done < "$PLAN_FILE"

finished_at="$(date -Is)"
json_event event=queue_finished exit_status="$overall" finished_at="$finished_at"
"$UV" run --group "$UV_GROUP" -- python - "$SUMMARY_FILE" "$started_at" "$finished_at" "$overall" "$run_count" "$failure_count" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "started_at": sys.argv[2],
    "finished_at": sys.argv[3],
    "exit_status": int(sys.argv[4]),
    "run_count": int(sys.argv[5]),
    "failure_count": int(sys.argv[6]),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
echo "[queue] finished_at=$finished_at overall=$overall run_count=$run_count failure_count=$failure_count" | tee -a "$CONSOLE_LOG"
touch "$DONE_FILE"
exit "$overall"
RUNNER
} > "$runner_file"
chmod +x "$runner_file"

monitor_cmd=(
  "$QDS_ROOT/scripts/monitor_system.sh"
  --interval "$MONITOR_INTERVAL"
  --output "$monitor_log"
  --stop-file "$done_file"
)

tmux new-session -d -s "$SESSION" -n queue -c "$QDS_ROOT" "$runner_file"
tmux split-window -h -t "$SESSION:queue" -c "$QDS_ROOT" "$(join_shell "${monitor_cmd[@]}")"
tmux select-layout -t "$SESSION:queue" even-horizontal >/dev/null
tmux select-pane -t "$SESSION:queue.0"

echo "Started tmux session: $SESSION"
echo "Queue ID:       $QUEUE_ID"
echo "Queue dir:      $(display_path "$QUEUE_DIR")"
echo "Plan file:      $(display_path "$QUEUE_DIR/queue_plan.tsv")"
echo "Queue log:      $(display_path "$console_log")"
echo "Monitor log:    $(display_path "$monitor_log")"
echo "Status events:  $(display_path "$status_file")"

if [[ "$ATTACH" == "1" ]]; then
  if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$SESSION"
  else
    tmux attach -t "$SESSION"
  fi
else
  echo "Attach later with: tmux attach -t $SESSION"
fi
