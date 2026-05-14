#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/monitor_system.sh [options]

Periodically append lightweight host/GPU telemetry to a log file.

Options:
  --interval SECONDS   Sample interval. Default: 10.
  --output PATH        Log path. Default: artifacts/benchmarks/system_monitor.log.
  --stop-file PATH     Stop after this file exists, after writing one final sample.
  --once               Write one sample and exit.
  -h, --help           Show this help.
EOF
}

interval=10
output="artifacts/benchmarks/system_monitor.log"
stop_file=""
once=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      interval="$2"
      shift 2
      ;;
    --output)
      output="$2"
      shift 2
      ;;
    --stop-file)
      stop_file="$2"
      shift 2
      ;;
    --once)
      once=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$interval" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "--interval must be numeric: $interval" >&2
  exit 2
fi

output_dir="$(dirname "$output")"
mkdir -p "$output_dir"
monitor_start_epoch="$(date +%s)"

recent_kernel_markers() {
  if ! command -v dmesg >/dev/null 2>&1; then
    echo "dmesg not found on PATH"
    return 0
  fi

  dmesg -T 2>/dev/null \
    | grep -Ei "oom|out of memory|killed process|nvrm|xid|gpu|cuda|reset|thermal|power" \
    | while IFS= read -r line; do
        if [[ "$line" =~ ^\[([^]]+)\] ]]; then
          marker_epoch="$(date -d "${BASH_REMATCH[1]}" +%s 2>/dev/null || true)"
          if [[ -n "$marker_epoch" && "$marker_epoch" =~ ^[0-9]+$ && "$marker_epoch" -ge "$monitor_start_epoch" ]]; then
            printf '%s\n' "$line"
          fi
        fi
      done \
    | tail -40 \
    || true
}

echo "[monitor] started_at=$(date -Is) start_epoch=$monitor_start_epoch interval=${interval}s output=$output stop_file=${stop_file:-none}" >> "$output"

trap 'echo "[monitor] stopped_by_signal_at=$(date -Is)" >> "$output"; exit 130' INT TERM

while true; do
  {
    echo "===== qds-monitor $(date -Is) ====="
    echo "[system]"
    hostname || true
    uptime || true

    echo "[memory]"
    free -h || true

    echo "[swap]"
    swapon --show || true

    echo "[disk]"
    df -h . "$output_dir" 2>/dev/null || true

    echo "[gpu-summary]"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi \
        --query-gpu=timestamp,name,driver_version,pstate,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.current.graphics,clocks.current.memory \
        --format=csv || echo "nvidia-smi gpu summary failed with status=$?"
      echo "[gpu-processes]"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
        || echo "nvidia-smi compute-app query failed or no compute processes were visible"
    else
      echo "nvidia-smi not found on PATH"
    fi

    echo "[top-rss]"
    ps -eo pid,ppid,stat,%cpu,%mem,rss,vsz,etime,cmd --sort=-rss | head -20 || true

    echo "[kernel-markers]"
    recent_kernel_markers
    echo
  } >> "$output" 2>&1

  if [[ "$once" == "1" ]]; then
    break
  fi
  if [[ -n "$stop_file" && -e "$stop_file" ]]; then
    echo "[monitor] stop_file_detected_at=$(date -Is) stop_file=$stop_file" >> "$output"
    break
  fi
  sleep "$interval"
done
