#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/benchmark_preflight.sh [options]

Check local prerequisites before launching the minimum realistic range benchmark.

Options:
  --session NAME       tmux session name. Default: qds-range-benchmark.
  --csv-path PATH      Cleaned CSV file/directory. Default: ../AISDATA/cleaned.
  --cache-dir PATH     Cache directory. Default: artifacts/cache/range_workload_matrix_min_realistic.
  --artifact-root PATH Benchmark family root. Default:
                       artifacts/benchmarks/range_workload_matrix_min_realistic.
  --python PATH        Python executable. Default: ../.venv/bin/python.
  --min-free-gb N      Required free space on artifact filesystem. Default: 20.
  -h, --help           Show this help.

Environment variables with the same names used by run_range_benchmark_tmux.sh
are honored: SESSION, CSV_PATH, CACHE_DIR, ARTIFACT_ROOT, PYTHON.
EOF
}

QDS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-qds-range-benchmark}"
CSV_PATH="${CSV_PATH:-../AISDATA/cleaned}"
CACHE_DIR="${CACHE_DIR:-artifacts/cache/range_workload_matrix_min_realistic}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/benchmarks/range_workload_matrix_min_realistic}"
PYTHON="${PYTHON:-$QDS_ROOT/../.venv/bin/python}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION="$2"
      shift 2
      ;;
    --csv-path)
      CSV_PATH="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --min-free-gb)
      MIN_FREE_GB="$2"
      shift 2
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

failures=0
warnings=0

ok() {
  echo "[ok] $*"
}

warn() {
  warnings=$((warnings + 1))
  echo "[warn] $*"
}

fail() {
  failures=$((failures + 1))
  echo "[fail] $*"
}

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$QDS_ROOT" "$1" ;;
  esac
}

is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

cd "$QDS_ROOT" || exit 2

if command -v tmux >/dev/null 2>&1; then
  ok "tmux is available"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    fail "tmux session already exists: $SESSION"
  else
    ok "tmux session name is available: $SESSION"
  fi
else
  fail "tmux is not installed or not on PATH"
fi

if [[ -x "$PYTHON" ]]; then
  ok "Python executable is available: $PYTHON"
  python_summary="$("$PYTHON" - <<'PY' 2>/dev/null
import sys
try:
    import torch
    print(f"python={sys.version.split()[0]} torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
except Exception as exc:
    print(f"python={sys.version.split()[0]} torch_import_error={type(exc).__name__}: {exc}")
    raise
PY
)"
  if [[ "$?" -eq 0 ]]; then
    ok "$python_summary"
  else
    fail "Python could not import torch cleanly"
  fi
else
  fail "Python executable is not executable: $PYTHON"
fi

csv_abs="$(abs_path "$CSV_PATH")"
if [[ -d "$csv_abs" ]]; then
  csv_count="$(find "$csv_abs" -maxdepth 1 -type f -iname '*.csv' | wc -l)"
  if [[ "$csv_count" -ge 2 ]]; then
    ok "cleaned CSV directory has at least two CSV files: $csv_abs ($csv_count files)"
  else
    fail "cleaned CSV directory needs at least two CSV files: $csv_abs ($csv_count files found)"
  fi
elif [[ -f "$csv_abs" ]]; then
  ok "cleaned CSV file exists: $csv_abs"
  warn "minimum realistic profile normally expects a directory with two cleaned CSV days"
else
  fail "cleaned CSV path does not exist: $csv_abs"
fi

for path in "$CACHE_DIR" "$ARTIFACT_ROOT"; do
  path_abs="$(abs_path "$path")"
  if mkdir -p "$path_abs" 2>/dev/null && tmp_file="$(mktemp "$path_abs/.preflight.XXXXXX" 2>/dev/null)"; then
    rm -f "$tmp_file"
    ok "writable directory: $path_abs"
  else
    fail "directory is not writable: $path_abs"
  fi
done

artifact_abs="$(abs_path "$ARTIFACT_ROOT")"
if is_positive_int "$MIN_FREE_GB"; then
  free_kb="$(df -Pk "$artifact_abs" 2>/dev/null | awk 'NR==2 {print $4}')"
  if [[ -n "$free_kb" ]]; then
    min_free_kb=$((MIN_FREE_GB * 1024 * 1024))
    free_gb=$((free_kb / 1024 / 1024))
    if [[ "$free_kb" -ge "$min_free_kb" ]]; then
      ok "artifact filesystem has ${free_gb}GB free"
    else
      fail "artifact filesystem has ${free_gb}GB free; need at least ${MIN_FREE_GB}GB"
    fi
  else
    warn "could not determine free disk space for $artifact_abs"
  fi
else
  fail "--min-free-gb must be a positive integer: $MIN_FREE_GB"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_line="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  if [[ -n "$gpu_line" ]]; then
    ok "nvidia-smi visible GPU: $gpu_line"
  else
    warn "nvidia-smi is available but did not return a GPU row"
  fi
else
  warn "nvidia-smi is not available; benchmark can run but GPU telemetry will be limited"
fi

echo "[summary] failures=$failures warnings=$warnings"
if [[ "$failures" -gt 0 ]]; then
  exit 1
fi
exit 0
