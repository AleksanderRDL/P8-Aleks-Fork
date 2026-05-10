#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/clean_smoke_artifacts.sh [--yes]

Remove known local smoke/test artifact directories. Without --yes, this command
prints what would be removed and exits without deleting anything.
EOF
}

confirm=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)
      confirm=1
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

QDS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QDS_ROOT"

paths=(
  artifacts/benchmarks/artifact_layout_smoke
  artifacts/benchmarks/matrix_smoke
  artifacts/benchmarks/matrix_smoke_csv
  artifacts/benchmarks/smoke_range_matrix_patch
  artifacts/benchmarks/smoke_runtime
  artifacts/benchmarks/task7_smoke
  artifacts/benchmarks/task10_default_smoke
  artifacts/benchmarks/task10_tf32_smoke
  artifacts/benchmarks/task11_bs_smoke
  artifacts/benchmarks/task12_bf16_small
  artifacts/benchmarks/task12_tf32_small
  artifacts/benchmarks/task13_infbs1
  artifacts/benchmarks/task13_infbs4
  artifacts/benchmarks/tmux_layout_smoke
  artifacts/cache/matrix_smoke_csv
  artifacts/cache/smoke_cleaned_csv
  artifacts/results/smoke_cleaned_csv
  artifacts/results/smoke_synthetic
)

if [[ "$confirm" == "1" ]]; then
  echo "Deleting known smoke/test artifact directories:"
else
  echo "Dry run. These known smoke/test artifact directories would be deleted:"
fi

found=0
for path in "${paths[@]}"; do
  if [[ -e "$path" ]]; then
    found=1
    echo "  $path"
    if [[ "$confirm" == "1" ]]; then
      rm -rf -- "$path"
    fi
  fi
done

if [[ "$found" == "0" ]]; then
  echo "  (none found)"
fi

if [[ "$confirm" != "1" ]]; then
  echo "Pass --yes or run 'make clean-smoke-artifacts CONFIRM=1' to delete them."
fi
