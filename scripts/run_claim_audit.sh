#!/usr/bin/env bash
set -euo pipefail

to_windows_path() {
  local path="$1"
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -w "$path"
  elif [[ "$path" =~ ^/mnt/([A-Za-z])/(.*)$ ]]; then
    local drive="${BASH_REMATCH[1]^^}"
    local rest="${BASH_REMATCH[2]//\//\\}"
    printf '%s:\\%s' "$drive" "$rest"
  elif [[ "$path" =~ ^/([A-Za-z])/(.*)$ ]]; then
    local drive="${BASH_REMATCH[1]^^}"
    local rest="${BASH_REMATCH[2]//\//\\}"
    printf '%s:\\%s' "$drive" "$rest"
  else
    printf '%s' "$path"
  fi
}

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN=(python)
  PY_SRC="${PWD}/src"
  PY_SEP=":"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN=(python.exe)
  PY_SRC="$(to_windows_path "${PWD}/src")"
  PY_SEP=";"
elif command -v py.exe >/dev/null 2>&1; then
  PYTHON_BIN=(py.exe -3)
  PY_SRC="$(to_windows_path "${PWD}/src")"
  PY_SEP=";"
else
  echo "Could not find python on PATH" >&2
  exit 1
fi
export PYTHONPATH="${PY_SRC}${PYTHONPATH:+${PY_SEP}${PYTHONPATH}}"
"${PYTHON_BIN[@]}" -c 'import runpy, sys; sys.path.insert(0, sys.argv[1]); runpy.run_path(sys.argv[2], run_name="__main__")' "$PY_SRC" experiments/run_claim_audit.py
