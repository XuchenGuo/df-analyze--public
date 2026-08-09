#!/bin/bash

set -euo pipefail

TESTS=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT="$(dirname "$TESTS")"
PYTEST="${PYTEST:-$ROOT/.venv/bin/pytest}"

cd "$ROOT"
export DF_ANALYZE_RUN_REAL_TABPFN_TESTS=1
"$PYTEST" test/test_real_tabpfn_integration.py -m licensed_integration -x
