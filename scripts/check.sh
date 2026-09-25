#!/usr/bin/env bash
# Run the same checks CI runs, locally and for free.
#
# On a private repository GitHub Actions minutes are a limited monthly allowance, so
# the cheapest verification is the one that never touches a runner. This script is the
# local equivalent of the `check` job in .github/workflows/tests.yml -- if it passes
# here, CI has nothing new to tell you.
#
#   ./scripts/check.sh            use the existing .venv, or create one
#   ./scripts/check.sh --fresh    rebuild .venv from scratch first
#
# Exits non-zero on the first failure, so it is safe to use as a pre-push hook:
#   ln -sf ../../scripts/check.sh .git/hooks/pre-push

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

VENV=".venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip install -q --retries 5 --timeout 60"

if [ "${1:-}" = "--fresh" ]; then
    echo "→ rebuilding $VENV"
    rm -rf "$VENV"
fi

if [ ! -x "$PY" ]; then
    echo "→ creating $VENV"
    python3 -m venv "$VENV" || { echo "✗ could not create a virtualenv"; exit 1; }
fi

echo "→ installing (skipped silently if PyPI is unreachable)"
if ! ($PIP --upgrade pip && $PIP -e ".[dev]"); then
    echo "⚠ install failed — continuing with whatever is already in $VENV"
fi

fail=0

run() {
    local label="$1"; shift
    printf '\n\033[1m→ %s\033[0m\n' "$label"
    if "$@"; then
        printf '\033[32m✓ %s\033[0m\n' "$label"
    else
        printf '\033[31m✗ %s\033[0m\n' "$label"
        fail=1
    fi
}

# Lint is optional locally: ruff may be missing if PyPI was unreachable, and that
# should not masquerade as a lint failure.
if [ -x "$VENV/bin/ruff" ]; then
    run "lint (ruff)" "$VENV/bin/ruff" check .
else
    echo "⚠ ruff not installed — skipping lint (CI still checks it)"
fi

run "tests" "$PY" -m pytest -q

run "demo" "$VENV/bin/pocketscribe" demo --output /tmp/pocketscribe_demo.html
run "demo (consensus)" "$VENV/bin/pocketscribe" demo --consensus \
    --output /tmp/pocketscribe_consensus.html

# The consensus report must distinguish the three evidence classes; a report that
# silently lost that distinction would still be a valid HTML file.
printf '\n\033[1m→ consensus report content\033[0m\n'
if grep -q "same-family-only" /tmp/pocketscribe_consensus.html \
   && grep -q "cross-family" /tmp/pocketscribe_consensus.html; then
    printf '\033[32m✓ consensus report content\033[0m\n'
else
    printf '\033[31m✗ consensus report is missing the agreement classes\033[0m\n'
    fail=1
fi

printf '\n'
if [ "$fail" -eq 0 ]; then
    printf '\033[32m════ all checks passed ════\033[0m\n'
    printf 'Reports: /tmp/pocketscribe_demo.html  /tmp/pocketscribe_consensus.html\n'
else
    printf '\033[31m════ something failed — see above ════\033[0m\n'
fi
exit "$fail"
