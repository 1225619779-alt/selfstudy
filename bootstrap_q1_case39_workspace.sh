#!/usr/bin/env bash
set -euo pipefail

# Safe workspace bootstrap for a separate case39 expansion worktree.
# Usage:
#   bash bootstrap_q1_case39_workspace.sh ~/projects/DDET-MTD ~/projects/DDET-MTD-q1-case39 q1_case39_expansion

SRC_REPO="${1:-$HOME/projects/DDET-MTD}"
TARGET_DIR="${2:-$HOME/projects/DDET-MTD-q1-case39}"
BRANCH_NAME="${3:-q1_case39_expansion}"

if [[ ! -d "$SRC_REPO/.git" ]]; then
  echo "ERROR: source repo is not a git repository: $SRC_REPO" >&2
  exit 1
fi

if [[ -e "$TARGET_DIR" ]]; then
  echo "ERROR: target path already exists: $TARGET_DIR" >&2
  exit 1
fi

cd "$SRC_REPO"
CURRENT_HEAD="$(git rev-parse --short HEAD)"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
STATUS_SUMMARY="$(git status --short || true)"

if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
  echo "Using existing local branch: $BRANCH_NAME"
else
  git branch "$BRANCH_NAME"
  echo "Created new branch: $BRANCH_NAME from HEAD $CURRENT_HEAD"
fi

git worktree add "$TARGET_DIR" "$BRANCH_NAME"

echo
echo "Workspace created."
echo "  source repo : $SRC_REPO"
echo "  source head : $CURRENT_HEAD"
echo "  source branch: $CURRENT_BRANCH"
echo "  new worktree: $TARGET_DIR"
echo "  new branch  : $BRANCH_NAME"
echo
if [[ -n "$STATUS_SUMMARY" ]]; then
  echo "NOTE: source repo had local changes at bootstrap time."
  echo "$STATUS_SUMMARY"
  echo
fi

echo "Recommended next steps:"
echo "  cd $TARGET_DIR"
echo "  # optional: create a separate venv for case39 isolation"
echo "  python3 -m venv .venv_q1_case39"
echo "  source .venv_q1_case39/bin/activate"
echo "  pip install -r requirements.txt  # or your project-specific setup"
echo "  python phase3_case39_capability_audit.py --repo_root . --output metric/case39/preflight/case39_capability_audit.json"
