#!/usr/bin/env bash
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
NC="\033[0m"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_CONTEXT="$REPO_DIR"
DOCKER_BUILD_TIMEOUT="${DOCKER_BUILD_TIMEOUT:-900}"
INFER_TIMEOUT="${INFER_TIMEOUT:-300}"

log() { printf "%b\n" "$*"; }
pass() { printf "%b\n" "${GREEN}[PASS]${NC} $*"; }
fail() { printf "%b\n" "${RED}[FAIL]${NC} $*"; }
hint() { printf "%b\n" "${YELLOW}[HINT]${NC} $*"; }
stop_at() { printf "%b\n" "${RED}${BOLD}Stopped at:${NC} $*"; exit 1; }

run_with_timeout() {
  local timeout_s="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$timeout_s" "$@"
  else
    # macOS default: no `timeout` in PATH
    perl -e 'alarm shift; exec @ARGV' "$timeout_s" "$@"
  fi
}

log "${BOLD}Step 1/3: Running baseline inference${NC} ..."
INFER_OK=false
INFER_OUTPUT=$(cd "$REPO_DIR" && run_with_timeout "$INFER_TIMEOUT" python inference.py 2>&1) && INFER_OK=true
if [ "$INFER_OK" = true ]; then
  pass "inference.py completed"
  printf "%s\n" "$INFER_OUTPUT" | tail -20
else
  fail "inference.py failed or timed out (timeout=${INFER_TIMEOUT}s)"
  printf "%s\n" "$INFER_OUTPUT" | tail -50
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Docker build${NC} ..."
if [ -f "$DOCKER_CONTEXT/Dockerfile" ]; then
  log "  Found Dockerfile in $DOCKER_CONTEXT"
else
  fail "Missing Dockerfile in repo root"
  stop_at "Step 2"
fi

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker Desktop"
  stop_at "Step 2"
fi

BUILD_OK=false
BUILD_OUTPUT=$(cd "$REPO_DIR" && run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true
if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."
if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true
if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"
exit 0
