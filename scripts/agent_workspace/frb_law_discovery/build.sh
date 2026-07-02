#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
STAMP="$(date +%y%m%d_frb_law_discovery_%H%M%S)"
TARGET_DIR="${REPO_ROOT}/logs/scripts/agent_workspace/${STAMP}"

mkdir -p "${TARGET_DIR}"
cp -a "${SCRIPT_DIR}/.gitignore" "${TARGET_DIR}/.gitignore"
cp -a "${SCRIPT_DIR}/README.md" "${TARGET_DIR}/README.md"
cp -a "${SCRIPT_DIR}/feature_extraction.py" "${TARGET_DIR}/feature_extraction.py"
cp -a "${SCRIPT_DIR}/run_sr_agent.py" "${TARGET_DIR}/run_sr_agent.py"
cp -al "${SCRIPT_DIR}/data" "${TARGET_DIR}/data"

cat <<EOF
Created workspace:
${TARGET_DIR}

Run Codex there with workspace-write access, outbound network enabled, and approvals only when it leaves the sandbox:

cd "${TARGET_DIR}"
codex --sandbox workspace-write --ask-for-approval on-request -c 'sandbox_workspace_write.network_access=true' "\$(cat README.md)"

Notes:
- network_access=true lets the agent call run_sr_agent.py without network approval prompts.
- workspace-write keeps file writes inside this workspace unless further approval is requested.
- README.md is passed as the initial prompt.
EOF
