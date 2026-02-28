#!/usr/bin/env bash
# verify_connection.sh
# ====================
# Quick sanity check — run after setup_zenml.py to confirm the stack is live.
# Usage (Git Bash / WSL / bash):
#   bash verify_connection.sh
#
# On Windows PowerShell without bash, run the commands manually (see below).

PYTHON="C:/Users/youss/AppData/Local/Programs/Python/Python311/python.exe"
ZENML="C:/Users/youss/AppData/Local/Programs/Python/Python311/Scripts/zenml.exe"

echo ""
echo "=============================================="
echo "  ZenML Connection Verification"
echo "=============================================="

echo ""
echo "[1] ZenML status:"
"$ZENML" status

echo ""
echo "[2] Stack list (active stack has * prefix):"
"$ZENML" stack list

echo ""
echo "[3] Active stack detail:"
"$ZENML" stack describe

echo ""
echo "[4] Service health:"
"$PYTHON" check_infra.py

echo ""
echo "=============================================="
echo "  If cifar_stack has * --> you are ready!"
echo "  Run: $PYTHON run_pipeline.py"
echo "=============================================="
echo ""
