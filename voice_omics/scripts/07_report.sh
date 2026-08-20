#!/usr/bin/env bash
# Aggregate every run into paper-ready tables under results/reports/.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
echo "==> building reports"
$PY report.py
