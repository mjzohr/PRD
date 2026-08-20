#!/usr/bin/env bash
# Slice raw ICBHI (development) and CWLSD (external) audio into 6s segments + manifests.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
echo "==> preprocessing ICBHI development audio"
$PY preprocessing.py
echo "==> preprocessing CWLSD external audio"
$PY preprocess_cwlsd.py
