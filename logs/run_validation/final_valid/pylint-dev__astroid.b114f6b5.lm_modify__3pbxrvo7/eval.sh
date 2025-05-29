#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
export PYTHONPATH=/testbed
: '>>>>> Start Test Output'
pytest --disable-warnings --color=no --tb=no --verbose
: '>>>>> End Test Output'
