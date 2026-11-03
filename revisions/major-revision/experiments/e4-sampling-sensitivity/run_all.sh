#!/bin/bash
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
for v in reference uniform_s1 uniform_s2 uniform_s3 type_floor batch_balanced mad_qc cap10000; do
  $PY run.py $v
done
$PY analyze.py > analysis.log 2>&1
echo ALLDONE
