#!/bin/bash
# ============================================================================
# Merge all per-job out_*.h5 on the UAF store into a single merged.h5.
# Run at a site that mounts the store (UAF).  Uses H5_merge.py inside el7.
# Usage:  merge.sh [output.h5]      (default: <STORE_LOCAL>/merged.h5)
# ============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
AOJ="$(cd "$HERE/.." && pwd)"

if [ -z "${STORE_LOCAL:-}" ] || [ ! -d "$STORE_LOCAL/h5" ]; then
  echo "merge: STORE_LOCAL/h5 ('$STORE_LOCAL/h5') not mounted -- run merge where the store is mounted (UAF)." >&2
  exit 1
fi
OUT="${1:-$STORE_LOCAL/merged.h5}"
shopt -s nullglob
INPUTS=("$STORE_LOCAL"/h5/out_*.h5)
if [ ${#INPUTS[@]} -eq 0 ]; then echo "merge: no $STORE_LOCAL/h5/out_*.h5 found."; exit 1; fi
echo "Merging ${#INPUTS[@]} file(s) -> $OUT"

RUN="$(mktemp /tmp/aoj_merge.XXXXXX.sh)"
{
  echo "source /cvmfs/cms.cern.ch/cmsset_default.sh"
  echo "export SCRAM_ARCH=$SCRAM_ARCH"
  echo "cd $AOJ"
  echo "eval \$(scram runtime -sh)"
  echo "python H5_merge.py $OUT ${INPUTS[*]}"
} > "$RUN"
/cvmfs/cms.cern.ch/common/cmssw-el7 -B "${CONTAINER_BINDS:-/ceph}" \
    --command-to-run bash "$RUN"
rm -f "$RUN"
echo "merge: wrote $OUT"
