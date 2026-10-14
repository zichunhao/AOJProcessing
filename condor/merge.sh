#!/bin/bash
# ============================================================================
# Merge per-job out_*.h5 on the UAF store, one sample at a time.
# Run at a site that mounts the store (UAF).  Uses H5_merge.py inside el7.
# Usage:  merge.sh                    data: out_2016[GH]_sub*.h5 -> <STORE_LOCAL>/merged.h5
#         merge.sh <label> [out.h5]   one sample, e.g. QCD_Pt300 or TTToHadronic:
#                                     out_<label>_sub*.h5 -> <STORE_LOCAL>/merged_<label>.h5
#                                     (a relative out.h5 is relative to the repo dir)
# Never merge different MC samples (or MC with data) together: each pT bin / process
# needs its own cross-section normalization (see the sumGenWeight attribute).
# An existing output is never overwritten (H5_merge.py truncates its output first, and
# merged.h5 is the source of the published shards) unless AOJ_MERGE_FORCE=1.
# ============================================================================
set -e
export AOJ_MODE=jet                  # jet-level outputs only (event mode: merge_event.sh)
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
AOJ="$(cd "$HERE/.." && pwd)"

if [ -z "${STORE_LOCAL:-}" ] || [ ! -d "$STORE_LOCAL/h5" ]; then
  echo "merge: STORE_LOCAL/h5 ('$STORE_LOCAL/h5') not mounted -- run merge where the store is mounted (UAF)." >&2
  exit 1
fi
LABEL="${1:-}"
shopt -s nullglob
if [ -z "$LABEL" ]; then
  OUT="$STORE_LOCAL/merged.h5"
  PATTERN="out_2016G_sub*.h5 out_2016H_sub*.h5"
  INPUTS=("$STORE_LOCAL"/h5/out_2016G_sub*.h5 "$STORE_LOCAL"/h5/out_2016H_sub*.h5)
else
  OUT="${2:-$STORE_LOCAL/merged_${LABEL}.h5}"
  PATTERN="out_${LABEL}_sub*.h5"
  INPUTS=("$STORE_LOCAL"/h5/out_${LABEL}_sub*.h5)
fi
case "$OUT" in /*) ;; *) OUT="$AOJ/$OUT" ;; esac   # H5_merge.py runs from $AOJ
if [ ${#INPUTS[@]} -eq 0 ]; then echo "merge: no $PATTERN in $STORE_LOCAL/h5/."; exit 1; fi
if [ -e "$OUT" ] && [ "${AOJ_MERGE_FORCE:-0}" != 1 ]; then
  echo "merge: $OUT already exists -- not overwriting it (move it away, or rerun with AOJ_MERGE_FORCE=1)." >&2
  exit 1
fi
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
