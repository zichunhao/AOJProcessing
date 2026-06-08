#!/bin/bash
# ============================================================================
# Salvage step (UAF only): for PFNano ROOT files that validate.sh marked
# complete, run ONLY H5_maker locally (reusing the expensive PFNano compute)
# and write out_<ds>_<NN>.h5 to the UAF store.  Throttled to SALVAGE_PARALLEL.
# Reads salvage_lists.txt:  <ds> <root-path> <out.h5>
# ============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
AOJ="$(cd "$HERE/.." && pwd)"
LIST="$HERE/salvage_lists.txt"

if [ ! -s "$LIST" ]; then echo "salvage: nothing to do (salvage_lists.txt empty)."; exit 0; fi
if [ -z "${STORE_LOCAL:-}" ] || [ ! -d "$STORE_LOCAL" ]; then
  echo "salvage: STORE_LOCAL ('$STORE_LOCAL') not mounted -- salvage is UAF-only." >&2; exit 1
fi
mkdir -p "$STORE_LOCAL/h5"
CERT="Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
echo "Salvaging $(wc -l < "$LIST") complete chunk(s), ${SALVAGE_PARALLEL} at a time -> $STORE_LOCAL/h5/"

RUN="$(mktemp /tmp/aoj_salvage.XXXXXX.sh)"
{
  echo "source /cvmfs/cms.cern.ch/cmsset_default.sh"
  echo "export SCRAM_ARCH=$SCRAM_ARCH"
  echo "cd $AOJ"
  echo "eval \$(scram runtime -sh)"
  echo "n=0"
  echo "while read -r ds path out; do"
  echo "  [ -z \"\$ds\" ] && continue"
  echo "  if [ -e \"$STORE_LOCAL/h5/\$out\" ]; then echo \"[skip] \$out exists\"; continue; fi"
  echo "  echo \"[salvage] \$out  <-  \$path\""
  echo "  ( python H5_maker.py -i \"$AOJ/\$path\" -o \"$STORE_LOCAL/h5/\$out\" --sample_type data -j $CERT \\"
  echo "      && echo \"[done] \$out\" || { echo \"[FAIL] \$out\"; rm -f \"$STORE_LOCAL/h5/\$out\"; } ) &"
  echo "  n=\$((n+1)); [ \$((n % $SALVAGE_PARALLEL)) -eq 0 ] && wait"
  echo "done < $LIST"
  echo "wait"
} > "$RUN"

echo "Running salvage inside cmssw-el7 (log: $HERE/salvage.log)"
/cvmfs/cms.cern.ch/common/cmssw-el7 -B "${CONTAINER_BINDS:-/ceph}" \
    --command-to-run bash "$RUN" > "$HERE/salvage.log" 2>&1
rm -f "$RUN"
echo "salvage: finished. produced $(ls -1 "$STORE_LOCAL"/h5/out_*.h5 2>/dev/null | wc -l) file(s); see $HERE/salvage.log"
