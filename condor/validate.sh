#!/bin/bash
# ============================================================================
# Validate existing run_<ds>_NN/nano_data2016.root files, then split them into:
#   salvage_lists.txt : <ds> <root-path> out_<ds>_<NN>.h5   (complete -> just H5)
#   regen_lists.txt   : <ds> split_<ds>/chunk_<NN> <ds>_<NN> (bad -> redo PFNano)
# Mapping assumes step1.sh's convention: run_<ds>_NN came from split_<ds>/chunk_NN.
# ============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
AOJ="$(cd "$HERE/.." && pwd)"
cd "$AOJ"

mapfile -t ROOTS < <(ls -1 run_2016G_*/nano_data2016.root run_2016H_*/nano_data2016.root 2>/dev/null || true)
: > "$HERE/salvage_lists.txt"
: > "$HERE/regen_lists.txt"
if [ ${#ROOTS[@]} -eq 0 ]; then
  echo "validate: no run_*/nano_data2016.root found -- nothing to salvage."
  exit 0
fi
echo "Validating ${#ROOTS[@]} existing PFNano files inside cmssw-el7 ..."

RUN="$(mktemp /tmp/aoj_validate.XXXXXX.sh)"
{
  echo "source /cvmfs/cms.cern.ch/cmsset_default.sh"
  echo "export SCRAM_ARCH=$SCRAM_ARCH"
  echo "cd $AOJ"
  echo "eval \$(scram runtime -sh)"
  echo "python $HERE/validate.py ${ROOTS[*]}"
} > "$RUN"
/cvmfs/cms.cern.ch/common/cmssw-el7 --command-to-run bash "$RUN" \
    > "$HERE/validation.tsv" 2> "$HERE/validation.err" || { cat "$HERE/validation.err" >&2; exit 1; }
rm -f "$RUN"

echo "---- validation report ----"
column -t "$HERE/validation.tsv" 2>/dev/null || cat "$HERE/validation.tsv"
echo "---------------------------"

while IFS=$'\t' read -r path ok n reason; do
  [ -z "${path:-}" ] && continue
  d="$(dirname "$path")"; rest="${d#run_}"; ds="${rest%_*}"; nn="${rest##*_}"
  if [ "$ok" = "OK" ]; then
    echo "$ds $path out_${ds}_${nn}.h5"          >> "$HERE/salvage_lists.txt"
  else
    echo "$ds split_${ds}/chunk_${nn} ${ds}_${nn}" >> "$HERE/regen_lists.txt"
  fi
done < "$HERE/validation.tsv"

echo "validate: $(wc -l < "$HERE/salvage_lists.txt") complete (salvage), $(wc -l < "$HERE/regen_lists.txt") incomplete (regenerate)."
