#!/bin/bash
# Run every line of jobs.txt (NAME maker-args...) through ttbar_maker.py, NPAR at a time.  On a non-el9
# host everything runs inside one cmssw-el9 container (LCG_105a has no el8 build).
# Outputs: OUTDIR/NAME.h5 + OUTDIR/NAME.log; jobs whose .h5 exists are skipped (re-run to retry failures).
#   ./run_jobs.sh [NPAR] [OUTDIR] [extra ttbar_maker.py args, e.g. --preset paper / --set fj_count_min_pt=300]
HERE="$(cd "$(dirname "$0")" && pwd)"
NPAR="${1:-20}"
OUTDIR="${2:-out}"
shift 2 2>/dev/null
if [ "$(sed -n 's/.*release \([0-9]*\).*/\1/p' /etc/redhat-release)" = 9 ]; then
  bash "$HERE/_inner.sh" "$HERE" "$NPAR" "$OUTDIR" "$@"
else
  /cvmfs/cms.cern.ch/common/cmssw-el9 --command-to-run bash "$HERE/_inner.sh" "$HERE" "$NPAR" "$OUTDIR" "$@"
fi
