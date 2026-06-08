#!/bin/bash
# ============================================================================
# AOJ Condor pipeline -- orchestrator
# ----------------------------------------------------------------------------
#   ./run.sh pack       pack the local CMSSW release -> store/cmssw/
#   ./run.sh validate   validate existing run_*/ PFNano files (UAF)
#   ./run.sh salvage    H5_maker on complete files locally (UAF, backgroundable)
#   ./run.sh jobs       build chunks + joblist.txt + submit.sub from regen_lists.txt
#   ./run.sh submit     condor_submit submit.sub
#   ./run.sh smoke      one-file end-to-end Condor smoke test
#   ./run.sh merge      merge all out_*.h5 -> merged.h5 (UAF)
#
#   ./run.sh salvage-and-submit   pack + validate + salvage(bg) + submit regen jobs
#   ./run.sh scratch              pack + regenerate EVERYTHING via Condor (e.g. LPC)
# ============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
cmd="${1:-help}"

case "$cmd" in
  pack)     bash "$HERE/pack_cmssw.sh" ;;
  validate) bash "$HERE/validate.sh" ;;
  salvage)  bash "$HERE/salvage.sh" ;;
  jobs)     bash "$HERE/make_jobs.sh" ;;
  submit)   cd "$HERE"; [ -s joblist.txt ] && condor_submit submit.sub || echo "joblist.txt empty -- nothing to submit." ;;
  smoke)    bash "$HERE/smoke.sh" "${2:-2016G}" ;;
  merge)    bash "$HERE/merge.sh" "${2:-}" ;;

  salvage-and-submit)
    bash "$HERE/pack_cmssw.sh"
    bash "$HERE/validate.sh"
    bash "$HERE/make_jobs.sh"                 # condor jobs for the INCOMPLETE chunks
    echo ">> launching salvage in background (log: $HERE/salvage.log)"
    nohup bash "$HERE/salvage.sh" >/dev/null 2>&1 &
    cd "$HERE"; [ -s joblist.txt ] && condor_submit submit.sub || echo "No regeneration jobs needed."
    ;;

  scratch)
    bash "$HERE/pack_cmssw.sh"
    printf "2016G file_lists/2016G.txt 2016G\n2016H file_lists/2016H.txt 2016H\n" > "$HERE/regen_lists.txt"
    bash "$HERE/make_jobs.sh"
    cd "$HERE"; condor_submit submit.sub
    ;;

  help|*)
    sed -n '2,20p' "$0"
    ;;
esac
