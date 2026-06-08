#!/bin/bash
# ============================================================================
# End-to-end smoke test: one Condor job over a single input file, exercising
# the full worker path (fetch CMSSW tarball -> PFNano -> H5_maker -> xrdcp).
# Verifies the pipeline works before launching the full batch.
# ============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
AOJ="$(cd "$HERE/.." && pwd)"
cd "$HERE"
mkdir -p chunks logs

SAMPLE="${1:-2016G}"
head -1 "$AOJ/file_lists/${SAMPLE}.txt" > "chunks/smoke_sub000"
echo "${SAMPLE} chunks/smoke_sub000 smoke_sub000 out_smoke_sub000.h5" > joblist.txt
bash "$HERE/gen_submit.sh"
echo "Smoke joblist (1 file from ${SAMPLE}):"; cat joblist.txt
condor_submit submit.sub
echo
echo "Watch with:  condor_q ; tail -f $HERE/logs/out_smoke_sub000.out"
echo "On success the file appears at: ${STORE_XRD}/h5/out_smoke_sub000.h5"
