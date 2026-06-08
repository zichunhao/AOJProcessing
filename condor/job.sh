#!/bin/bash
# ============================================================================
# AOJ per-chunk Condor worker  (runs INSIDE the EL7 singularity image)
# ----------------------------------------------------------------------------
# Self-contained & site-agnostic. Everything it needs arrives via the job
# environment (set by submit.sub from config.sh):
#     STORE_XRD CMSSW_VERSION CMSSW_TARBALL SCRAM_ARCH NTHREADS
# Args:  <dataset>  <chunkfile-basename>  <output.h5>
#
# Flow:  fetch CMSSW tarball (xrootd) -> cmsenv -> cmsRun PFNano
#        -> H5_maker -> xrdcp the small .h5 to <STORE>/h5/.  The 10-15 GB
#        intermediate NanoAOD never leaves the worker scratch dir.
# ============================================================================
set -e
echo "===== AOJ job start $(date) on $(hostname) ====="
DATASET="$1"; CHUNK="$2"; OUTNAME="$3"
: "${NTHREADS:=1}"
: "${CMSSW_VERSION:=CMSSW_10_6_30}"
: "${CMSSW_TARBALL:=${CMSSW_VERSION}.tar.gz}"
: "${SCRAM_ARCH:=slc7_amd64_gcc700}"
export SCRAM_ARCH
WORK="$(pwd)"
echo "dataset=$DATASET  chunk=$CHUNK  out=$OUTNAME"
echo "store=$STORE_XRD  scratch=$WORK  threads=$NTHREADS"

# data vs MC selection (QCD_* or *_mc* => MC)
case "$DATASET" in
  QCD*|*_mc*|MC_*) STYPE=MC;   CONFIG=pfnano_mc_2016UL_OpenData.py;   NANO=nano_mc2016post.root;;
  *)               STYPE=data; CONFIG=pfnano_data_2016UL_OpenData.py; NANO=nano_data2016.root;;
esac

# ---- bring up CMSSW (system /usr/bin/xrdcp in the image is usable now) ----
source /cvmfs/cms.cern.ch/cmsset_default.sh
echo "Fetching ${STORE_XRD}/cmssw/${CMSSW_TARBALL} ..."
xrdcp -f "${STORE_XRD}/cmssw/${CMSSW_TARBALL}" "${WORK}/${CMSSW_TARBALL}"
tar xzf "${WORK}/${CMSSW_TARBALL}" && rm -f "${WORK}/${CMSSW_TARBALL}"

cd "${WORK}/${CMSSW_VERSION}/src"
scram b ProjectRename || echo "WARN: ProjectRename returned non-zero (continuing)"
eval "$(scram runtime -sh)"
cd AOJProcessing
cp "${WORK}/${CHUNK}" ./input_chunk.txt
echo "=== input files ($(wc -l < input_chunk.txt)) ==="; cat input_chunk.txt

# ---- Stage 1: PFNano ----
# Tolerate the rare DeepBoostedJet/ParticleNet segfault on a pathological jet:
# cmsRun writes the NanoAOD incrementally, so on a crash we still salvage every
# event written before it and let H5_maker process those (ROOT recovers the
# unclosed file). Only fail the job if NO (partial) output was produced.
echo "===== Stage 1: cmsRun ${CONFIG}  ($(date)) ====="
set +e
cmsRun "${CONFIG}" inputFiles_load=input_chunk.txt nThreads="${NTHREADS}"
cmsrc=$?
set -e
if [ "${cmsrc}" -ne 0 ]; then
  echo "WARNING: cmsRun exited ${cmsrc} (known rare ParticleNet/DeepBoostedJet segfault)."
  if [ ! -s "${NANO}" ]; then echo "FATAL: no partial ${NANO} produced -- failing job."; exit "${cmsrc}"; fi
  echo "Salvaging the events written before the crash from partial ${NANO} ($(ls -l "${NANO}" | awk '{print $5}') bytes)."
fi

# ---- Stage 2: HDF5 ----
echo "===== Stage 2: H5_maker  ($(date)) ====="
if [ "$STYPE" = data ]; then
  python H5_maker.py -i "${NANO}" -o "${OUTNAME}" --sample_type data \
         -j Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt
else
  python H5_maker.py -i "${NANO}" -o "${OUTNAME}" --sample_type MC
fi

# ---- push the small output, drop the big intermediate ----
echo "===== Push ${OUTNAME} -> ${STORE_XRD}/h5/  ($(date)) ====="
xrdcp -f "${OUTNAME}" "${STORE_XRD}/h5/${OUTNAME}"
rm -f "${NANO}" "${OUTNAME}"
echo "===== AOJ job done $(date) ====="
