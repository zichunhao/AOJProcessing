#!/bin/bash
# ============================================================================
# AOJ per-chunk Condor worker  (runs INSIDE the EL7 singularity image)
# ----------------------------------------------------------------------------
# Self-contained & site-agnostic. Everything it needs arrives via the job
# environment (set by submit.sub from config.sh):
#     STORE_XRD CMSSW_VERSION CMSSW_TARBALL SCRAM_ARCH NTHREADS
#     optional: AOJ_TTBAR TTBAR_TARBALL TTBAR_PRESET TTBAR_SUBDIR TTBAR_PREFILTER AOJ_MAX_EVENTS (see below)
# Args:  <dataset>  <chunkfile-basename>  <output.h5>
#
# Flow:  fetch CMSSW tarball (xrootd) -> cmsenv -> STAGE inputs to scratch
#        -> cmsRun PFNano -> H5_maker -> xrdcp the small .h5 to <STORE>/h5/.
#        Inputs are copied locally first because streaming MINIAOD from eospublic
#        over the WAN is latency-bound off-site (~10x slower); the big NanoAOD and
#        the staged inputs never leave the worker scratch dir.
#
# Optional Stage 3: the semileptonic-ttbar tag-and-probe
# (ttbar_semilep/ttbar_maker.py, see ttbar_semilep/LPC_PROMPT.md) on the same
# NanoAOD, pushed to <STORE>/ttbar/tt_<chunk>.h5. Selected by AOJ_TTBAR:
#   (unset)  jet-level only: the default, unchanged
#   also     MC only: jet-level as above, then Stage 3; a Stage 3 failure only
#            warns (the jet-level .h5 is already on the store)
#   only     Stage 3 instead of H5_maker (no jet-level .h5); a failure holds the
#            job. MC or data (SingleMuon / SingleElectron: golden JSON). With
#            TTBAR_PREFILTER=1 (config.sh's default for "only") Stage 1 runs
#            ttbar_semilep/pfnano_ttskim.py, which writes only the events with a
#            lepton (pT > 20) and an AK8 jet (pT > 200), looser than any preset;
#            the MC normalisation (Runs/genEventSumw) still counts every event.
# Its code is <STORE>/cmssw/${TTBAR_TARBALL} (pack_ttbar.sh), unpacked over the
# CMSSW tarball's copy right after that tarball, before any staging, so a missing
# or broken one holds the job at once. TTBAR_PRESET (set with TTBAR_SUBDIR by
# gen_submit.sh) picks a non-default ttbar_maker preset, written to
# <STORE>/${TTBAR_SUBDIR}/ instead of ttbar/. AOJ_MAX_EVENTS=N passes maxEvents=N to
# cmsRun (pilots/tests only: the outputs then cover only N events per chunk).
# ============================================================================
set -e
echo "===== AOJ job start $(date) on $(hostname) ====="
DATASET="$1"; CHUNK="$2"; OUTNAME="$3"
: "${NTHREADS:=1}"
: "${CMSSW_VERSION:=CMSSW_10_6_30}"
: "${CMSSW_TARBALL:=${CMSSW_VERSION}.tar.gz}"
: "${SCRAM_ARCH:=slc7_amd64_gcc700}"
export SCRAM_ARCH
# Resilience to transient eospublic (CERN Open Data) connection errors at staging.
export XRD_CONNECTIONRETRY=8 XRD_REQUESTTIMEOUT=600 XRD_STREAMTIMEOUT=120
WORK="$(pwd)"
echo "dataset=$DATASET  chunk=$CHUNK  out=$OUTNAME"
echo "store=$STORE_XRD  scratch=$WORK  threads=$NTHREADS"

# data vs MC selection (QCD*, TT*, ST_*, WJets*, ZJets*, DY*, WW, WZ, ZZ, *_mc*, MC_* => MC; keep it in step
# with the MC labels of ttbar_semilep/samples.py). ttbar (TT*) also gets the per-jet top_match label in
# gen_info (no jets are dropped).
H5OPTS=""
case "$DATASET" in
  QCD*|TT*|ST_*|WJets*|ZJets*|DY*|WW|WZ|ZZ|*_mc*|MC_*) STYPE=MC;   CONFIG=pfnano_mc_2016UL_OpenData.py;   NANO=nano_mc2016post.root;;
  *)                               STYPE=data; CONFIG=pfnano_data_2016UL_OpenData.py; NANO=nano_data2016.root;;
esac
case "$DATASET" in TT*) H5OPTS="--top_label";; esac
case "${AOJ_TTBAR:-}" in
  "") ;;
  also|only)
    [ "$STYPE" = MC ] || [ "${AOJ_TTBAR}" = only ] || { echo "FATAL: AOJ_TTBAR=also is MC-only ($DATASET is data; use only)"; exit 2; }
    : "${TTBAR_TARBALL:=ttbar_semilep_v1.tar.gz}"
    : "${TTBAR_PREFILTER:=0}"
    [ "${TTBAR_PREFILTER}" = 0 ] || [ "${AOJ_TTBAR}" = only ] || { echo "FATAL: TTBAR_PREFILTER=1 needs AOJ_TTBAR=only"; exit 2; }
    echo "ttbar stage: AOJ_TTBAR=${AOJ_TTBAR}  code=${STORE_XRD}/cmssw/${TTBAR_TARBALL}${TTBAR_PRESET:+  preset=${TTBAR_PRESET}} -> ${TTBAR_SUBDIR:-ttbar}/  pre-filter=${TTBAR_PREFILTER}";;
  *) echo "FATAL: AOJ_TTBAR='${AOJ_TTBAR}' (expected unset, also or only)"; exit 2;;
esac
if [ -n "${AOJ_MAX_EVENTS:-}" ]; then echo "TEST RUN: AOJ_MAX_EVENTS=${AOJ_MAX_EVENTS}: only that many events per chunk"; fi

# ---- bring up CMSSW (system /usr/bin/xrdcp in the image is usable now) ----
source /cvmfs/cms.cern.ch/cmsset_default.sh
echo "Fetching ${STORE_XRD}/cmssw/${CMSSW_TARBALL} ..."
xrdcp -f "${STORE_XRD}/cmssw/${CMSSW_TARBALL}" "${WORK}/${CMSSW_TARBALL}"
tar xzf "${WORK}/${CMSSW_TARBALL}" && rm -f "${WORK}/${CMSSW_TARBALL}"

cd "${WORK}/${CMSSW_VERSION}/src"
scram b ProjectRename || echo "WARN: ProjectRename returned non-zero (continuing)"
eval "$(scram runtime -sh)"
cd AOJProcessing
if [ -n "${AOJ_TTBAR:-}" ]; then
  echo "Fetching ${STORE_XRD}/cmssw/${TTBAR_TARBALL} ..."
  rm -rf ttbar_semilep
  if ! { xrdcp -f "${STORE_XRD}/cmssw/${TTBAR_TARBALL}" "${WORK}/${TTBAR_TARBALL}" &&
         tar xzf "${WORK}/${TTBAR_TARBALL}" && [ -f ttbar_semilep/ttbar_maker.py ]; }; then
    echo "FATAL: cannot fetch/unpack ${TTBAR_TARBALL} (was condor/pack_ttbar.sh run for it?)"; exit 2
  fi
  rm -f "${WORK}/${TTBAR_TARBALL}"
  if [ "${TTBAR_PREFILTER:-0}" = 1 ] && [ ! -f ttbar_semilep/pfnano_ttskim.py ]; then
    echo "FATAL: TTBAR_PREFILTER=1 but ${TTBAR_TARBALL} has no ttbar_semilep/pfnano_ttskim.py"; exit 2
  fi
fi
cp "${WORK}/${CHUNK}" ./remote_chunk.txt
echo "=== input files ($(wc -l < remote_chunk.txt)) ==="; cat remote_chunk.txt

# ---- Stage inputs to local scratch ----
# Streaming MINIAOD from eospublic over the WAN is latency-bound (PFNano touches
# many branches/event) and ran ~10x slower off-site. Copy each input locally
# first (one bulk sequential transfer), then point cmsRun at the local files so
# processing becomes CPU-bound, like on-site.
echo "===== Staging inputs to scratch  ($(date)) ====="
mkdir -p "${WORK}/inputs"
: > local_chunk.txt
i=0
while read -r f; do
  [ -z "$f" ] && continue
  loc="${WORK}/inputs/in_${i}.root"
  echo "  staging $f"
  n=0
  until xrdcp -f -s "$f" "$loc"; do
    n=$((n+1)); [ "$n" -ge 6 ] && { echo "FATAL: failed to stage $f after $n tries (eospublic unreachable?)"; exit 1; }
    echo "  stage retry $n (eospublic hiccup); waiting $((n*30))s ..."; sleep $((n*30))
  done
  echo "file:${loc}" >> local_chunk.txt
  i=$((i+1))
done < remote_chunk.txt
echo "staged $(wc -l < local_chunk.txt) file(s), $(du -sh "${WORK}/inputs" 2>/dev/null | cut -f1) total"

# ---- Stage 1: PFNano ----
# If cmsRun fails but left a NanoAOD, go on with it rather than failing at once.
# Note: the CMSSW_10_6_30 NanoAOD output writes its trees only when the file is
# closed, so a crash before that leaves no readable Events tree; H5_maker (or
# ttbar_maker) then fails and the job is held (condor_release reruns it).
echo "===== Stage 1: cmsRun ${CONFIG}  ($(date)) ====="
set +e
# Thin the per-event log spam (~20 MB/job otherwise): keep every 10000th
# "Begin processing" line; drop the MC-only Rivet HTXS warning (meaningless here).
thin() { awk '/Begin processing the/ { if (++n % 10000) next } /HiggsTemplateCrossSections/ { next } { print; fflush() }'; }
CMSRUN_CFG="${CONFIG}"
if [ "${TTBAR_PREFILTER:-0}" = 1 ]; then
  export AOJ_TTSKIM_BASE="${CONFIG}"; CMSRUN_CFG=ttbar_semilep/pfnano_ttskim.py
  echo "Stage 1 through the ttbar pre-filter: ${CMSRUN_CFG} on ${CONFIG}"
fi
cmsRun "${CMSRUN_CFG}" inputFiles_load=local_chunk.txt nThreads="${NTHREADS}" ${AOJ_MAX_EVENTS:+maxEvents=${AOJ_MAX_EVENTS}} > >(thin) 2> >(thin >&2)
cmsrc=$?
set -e
if [ "${cmsrc}" -ne 0 ]; then
  echo "WARNING: cmsRun exited ${cmsrc} (known rare ParticleNet/DeepBoostedJet segfault)."
  if [ ! -s "${NANO}" ]; then echo "FATAL: no partial ${NANO} produced -- failing job."; exit "${cmsrc}"; fi
  echo "Salvaging the events written before the crash from partial ${NANO} ($(ls -l "${NANO}" | awk '{print $5}') bytes)."
fi

if [ "${AOJ_TTBAR:-}" != only ]; then
# ---- Stage 2: HDF5 ----
echo "===== Stage 2: H5_maker  ($(date)) ====="
if [ "$STYPE" = data ]; then
  python H5_maker.py -i "${NANO}" -o "${OUTNAME}" --sample_type data \
         -j Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt
else
  python H5_maker.py -i "${NANO}" -o "${OUTNAME}" --sample_type MC ${H5OPTS}
fi

# ---- push the small output ----
echo "===== Push ${OUTNAME} -> ${STORE_XRD}/h5/  ($(date)) ====="
xrdcp -f "${OUTNAME}" "${STORE_XRD}/h5/${OUTNAME}"
fi

# ---- Stage 3 (optional): semileptonic-ttbar tag-and-probe, see the header ----
# Called as "ttbar_stage || ...", where set -e is off: every step returns explicitly.
# py3env.sh runs python3 from the LCG_105a centos7 view, isolated from cmsenv.
ttbar_stage() {
  local tt="tt_${OUTNAME#out_}" sub="${TTBAR_SUBDIR:-ttbar}" pre=() typ=(--sample_type MC)
  [ -n "${TTBAR_PRESET:-}" ] && pre=(--preset "${TTBAR_PRESET}")
  [ "$STYPE" = data ] && typ=(--sample_type data -j Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt)
  echo "===== Stage 3: ttbar_maker -> ${tt}  ($(date)) ====="
  bash ttbar_semilep/py3env.sh ttbar_semilep/ttbar_maker.py -i "${NANO}" -o "${tt}" "${typ[@]}" \
       --era 2016postVFP --trigger lepton_or_jetht --store one_probe "${pre[@]}" || return 1
  echo "===== Push ${tt} -> ${STORE_XRD}/${sub}/  ($(date)) ====="
  xrdcp -f "${tt}" "${STORE_XRD}/${sub}/${tt}" || return 1
  rm -f "${tt}"
}
case "${AOJ_TTBAR:-}" in
  also) ttbar_stage || echo "WARNING: Stage 3 failed: ${OUTNAME} is on the store, tt_${OUTNAME#out_} is not (rerun with AOJ_TTBAR=only)";;
  only) ttbar_stage || { echo "FATAL: Stage 3 (ttbar_maker) failed"; exit 1; };;
esac

# ---- drop the big intermediate ----
rm -f "${NANO}" "${OUTNAME}"; rm -rf "${WORK}/inputs"
echo "===== AOJ job done $(date) ====="
