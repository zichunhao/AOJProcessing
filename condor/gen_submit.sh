#!/bin/bash
# Render submit.templ.sub -> submit.sub using values from config.sh.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
# job.sh's optional Stage 3 (AOJ_TTBAR) and test knob (AOJ_MAX_EVENTS) go into the
# job environment only when set, so the default submit.sub is unchanged. Stage 3
# jobs also get their own log names (logs/out_<chunk>.h5.ttbar-<mode>.out/.err/.log),
# so they never write over a jet-level job's logs for the same chunk.
EXTRA_ENV=""; LOGTAG=""
if [ -n "${AOJ_TTBAR:-}" ]; then
  EXTRA_ENV=" AOJ_TTBAR=${AOJ_TTBAR} TTBAR_TARBALL=${TTBAR_TARBALL}"; LOGTAG=".ttbar-${AOJ_TTBAR}"
  # the jobs fetch it first thing; refuse to queue them if it is not on the store
  _t="${STORE_XRD#root://}"; _server="root://${_t%%//*}/"; _dir="/${_t#*//}"
  if [ -n "${STORE_LOCAL:-}" ] && [ -d "$STORE_LOCAL/cmssw" ]; then
    [ -s "$STORE_LOCAL/cmssw/${TTBAR_TARBALL}" ]
  else
    timeout 120 xrdfs "$_server" stat "${_dir}/cmssw/${TTBAR_TARBALL}" > /dev/null 2>&1
  fi || { echo "gen_submit: ${STORE_XRD}/cmssw/${TTBAR_TARBALL} not found: run pack_ttbar.sh first" >&2; exit 1; }
  echo "gen_submit: AOJ_TTBAR=${AOJ_TTBAR}, ttbar code ${TTBAR_TARBALL}"
fi
if [ -n "${AOJ_MAX_EVENTS:-}" ]; then
  EXTRA_ENV="${EXTRA_ENV} AOJ_MAX_EVENTS=${AOJ_MAX_EVENTS}"
  echo "gen_submit: WARNING AOJ_MAX_EVENTS=${AOJ_MAX_EVENTS}: every job reads only that many events (pilots/tests only)" >&2
fi
sed -e "s#@PROXY@#${PROXY}#g" \
    -e "s#@EL7_IMAGE@#${EL7_IMAGE}#g" \
    -e "s#@STORE_XRD@#${STORE_XRD}#g" \
    -e "s#@SCRAM_ARCH@#${SCRAM_ARCH}#g" \
    -e "s#@NTHREADS@#${NTHREADS}#g" \
    -e "s#@CMSSW_VERSION@#${CMSSW_VERSION}#g" \
    -e "s#@CMSSW_TARBALL@#${CMSSW_TARBALL}#g" \
    -e "s#@EXTRA_ENV@#${EXTRA_ENV}#g" \
    -e "s#@LOGTAG@#${LOGTAG}#g" \
    -e "s#@REQUEST_MEMORY@#${REQUEST_MEMORY}#g" \
    -e "s#@REQUEST_DISK@#${REQUEST_DISK}#g" \
    "$HERE/submit.templ.sub" > "$HERE/submit.sub"
echo "wrote $HERE/submit.sub  (site=$SITE)"
