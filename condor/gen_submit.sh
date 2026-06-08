#!/bin/bash
# Render submit.templ.sub -> submit.sub using values from config.sh.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
sed -e "s#@PROXY@#${PROXY}#g" \
    -e "s#@EL7_IMAGE@#${EL7_IMAGE}#g" \
    -e "s#@STORE_XRD@#${STORE_XRD}#g" \
    -e "s#@SCRAM_ARCH@#${SCRAM_ARCH}#g" \
    -e "s#@NTHREADS@#${NTHREADS}#g" \
    -e "s#@CMSSW_VERSION@#${CMSSW_VERSION}#g" \
    -e "s#@CMSSW_TARBALL@#${CMSSW_TARBALL}#g" \
    -e "s#@REQUEST_MEMORY@#${REQUEST_MEMORY}#g" \
    -e "s#@REQUEST_DISK@#${REQUEST_DISK}#g" \
    "$HERE/submit.templ.sub" > "$HERE/submit.sub"
echo "wrote $HERE/submit.sub  (site=$SITE)"
