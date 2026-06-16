#!/bin/bash
# ============================================================================
# Pack the local CMSSW release (built PFNano plugin + AOJProcessing code) into
# a tarball on storage, so each Condor job can fetch & cmsenv it.
# Excludes the giant NanoAOD outputs and scratch dirs -> ~250 MB.
# ============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
# condor/  is  <REL>/src/AOJProcessing/condor  -> REL is 3 levels up
REL_DIR="$(cd "$HERE/../../.." && pwd)"          # .../CMSSW_10_6_30
PARENT="$(cd "$REL_DIR/.." && pwd)"              # dir containing CMSSW_10_6_30
if [ "$(basename "$REL_DIR")" != "$CMSSW_VERSION" ]; then
  echo "ERROR: expected $CMSSW_VERSION at $REL_DIR" >&2; exit 1
fi

# Fail fast if the golden-JSON cert is missing -- H5_maker needs it for data, and
# it must be inside the tarball (it's tracked in git; a fresh clone has it).
CERT_FILE="$HERE/../Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
if [ ! -f "$CERT_FILE" ]; then
  echo "ERROR: golden-JSON cert missing: $CERT_FILE" >&2
  echo "  Required by H5_maker for data. It is tracked in git -- run 'git pull'." >&2
  exit 1
fi

TARBALL="/tmp/${CMSSW_TARBALL}"
echo "Packing $CMSSW_VERSION from $PARENT -> $TARBALL ..."
cd "$PARENT"
tar czf "$TARBALL" \
  --exclude="${CMSSW_VERSION}/src/AOJProcessing/run_*" \
  --exclude="${CMSSW_VERSION}/src/AOJProcessing/split_*" \
  --exclude="${CMSSW_VERSION}/src/AOJProcessing/condor/chunks" \
  --exclude="${CMSSW_VERSION}/src/AOJProcessing/condor/logs" \
  --exclude="${CMSSW_VERSION}/src/AOJProcessing/condor/*.tar.gz" \
  --exclude="${CMSSW_VERSION}/tmp" \
  --exclude="*.swp" --exclude="*.un~" \
  "${CMSSW_VERSION}"
echo "Tarball size: $(du -h "$TARBALL" | cut -f1)"

echo "Uploading -> ${STORE_XRD}/cmssw/${CMSSW_TARBALL}"
if [ -n "${STORE_LOCAL:-}" ] && [ -d "$STORE_LOCAL" ]; then
  mkdir -p "$STORE_LOCAL/cmssw"
  cp -f "$TARBALL" "$STORE_LOCAL/cmssw/${CMSSW_TARBALL}"
else
  xrdcp -f "$TARBALL" "${STORE_XRD}/cmssw/${CMSSW_TARBALL}"
fi
rm -f "$TARBALL"
echo "Done."
