#!/bin/bash
# ============================================================================
# Pack ttbar_semilep/ -> <STORE>/cmssw/${TTBAR_TARBALL}, the code that job.sh's
# optional Stage 3 (AOJ_TTBAR=also|only) fetches, and make sure <STORE>/ttbar/,
# where Stage 3 pushes tt_<chunk>.h5, exists. The tarball is kept apart from the
# CMSSW one, so updating the ttbar code never touches the release that queued
# jet-level jobs fetch.
#
# It packs the COMMITTED ttbar_semilep/ (git archive HEAD), so every tarball
# matches a commit, and refuses while ttbar_semilep/ has uncommitted changes.
# Each tarball name is written once and never replaced: queued jobs fetch it
# when they start, and the store refuses renames over xrootd, so an in-place
# update could not be atomic. To ship new ttbar code, commit it, bump the version
# in config.sh's TTBAR_TARBALL and run this again.
# ============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
AOJ="$(cd "$HERE/.." && pwd)"
# parse root://HOST//ABSPATH  ->  server=root://HOST/  dir=/ABSPATH
_t="${STORE_XRD#root://}"; _server="root://${_t%%//*}/"; _dir="/${_t#*//}"
LOCAL=""; [ -n "${STORE_LOCAL:-}" ] && [ -d "$STORE_LOCAL" ] && LOCAL=1

# remote_size <path below the store>: bytes, or nothing if absent
remote_size() {
  if [ -n "$LOCAL" ]; then
    [ -e "$STORE_LOCAL/$1" ] && stat -c %s "$STORE_LOCAL/$1"
  else
    timeout 120 xrdfs "$_server" stat "${_dir}/$1" 2>/dev/null | awk '$1 == "Size:" {print $2}'
  fi
  return 0
}

# <STORE>/ttbar/ and <STORE>/ttbar_skim/ (idempotent; also when the tarball itself is already there)
for _sub in ttbar ttbar_skim; do
  if [ -n "$LOCAL" ]; then
    mkdir -p "$STORE_LOCAL/cmssw" "$STORE_LOCAL/$_sub"
  elif ! timeout 120 xrdfs "$_server" stat "${_dir}/$_sub" > /dev/null 2>&1; then
    timeout 120 xrdfs "$_server" mkdir -p "${_dir}/$_sub"
  fi
done

have="$(remote_size "cmssw/${TTBAR_TARBALL}")"
if [ -n "$have" ]; then
  echo "pack_ttbar: ${STORE_XRD}/cmssw/${TTBAR_TARBALL} already exists (${have} bytes); it is never replaced." >&2
  echo "            For new ttbar code, bump TTBAR_TARBALL in config.sh. If that file is a broken" >&2
  echo "            upload that no job uses yet, remove it (xrdfs ${_server} rm ${_dir}/cmssw/${TTBAR_TARBALL})" >&2
  echo "            and run this again." >&2
  exit 1
fi

if [ -n "$(git -C "$AOJ" status --porcelain -- ttbar_semilep)" ]; then
  echo "pack_ttbar: ttbar_semilep/ has uncommitted changes; commit them first:" >&2
  git -C "$AOJ" status --short -- ttbar_semilep >&2
  exit 1
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git -C "$AOJ" archive --format=tar HEAD ttbar_semilep | gzip -n > "$TMP/$TTBAR_TARBALL"
size="$(stat -c %s "$TMP/$TTBAR_TARBALL")"
echo "Packed ttbar_semilep/ at $(git -C "$AOJ" rev-parse --short HEAD) -> $TTBAR_TARBALL (${size} bytes, $(tar tzf "$TMP/$TTBAR_TARBALL" | wc -l) entries)"

echo "Uploading -> ${STORE_XRD}/cmssw/${TTBAR_TARBALL}"
if [ -n "$LOCAL" ]; then
  cp "$TMP/$TTBAR_TARBALL" "$STORE_LOCAL/cmssw/${TTBAR_TARBALL}"
else
  xrdcp "$TMP/$TTBAR_TARBALL" "${STORE_XRD}/cmssw/${TTBAR_TARBALL}" || true
fi
got="$(remote_size "cmssw/${TTBAR_TARBALL}")"
if [ "$got" != "$size" ]; then
  echo "pack_ttbar: upload failed (${got:-no file} bytes on the store, ${size} expected); removing it." >&2
  if [ -n "$LOCAL" ]; then rm -f "$STORE_LOCAL/cmssw/${TTBAR_TARBALL}"
  else timeout 120 xrdfs "$_server" rm "${_dir}/cmssw/${TTBAR_TARBALL}" || true; fi
  exit 1
fi
echo "Done."
