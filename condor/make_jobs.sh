#!/bin/bash
# ============================================================================
# Build per-job chunk files + joblist.txt (+ submit.sub) from regen_lists.txt
# ----------------------------------------------------------------------------
# regen_lists.txt   (one source file-list to (re)process per line):
#     <dataset>  <path-to-filelist>  <label>
# e.g.
#     2016H  split_2016H/chunk_03   2016H_03      # regenerate one bad chunk
#     2016G  file_lists/2016G.txt   2016G         # regenerate a whole dataset
#
# Each file-list is split into FILES_PER_JOB-line chunks; one Condor job per
# chunk.  Jobs whose output .h5 already exists on storage are skipped
# (idempotent -- safe to re-run after failures/evictions).
# ============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
AOJ="$(cd "$HERE/.." && pwd)"
cd "$HERE"
mkdir -p chunks logs

REGEN="${1:-$HERE/regen_lists.txt}"
if [ ! -s "$REGEN" ]; then
  echo "make_jobs: '$REGEN' is empty/missing -> no regeneration jobs."
  : > "$HERE/joblist.txt"
  bash "$HERE/gen_submit.sh"
  exit 0
fi

: > "$HERE/joblist.txt"
njobs=0; skipped=0

# Outputs already on the store -> idempotent skip. Works whether the store is
# locally mounted (UAF ceph) or remote (LPC submitting to the UCSD store): list
# the h5 dir once via the mount if present, else via xrootd.
existing="$(mktemp)"
if [ -n "${STORE_LOCAL:-}" ] && [ -d "${STORE_LOCAL}/h5" ]; then
  ls "${STORE_LOCAL}/h5" 2>/dev/null > "$existing" || true
else
  # parse root://HOST//ABSPATH  ->  server=root://HOST/  dir=/ABSPATH/h5
  _t="${STORE_XRD#root://}"; _server="root://${_t%%//*}/"; _dir="/${_t#*//}/h5"
  xrdfs "$_server" ls "$_dir" 2>/dev/null | sed 's:.*/::' > "$existing" || true
fi

while read -r dataset filelist label; do
  [ -z "${dataset:-}" ] && continue
  flist="$filelist"
  [ -f "$flist" ] || flist="$AOJ/$filelist"
  if [ ! -f "$flist" ]; then echo "  WARN: missing file-list '$filelist' -- skipping"; continue; fi

  rm -f chunks/${label}_sub*
  split -l "$FILES_PER_JOB" -d -a 3 "$flist" "chunks/${label}_sub"
  for c in chunks/${label}_sub*; do
    sub="$(basename "$c")"          # e.g. 2016H_03_sub000
    out="out_${sub}.h5"
    if grep -qxF "$out" "$existing"; then
      skipped=$((skipped+1)); continue
    fi
    # columns: dataset  chunkpath(rel to condor/)  chunkbase  outname
    echo "${dataset} chunks/${sub} ${sub} ${out}" >> "$HERE/joblist.txt"
    njobs=$((njobs+1))
  done
done < "$REGEN"

rm -f "$existing"
bash "$HERE/gen_submit.sh"
echo "make_jobs: $njobs job(s) queued, $skipped already on storage (FILES_PER_JOB=$FILES_PER_JOB)."
