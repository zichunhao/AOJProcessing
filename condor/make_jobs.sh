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

# What counts as already done: the jet-level out_<chunk>.h5 in h5/, or for
# AOJ_TTBAR=only (no jet-level output) tt_<chunk>.h5 in ttbar/.
OUTDIR=h5; PREFIX=out_
if [ "${AOJ_TTBAR:-}" = only ]; then OUTDIR=ttbar; PREFIX=tt_; fi

# list_store <dir> <file>: names in <STORE>/<dir> -> <file>, through the local
# mount if present, else xrootd with a timeout.
list_store() {
  if [ -n "${STORE_LOCAL:-}" ] && [ -d "${STORE_LOCAL}/$1" ]; then
    ls "${STORE_LOCAL}/$1" 2>/dev/null > "$2" || true
    return 0
  fi
  # parse root://HOST//ABSPATH  ->  server=root://HOST/  dir=/ABSPATH/<dir>
  local _t _server _dir
  _t="${STORE_XRD#root://}"; _server="root://${_t%%//*}/"; _dir="/${_t#*//}/$1"
  if timeout 180 xrdfs "$_server" ls "$_dir" > "$2.raw" 2>/dev/null; then
    sed 's:.*/::' "$2.raw" > "$2"; rm -f "$2.raw"
  else
    echo "ERROR: listing the store's $1/ via xrootd timed out/failed (common from off-site;" >&2
    echo "       ttbar/ exists once pack_ttbar.sh has run)." >&2
    echo "       Re-run with an explicit missing list:  AOJ_MISSING=missing.txt ./run.sh jobs" >&2
    rm -f "$2" "$2.raw"; exit 1
  fi
}

# Decide which chunks to (re)submit. Two modes:
#  - AOJ_MISSING=<file>: submit ONLY the chunk labels it lists (one per line, e.g.
#    "2016G_sub014"); the store is NOT listed. Use this from LPC, where listing
#    the UCSD store through the redirector (a directory ls) is slow/hangs.
#  - otherwise (idempotent skip): list the store's h5/ (ttbar/ for AOJ_TTBAR=only)
#    once and skip outputs already there. AOJ_TTBAR=also skips chunks that have
#    out_ (no second jet-level run) and lists those without tt_ in missing_tt.txt,
#    to backfill with AOJ_TTBAR=only.
existing="$(mktemp)"; existing_tt=""; no_tt=0
if [ -n "${AOJ_MISSING:-}" ] && [ -f "$AOJ_MISSING" ]; then
  echo "make_jobs: AOJ_MISSING set -> submitting only the $(grep -c . "$AOJ_MISSING") listed chunk(s); skipping store listing."
else
  list_store "$OUTDIR" "$existing"
  if [ "${AOJ_TTBAR:-}" = also ]; then
    existing_tt="$(mktemp)"; list_store ttbar "$existing_tt"; : > "$HERE/missing_tt.txt"
  fi
fi

while read -r dataset filelist label; do
  [ -z "${dataset:-}" ] && continue
  flist="$filelist"
  [ -f "$flist" ] || flist="$AOJ/$filelist"
  if [ ! -f "$flist" ]; then echo "  WARN: missing file-list '$filelist' -- skipping"; continue; fi

  rm -f chunks/${label}_sub*
  # 3-digit chunk suffix (the established naming); widen only when >1000 chunks.
  # Skip an empty list (split writes no chunk for it).
  if ! grep -q . "$flist"; then echo "  WARN: empty file-list '$filelist' -- skipping"; continue; fi
  nlines=$(grep -c '' "$flist")          # every line split -l counts, blank ones too
  nchunk=$(( (nlines + FILES_PER_JOB - 1) / FILES_PER_JOB ))
  width=3; [ "$nchunk" -gt 1000 ] && width=${#nchunk}
  split -l "$FILES_PER_JOB" -d -a "$width" "$flist" "chunks/${label}_sub"
  for c in chunks/${label}_sub*; do
    sub="$(basename "$c")"          # e.g. 2016G_sub014
    out="out_${sub}.h5"
    if [ -n "${AOJ_MISSING:-}" ] && [ -f "$AOJ_MISSING" ]; then
      grep -qxF "$sub" "$AOJ_MISSING" || continue          # queue only listed chunks
    elif grep -qxF "${PREFIX}${sub}.h5" "$existing"; then
      skipped=$((skipped+1))
      if [ -n "$existing_tt" ] && ! grep -qxF "tt_${sub}.h5" "$existing_tt"; then
        echo "$sub" >> "$HERE/missing_tt.txt"; no_tt=$((no_tt+1))
      fi
      continue
    fi
    # columns: dataset  chunkpath(rel to condor/)  chunkbase  outname
    echo "${dataset} chunks/${sub} ${sub} ${out}" >> "$HERE/joblist.txt"
    njobs=$((njobs+1))
  done
done < "$REGEN"

rm -f "$existing" ${existing_tt:+"$existing_tt"}
bash "$HERE/gen_submit.sh"
echo "make_jobs: $njobs job(s) queued, $skipped already on storage (FILES_PER_JOB=$FILES_PER_JOB)."
if [ "$no_tt" -gt 0 ]; then
  echo "make_jobs: $no_tt of those have out_<chunk>.h5 but no tt_<chunk>.h5 (condor/missing_tt.txt); for them:"
  echo "           AOJ_TTBAR=only AOJ_MISSING=missing_tt.txt ./run.sh jobs && ./run.sh submit"
fi
