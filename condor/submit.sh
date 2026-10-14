#!/bin/bash
# ============================================================================
# condor_submit submit.sub -- from SUBMIT_DIR when config.sh sets one.
# ----------------------------------------------------------------------------
# LPC schedds cannot see the /eos FUSE mount: a cluster submitted from a dir on
# /eos goes HELD at once with "Cannot access initial working directory". So on
# LPC the job inputs (job.sh, submit.sub, joblist.txt, chunks/) are copied to
# SUBMIT_DIR (nobackup) and submitted from there; the job logs land in
# SUBMIT_DIR/logs. With SUBMIT_DIR empty (UAF) this submits in place.
# Jet mode only: event mode submits submit_event.sub in place (see run.sh).
# AOJ_SUBMIT_HOLD=1 submits the cluster HELD, to queue it behind work that is
# still running; start it later with condor_release -name <schedd> <cluster>.
# Queued jobs read job.sh and their chunk file from SUBMIT_DIR when they START,
# so this refuses to replace a different job.sh or chunk file there (another
# cluster may still need it) unless AOJ_SUBMIT_FORCE=1.
# ============================================================================
set -e
export AOJ_MODE=jet                  # jet-level files only (event mode: see run.sh)
HERE="$(cd "$(dirname "$0")" && pwd)"
# make_jobs.sh fixed the ttbar mode in submit.sub; take it from there, so that
# "AOJ_TTBAR=only ./run.sh jobs && ./run.sh submit" also picks its SUBMIT_DIR.
if [ -z "${AOJ_TTBAR:-}" ] && [ -f "$HERE/submit.sub" ]; then
  _m="$(sed -n 's/^environment .* AOJ_TTBAR=\([a-z]*\).*/\1/p' "$HERE/submit.sub")"
  if [ -n "$_m" ]; then export AOJ_TTBAR="$_m"; fi
fi
source "$HERE/config.sh"
cd "$HERE"
if [ ! -s joblist.txt ]; then echo "joblist.txt empty -- nothing to submit."; exit 0; fi

# -ef: a SUBMIT_DIR that is condor/ itself under another spelling means "in place"
if [ -n "${SUBMIT_DIR:-}" ] && ! [ "$SUBMIT_DIR" -ef "$HERE" ]; then
  mkdir -p "$SUBMIT_DIR/chunks" "$SUBMIT_DIR/logs"
  if [ "${AOJ_SUBMIT_FORCE:-}" != 1 ]; then
    clash=""
    if [ -e "$SUBMIT_DIR/job.sh" ] && ! cmp -s job.sh "$SUBMIT_DIR/job.sh"; then clash="job.sh"; fi
    while read -r _ds c _rest; do
      t="$SUBMIT_DIR/chunks/$(basename "$c")"
      if [ -e "$t" ] && ! cmp -s "$c" "$t"; then clash="$clash $c"; fi
    done < joblist.txt
    if [ -n "$clash" ]; then
      echo "submit: $SUBMIT_DIR already holds a different version of:$(echo "$clash" | cut -c1-200)" >&2
      echo "        Jobs queued from there read them when they start. Use another AOJ_SUBMIT_DIR, or" >&2
      echo "        AOJ_SUBMIT_FORCE=1 once no queued job there needs the old ones." >&2
      exit 1
    fi
  fi
  cp -f job.sh submit.sub joblist.txt "$SUBMIT_DIR/"
  # only the chunk files this joblist needs (column 2 = chunks/<label>_subNNN)
  awk '{print $2}' joblist.txt | xargs -I{} cp -f {} "$SUBMIT_DIR/chunks/"
  echo "submitting from $SUBMIT_DIR (logs: $SUBMIT_DIR/logs)"
  cd "$SUBMIT_DIR"
fi
if [ "${AOJ_SUBMIT_HOLD:-}" = 1 ]; then
  condor_submit -append "hold = True" submit.sub
else
  condor_submit submit.sub
fi
