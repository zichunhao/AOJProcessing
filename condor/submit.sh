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
# ============================================================================
set -e
export AOJ_MODE=jet                  # jet-level files only (event mode: see run.sh)
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
cd "$HERE"
if [ ! -s joblist.txt ]; then echo "joblist.txt empty -- nothing to submit."; exit 0; fi

# -ef: a SUBMIT_DIR that is condor/ itself under another spelling means "in place"
if [ -n "${SUBMIT_DIR:-}" ] && ! [ "$SUBMIT_DIR" -ef "$HERE" ]; then
  mkdir -p "$SUBMIT_DIR/chunks" "$SUBMIT_DIR/logs"
  cp -f job.sh submit.sub joblist.txt "$SUBMIT_DIR/"
  # only the chunk files this joblist needs (column 2 = chunks/<label>_subNNN)
  awk '{print $2}' joblist.txt | xargs -I{} cp -f {} "$SUBMIT_DIR/chunks/"
  echo "submitting from $SUBMIT_DIR (logs: $SUBMIT_DIR/logs)"
  cd "$SUBMIT_DIR"
fi
condor_submit submit.sub
