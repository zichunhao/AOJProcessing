#!/bin/bash
# (run_jobs.sh helper; el9) $1 = rnd dir, $2 = parallel jobs, $3 = output dir, $4... = extra maker args
export X509_USER_PROXY="$HOME/.x509/x509up_u$(id -u)"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105a/x86_64-el9-gcc11-opt/setup.sh >/dev/null 2>&1
cd "$1" || exit 1
NPAR="$2"
export OUTDIR="$3"
shift 3
export EXTRA="$*"
mkdir -p "$OUTDIR"
run1() {
  local name=$1; shift
  if [ -s "$OUTDIR/$name.h5" ]; then echo "$name skip"; return; fi
  # shellcheck disable=SC2086
  if python3 ../ttbar_maker.py "$@" $EXTRA -o "$OUTDIR/$name.h5.tmp" --store one_probe --timeout 900 > "$OUTDIR/$name.log" 2>&1; then
    mv "$OUTDIR/$name.h5.tmp" "$OUTDIR/$name.h5" && echo "$name ok"
  else
    echo "$name FAILED"
  fi
}
export -f run1
xargs -P "$NPAR" -L 1 bash -c 'run1 "$@"' _ < jobs.txt
