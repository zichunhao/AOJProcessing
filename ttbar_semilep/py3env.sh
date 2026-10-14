#!/bin/bash
# ============================================================================
# Run a ttbar_semilep python3 tool in an ISOLATED python3 environment.
# (Adapted from event_level/py3env.sh, so that ttbar_semilep does not need event_level/.)
# ----------------------------------------------------------------------------
# The tools need python3 + numpy/awkward/uproot/h5py; CMSSW_10_6_30 only has python2.  They are started
# from `env -i` so CMSSW's PYTHONPATH / LD_LIBRARY_PATH (cmsenv) never leak in.
#
#   bash ttbar_semilep/py3env.sh <script.py> [args...]
#
# EVT_PYENV (default auto):
#   lcg      LCG_105a view for this OS: x86_64-centos7-gcc11-opt on EL7 (e.g. inside the EL7 job image),
#            x86_64-el9-gcc11-opt on EL9; python 3.9.12, numpy 1.23.5, awkward 1.10.3, uproot 4.3.7,
#            h5py 3.8.0 (the versions tested).  Override the view with EVT_LCG_VIEW=<dir>.
#   cmssw14  fallback if /cvmfs/sft.cern.ch is not visible: CMSSW_14_2_3 externals via a one-time
#            `scram project` in EVT_PYENV_DIR (default $TMPDIR/aoj_py3env).
#   auto     lcg if its setup.sh is readable, else cmssw14.
# On an EL8 host (no LCG_105a build; the el9 view needs libcrypt.so.2) the command is re-run inside
# /cvmfs/cms.cern.ch/common/cmssw-el9 (set AOJ_PY3ENV_NO_CONTAINER=1 to disable).
# Passed through the clean environment: AOJ_TT_* / AOJ_EVT_* (test configuration), X509_USER_PROXY and
# X509_CERT_DIR (remote xrootd reads).
# ============================================================================
set -e
SCRIPT="${1:-}"
[ -n "$SCRIPT" ] || { echo "usage: py3env.sh <script.py> [args...]" >&2; exit 2; }
shift
[ -f "$SCRIPT" ] || { echo "py3env: no such script $SCRIPT" >&2; exit 2; }
SCRIPT="$(cd "$(dirname "$SCRIPT")" && pwd)/$(basename "$SCRIPT")"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

os_major="$(sed -n 's/.*release \([0-9][0-9]*\).*/\1/p' /etc/redhat-release 2>/dev/null | head -1)"
if [ "$os_major" = 8 ] && [ -z "${AOJ_PY3ENV_NO_CONTAINER:-}" ]; then
  EL9=/cvmfs/cms.cern.ch/common/cmssw-el9
  [ -x "$EL9" ] || { echo "py3env: FATAL EL8 host and no $EL9 (LCG_105a has no el8 build)" >&2; exit 3; }
  echo "py3env: EL8 host -> re-running inside cmssw-el9" >&2
  ARGS=()
  for a in "$@"; do ARGS+=("$(printf '%q' "$a")"); done
  # env vars survive into the container; the inner call sees EL9 and takes the normal path
  exec "$EL9" --command-to-run bash -c "cd $(printf '%q' "$PWD") && bash $(printf '%q' "$SELF") $(printf '%q' "$SCRIPT") ${ARGS[*]}"
fi
case "$os_major" in
  7) LCG_PLAT=x86_64-centos7-gcc11-opt; CMSSW14_ARCH=slc7_amd64_gcc12 ;;
  9) LCG_PLAT=x86_64-el9-gcc11-opt;     CMSSW14_ARCH=el9_amd64_gcc12 ;;
  *) LCG_PLAT=x86_64-el9-gcc11-opt;     CMSSW14_ARCH=el9_amd64_gcc12
     echo "py3env: WARNING unrecognised OS '$(cat /etc/redhat-release 2>/dev/null)'; trying el9 builds" >&2 ;;
esac
LCG_VIEW="${EVT_LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_105a/${LCG_PLAT}}"
MODE="${EVT_PYENV:-auto}"
if [ "$MODE" = auto ]; then
  if [ -r "${LCG_VIEW}/setup.sh" ]; then MODE=lcg; else MODE=cmssw14; fi
fi

TMPD="${TMPDIR:-${_CONDOR_SCRATCH_DIR:-/tmp}}"
CLEAN_ENV=(env -i HOME="${HOME:-$PWD}" USER="${USER:-aoj}" TMPDIR="$TMPD" PATH=/usr/bin:/bin
           OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
           PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1)
for _v in $(compgen -e | grep -E '^(AOJ_TT_|AOJ_EVT_|X509_USER_PROXY$|X509_CERT_DIR$)' || true); do
  CLEAN_ENV+=("$_v=${!_v}")
done

case "$MODE" in
  lcg)
    [ -r "${LCG_VIEW}/setup.sh" ] || { echo "py3env: FATAL ${LCG_VIEW}/setup.sh not readable (is /cvmfs/sft.cern.ch mounted?); try EVT_PYENV=cmssw14" >&2; exit 3; }
    echo "py3env: LCG view ${LCG_VIEW}" >&2
    exec "${CLEAN_ENV[@]}" bash -c '
      source "$0/setup.sh" >/dev/null 2>&1
      python3 -c "import numpy, awkward, uproot, h5py" || { echo "py3env: FATAL $0 lacks numpy/awkward/uproot/h5py" >&2; exit 3; }
      exec python3 "$@"' "$LCG_VIEW" "$SCRIPT" "$@"
    ;;
  cmssw14)
    BASE="${EVT_PYENV_DIR:-$TMPD/aoj_py3env}"
    echo "py3env: CMSSW_14_2_3 ($CMSSW14_ARCH) externals in $BASE" >&2
    exec "${CLEAN_ENV[@]}" SCRAM_ARCH="$CMSSW14_ARCH" bash -c '
      set -e
      here="$PWD"; base="$0"
      source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1
      mkdir -p "$base"; cd "$base"
      [ -d CMSSW_14_2_3/src ] || scram project CMSSW CMSSW_14_2_3 >/dev/null
      cd CMSSW_14_2_3/src; eval "$(scram runtime -sh)"; cd "$here"
      python3 -c "import numpy, awkward, uproot, h5py" || { echo "py3env: FATAL CMSSW_14_2_3 lacks numpy/awkward/uproot/h5py" >&2; exit 3; }
      exec python3 "$@"' "$BASE" "$SCRIPT" "$@"
    ;;
  *)
    echo "py3env: unknown EVT_PYENV='$MODE' (auto|lcg|cmssw14)" >&2; exit 2 ;;
esac
