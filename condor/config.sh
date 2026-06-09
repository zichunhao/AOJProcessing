#!/bin/bash
# ============================================================================
# AOJ Condor pipeline -- SITE CONFIGURATION
# ----------------------------------------------------------------------------
# This is the ONE file to edit when porting between clusters.
# Pick the SUBMIT site with AOJ_SITE (default uaf), e.g.  export AOJ_SITE=lpc
# Everything else (job.sh, submit file, helpers) reads from here.
# ============================================================================

# Submit site: explicit AOJ_SITE wins, otherwise auto-detect from the hostname.
if [ -n "${AOJ_SITE:-}" ]; then
  SITE="$AOJ_SITE"
else
  case "$(hostname -f 2>/dev/null || hostname)" in
    cmslpc*|*.fnal.gov) SITE=lpc ;;
    uaf*|*.t2.ucsd.edu) SITE=uaf ;;
    *)                  SITE=uaf ;;   # default
  esac
fi

# ----------------------------------------------------------------------------
# Canonical AOJ store -- ALWAYS UAF (UCSD T2), matching the HH4b convention
#   root://redirector.t2.ucsd.edu:1095//store/user/zichun/...
# Outputs (.h5) AND the shipped CMSSW tarball live here, even for jobs that run
# at LPC (they read the tarball from / write h5 back to UAF over xrootd).
# Override with AOJ_STORE_XRD if you ever need a different destination.
# ----------------------------------------------------------------------------
STORE_XRD="${AOJ_STORE_XRD:-root://redirector.t2.ucsd.edu:1095//store/user/zichun/parcel/AOJ}"

# ---- CMSSW release (identical everywhere; base resolves from cvmfs) ----
CMSSW_VERSION="CMSSW_10_6_30"
SCRAM_ARCH="slc7_amd64_gcc700"
EL7_IMAGE="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/el7:x86_64"

# ---- Job sizing ----
FILES_PER_JOB="${FILES_PER_JOB:-3}"          # MINIAOD input files per Condor job
                                             # (~76 min/file on a slow glidein, ~6 min on a fast one)
NTHREADS="${NTHREADS:-1}"                    # cmsRun threads  (== request_cpus)
REQUEST_MEMORY="${REQUEST_MEMORY:-4000}"     # MB
REQUEST_DISK="${REQUEST_DISK:-20000000}"     # KB (~19 GB; holds staged MINIAOD + nano)
                                             # scales with FILES_PER_JOB x MINIAOD size
SALVAGE_PARALLEL="${SALVAGE_PARALLEL:-3}"    # local H5_maker jobs at once during salvage

case "$SITE" in
  uaf)
    # ----------------------- UCSD UAF / T2 -----------------------
    # STORE_XRD's fuse mount on this host (used by pack/merge/salvage for speed):
    STORE_LOCAL="${AOJ_STORE_LOCAL:-/ceph/cms/store/user/zichun/parcel/AOJ}"
    CONTAINER_BINDS="/ceph"                  # extra bind for local cmssw-el7 steps
    PROXY="${X509_USER_PROXY:-$HOME/.x509/x509up_u$(id -u)}"
    ;;
  lpc)
    # ----------------------- Fermilab LPC ------------------------
    LPCUSER="${LPCUSER:-$USER}"                # <-- your Fermilab username; only
                                              #     used if you send output to EOS
    CONTAINER_BINDS="/eos,/uscms,/uscms_data" # binds for local cmssw-el7 steps
    PROXY="${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}"
    STORE_LOCAL="${AOJ_STORE_LOCAL:-}"        # UCSD store isn't fuse-mounted at LPC
    # Output defaults to the UCSD ceph (STORE_XRD above) even from LPC. To keep
    # output on FNAL EOS instead, uncomment these two lines:
    #STORE_XRD="root://cmseos.fnal.gov//store/user/${LPCUSER}/parcel/AOJ"
    #STORE_LOCAL="/eos/uscms/store/user/${LPCUSER}/parcel/AOJ"
    # If LPC condor complains about the OS, add +DesiredOS = "EL7" to submit.templ.sub.
    ;;
  *)
    echo "config.sh: unknown SITE='$SITE' (expected uaf|lpc)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

CMSSW_TARBALL="${CMSSW_VERSION}.tar.gz"

export SITE CMSSW_VERSION SCRAM_ARCH EL7_IMAGE FILES_PER_JOB NTHREADS \
       REQUEST_MEMORY REQUEST_DISK SALVAGE_PARALLEL \
       STORE_XRD STORE_LOCAL CONTAINER_BINDS PROXY CMSSW_TARBALL
