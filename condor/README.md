# AOJ Condor pipeline

HTCondor batch driver for the Aspen Open Jets processing (replaces the
local-tmux `step1/2/3.sh`, which lost everything when the session was killed).

**One Condor job per small chunk of input files, end-to-end:**
`MINIAOD (xrootd) → cmsRun PFNano → H5_maker → xrdcp the small .h5 to the store`.
The 10–15 GB intermediate NanoAOD stays on worker scratch and is never stored.
Jobs are **idempotent**: a chunk whose `.h5` already exists on the store is
skipped, so re-running after evictions/failures only does the missing work.

## Design (why it survives a killed session)
- HTCondor owns the jobs, not a shell/tmux — they keep running and auto-restart
  on eviction.
- Nothing relies on `/home` (not mounted on workers). Each job pulls a packed
  CMSSW tarball and pushes output over **xrootd**; inputs are public eospublic.
- CMSSW_10_6_30 is `slc7`, so jobs run inside the EL7 container
  (`+SingularityImage`, the same image used by `cmssw-el7`).

## Output location
Everything lands on the **UCSD ceph** (UAF T2), matching the HH4b convention:
`root://redirector.t2.ucsd.edu:1095//store/user/zichun/parcel/AOJ`
(= `~/ceph/parcel/AOJ` = `/ceph/cms/store/user/zichun/parcel/AOJ`)
```
parcel/AOJ/
  cmssw/CMSSW_10_6_30.tar.gz   # shipped release (built PFNano plugin + code)
  h5/out_<label>.h5            # one per Condor job
  merged.h5                    # final merged sample
```
This is the canonical store even for jobs submitted from LPC.

## Quick start (UAF)
```bash
cd CMSSW_10_6_30/src/AOJProcessing/condor
./run.sh smoke               # 1-file end-to-end test; check h5/out_smoke_sub000.h5
./run.sh salvage-and-submit  # pack + validate existing + salvage good + submit the rest
condor_q                     # watch
./run.sh merge               # when all jobs done -> merged.h5
```

`salvage-and-submit` does, in order:
1. **pack** the local release → `store/cmssw/`.
2. **validate** the existing `run_*/nano_data2016.root` (open in ROOT; a file is
   "complete" only if it is not recovered/zombie and has non-empty `Events`+`Runs`
   trees — size alone is not enough).
3. **salvage** (background, local): run only `H5_maker` on the complete files,
   reusing their PFNano compute → `store/h5/`.
4. **submit** Condor jobs that redo PFNano+H5 only for the incomplete/missing
   chunks (split `FILES_PER_JOB` at a time).

Individual steps are also available: `./run.sh pack|validate|salvage|jobs|submit|merge`.

## Knobs (`config.sh`, or env overrides)
| var | default | meaning |
|---|---|---|
| `AOJ_SITE` | `uaf` | submit site: `uaf` \| `lpc` |
| `FILES_PER_JOB` | `10` | MINIAOD files per Condor job |
| `NTHREADS` | `1` | cmsRun threads (= `request_cpus`) |
| `REQUEST_MEMORY` | `4000` | MB |
| `REQUEST_DISK` | `8000000` | KB (holds the transient NanoAOD) |
| `AOJ_STORE_XRD` | UCSD ceph | output store (xrootd) |

## Regenerate from scratch / other samples
`make_jobs.sh` reads `regen_lists.txt` (`<dataset> <file-list> <label>` per line).
`validate.sh` writes it automatically for the incomplete chunks. To process a
whole dataset (or MC) instead:
```bash
printf "2016G file_lists/2016G.txt 2016G\n2016H file_lists/2016H.txt 2016H\n" > regen_lists.txt
./run.sh jobs && ./run.sh submit
# or simply:  ./run.sh scratch
```
MC is auto-detected from the label (`QCD*`/`*_mc*` → MC config, no JSON cert).

## Porting to Fermilab LPC
1. Build CMSSW_10_6_30 + PFNano on LPC (same as UAF).
2. `export AOJ_SITE=lpc` and set `LPCUSER`. Output still defaults to UCSD ceph
   ("output to UAF"); set `AOJ_STORE_XRD`/`AOJ_STORE_LOCAL` to use LPC EOS instead.
3. `voms-proxy-init -voms cms` (config picks up `$X509_USER_PROXY`).
4. `./run.sh scratch` (the validate/salvage steps are UAF-only — they act on the
   local `run_*/` partials, which exist only on UAF).
If LPC condor complains about the OS, add `+DesiredOS = "EL7"` in
`submit.templ.sub` (usually unnecessary with `+SingularityImage`).
