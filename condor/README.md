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
`root://redir.t2.ucsd.edu:1095//store/user/zichun/parcel/AOJ`
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
./run.sh merge               # when all jobs done -> merged.h5 (data; `merge <label>` for one MC sample)
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

On LPC the jobs are submitted from `SUBMIT_DIR` (default `~/nobackup/aoj_condor`; see `config.sh` and `submit.sh`), because LPC schedds cannot read `/eos` and hold every job submitted from there. Job logs then live in `$SUBMIT_DIR/logs`.

Individual steps are also available: `./run.sh pack|validate|salvage|jobs|submit|merge`.

## Retrying failed jobs
Failed jobs are **held** (kept in the queue), not dropped — `submit.sub` sets
`on_exit_hold = (ExitBySignal == true) || (ExitCode != 0)`, and `job.sh` runs
under `set -e`. Condor retries each failed job **once** by itself: `periodic_release`
releases a first `job.sh failed` hold 30 min after it happened. A job held again after that
retry, or held for any other reason (memory, `condor_hold`, submitted on hold), stays held.
So after a batch:
```bash
condor_q <cluster>                 # HELD = failed twice (or held on purpose); running/idle = still going
condor_release <cluster>.<proc>    # re-run one by hand -- no resubmit / store-listing
```
Inspect its `logs/out_<chunk>.h5.err` to see why before releasing. To stop the automatic
retry (e.g. while the store is down), `condor_qedit <cluster> PeriodicRelease False`.
`condor_release <cluster>` releases every held job of the cluster, including ones
submitted with `AOJ_SUBMIT_HOLD=1` that are still waiting their turn. (Evictions are *not*
held — Condor auto-reschedules those.)

To (re)submit only specific chunks **without** listing the store (handy off-site,
where the redirector directory `ls` is slow), put their labels (e.g. `2016G_sub014`,
one per line) in a file and run `AOJ_MISSING=missing.txt ./run.sh jobs`.

## Knobs (`config.sh`, or env overrides)
| var | default | meaning |
|---|---|---|
| `AOJ_SITE` | `uaf` | submit site: `uaf` \| `lpc` |
| `FILES_PER_JOB` | `3` | MINIAOD files per Condor job |
| `NTHREADS` | `1` | cmsRun threads (= `request_cpus`) |
| `REQUEST_MEMORY` | `4000` | MB |
| `REQUEST_DISK` | `20000000` | KB (~19 GB; holds the staged MINIAOD + NanoAOD) |
| `AOJ_STORE_XRD` | UCSD ceph | output store (xrootd) |
| `AOJ_TTBAR` | unset | `also` / `only`: run the ttbar tag-and-probe stage (above) |
| `AOJ_TTBAR_TARBALL` | `ttbar_semilep_v2.tar.gz` | its code tarball in `<store>/cmssw/` (v1 = no `skim` preset) |
| `AOJ_TTBAR_PRESET` | unset (`an`) | ttbar_maker preset of that stage; a non-default one (e.g. `skim`) writes `<store>/ttbar_<preset>/` and gets its own `SUBMIT_DIR` and log tag |
| `AOJ_SUBMIT_HOLD` | unset | `1`: submit the cluster held |
| `AOJ_SUBMIT_FORCE` | unset | `1`: let `submit.sh` replace a different `job.sh`/chunk in `SUBMIT_DIR` |

## Regenerate from scratch / other samples
`make_jobs.sh` reads `regen_lists.txt` (`<dataset> <file-list> <label>` per line).
`validate.sh` writes it automatically for the incomplete chunks. To process a
whole dataset (or MC) instead:
```bash
printf "2016G file_lists/2016G.txt 2016G\n2016H file_lists/2016H.txt 2016H\n" > regen_lists.txt
./run.sh jobs && ./run.sh submit
# or simply:  ./run.sh scratch
```
MC is auto-detected from the label (`QCD*`/`TT*`/`ST_*`/`WJets*`/`*_mc*`/`MC_*` → MC config, no JSON
cert; `TT*` also adds the per-jet `top_match` label). For MC, put one line per sample in `regen_lists.txt`, e.g.
`QCD_Pt300 file_lists/QCD_Pt300.txt QCD_Pt300`, and merge each sample separately:
`./run.sh merge QCD_Pt300` → `merged_QCD_Pt300.h5` (for ttbar, `python H5_select.py merged_TTToHadronic.h5
merged_TTToHadronic_topmatched.h5 --top_match 2` then keeps the gen-matched tops). Never merge different MC samples together;
each one needs its own `xsec × lumi / sumGenWeight` normalization. Merge a sample only after
all its jobs have finished. `merge.sh` will not overwrite an existing output (`H5_merge.py`
truncates it first) unless `AOJ_MERGE_FORCE=1`. `make_jobs.sh` skips an empty file list.

## Optional ttbar tag-and-probe stage (`AOJ_TTBAR`, MC only)
`job.sh` can also run the semileptonic-ttbar tag-and-probe (`ttbar_semilep/ttbar_maker.py`; see
`ttbar_semilep/README.md` and `ttbar_semilep/LPC_PROMPT.md`) on the same NanoAOD before it is deleted,
and push `<store>/ttbar/tt_<chunk>.h5`:
```bash
./pack_ttbar.sh                                    # once per code version -> <store>/cmssw/ttbar_semilep_vN.tar.gz, <store>/ttbar/
AOJ_TTBAR=also ./run.sh jobs && ./run.sh submit   # jet-level out_<chunk>.h5 + tt_<chunk>.h5
AOJ_TTBAR=only ./run.sh jobs && ./run.sh submit   # tt_<chunk>.h5 only (no H5_maker)
```
- `make_jobs.sh` writes the setting into `submit.sub` (the job environment). Unset (the default),
  `job.sh`, `submit.sub` and every output are unchanged.
- `only` holds a job whose Stage 3 fails; `make_jobs.sh` then skips chunks whose `tt_<chunk>.h5` is in
  `<store>/ttbar/`. `also` is for samples with no jet-level output yet: a Stage 3 failure there only warns
  (the jet-level `.h5` is already pushed), and `make_jobs.sh` skips chunks whose `out_<chunk>.h5` exists,
  listing those without a `tt_` file in `missing_tt.txt` to backfill with
  `AOJ_TTBAR=only AOJ_MISSING=missing_tt.txt ./run.sh jobs && ./run.sh submit`.
- `AOJ_TTBAR_PRESET=skim` (needs `AOJ_TTBAR`) runs `ttbar_maker.py --preset skim` and pushes to
  `<store>/ttbar_skim/` (`make_jobs.sh` then lists that directory), from `~/nobackup/aoj_condor_ttbar_skim`
  with logs `*.ttbar-<mode>-skim.*`. Unset, nothing changes.
- On LPC, `AOJ_TTBAR` clusters default to their own `SUBMIT_DIR` (`~/nobackup/aoj_condor_ttbar`), and
  their logs are `logs/out_<chunk>.h5.ttbar-<mode>.{out,err,log}`. `submit.sh` refuses to replace a
  different `job.sh` or chunk file in a `SUBMIT_DIR` (queued jobs read them when they start) unless
  `AOJ_SUBMIT_FORCE=1`. `AOJ_SUBMIT_HOLD=1` submits a cluster held, to start later with
  `condor_release -name <schedd> <cluster>`.
- Data labels exit at once: the data route (SingleMuon/SingleElectron) has not been set up.
- The code is `<store>/cmssw/$TTBAR_TARBALL` (`config.sh`), a small tarball kept apart from the CMSSW
  one. Each job fetches it right after the CMSSW tarball and unpacks it over the tarball's copy of
  `ttbar_semilep/`, so a missing or broken one holds the job before any staging; `gen_submit.sh` refuses
  to write `submit.sub` when it is not on the store. `pack_ttbar.sh` packs the committed
  `ttbar_semilep/` (`git archive HEAD`; it refuses while there are uncommitted changes), checks the
  uploaded size, and writes each name once: the store refuses renames over xrootd, so an in-place update
  could not be atomic. To ship new code, commit it, bump the version in `config.sh` (an
  `AOJ_TTBAR_TARBALL` override has to be set for `run.sh jobs` as well), and run `pack_ttbar.sh`.
  `ttbar_semilep_v1.tar.gz` is commit afd845f.
- It runs `ttbar_maker.py --sample_type MC --era 2016postVFP --trigger lepton_or_jetht --store one_probe`
  through `ttbar_semilep/py3env.sh` (LCG_105a centos7 view inside the EL7 image): about a minute and
  350 MB per job.
- Normalise with `sum_genweight` (the `Runs` tree), summed over the files actually processed. The
  CMSSW_10_6_30 NanoAOD output writes its trees only when it closes the file, so a `cmsRun` crash before
  that leaves no readable `Events` tree: Stage 2 or 3 fails and the job is held (`condor_release`).
- `AOJ_MAX_EVENTS=N` passes `maxEvents=N` to `cmsRun` (`gen_submit.sh` and `job.sh` print a warning).
  It is for pilots and tests only.

## Porting to Fermilab LPC

CMSSW_10_6_30 is slc7 but LPC nodes are el8/el9, so build/run inside the EL7
container. `~/nobackup` is a symlink to `/uscms_data`, which is NOT mounted in
the container by default — bind it on entry, or the symlink dangles:

```bash
cmssw-el7 -B /uscms_data             # enter el7 with nobackup bound
cd ~/nobackup                        # build here (the /uscms/home quota is tiny)
cmsrel CMSSW_10_6_30
cd CMSSW_10_6_30/src && cmsenv
git clone https://github.com/cms-opendata-analyses/PFNanoProducerTool.git PhysicsTools/PFNano
git clone https://github.com/cms-nanoAOD/nanoAOD-tools.git PhysicsTools/NanoAODTools
git clone git@github.com:zichunhao/AOJProcessing.git
scram b -j8
cd AOJProcessing/condor
```

Then **`exit` the container and run `run.sh` from the bare LPC host** — only the
build (`scram b`) needs el7; `condor_submit` lives on the host, and the
validate/salvage/merge steps enter the el7 container themselves (jobs run in el7
on the workers via `+SingularityImage`):
```bash
exit                                 # leave cmssw-el7 (condor_submit is host-side)
cd ~/nobackup/CMSSW_10_6_30/src/AOJProcessing/condor
export AOJ_SITE=lpc
voms-proxy-init -rfc -voms cms
./run.sh pack && ./run.sh smoke      # then: ./run.sh scratch
```
(If `pack` reports `xrdcp: command not found` on the host, first
`source /cvmfs/cms.cern.ch/cmsset_default.sh`.)

- **Output / `LPCUSER` (note: usernames differ across sites):** output defaults
  to the UCSD ceph even from LPC (one collection point). That path hardcodes the
  UCSD name (`zichun`) and works from LPC because writes are authorized by your
  grid proxy, not your login — so you do **not** need `LPCUSER` for the default.
  `LPCUSER` lives in `config.sh`'s `lpc)` block (defaults to your login `$USER`)
  and is only used if you redirect output to FNAL EOS — uncomment the
  `STORE_XRD`/`STORE_LOCAL` lines there. Verify your EOS name first
  (`eos root://cmseos.fnal.gov ls /store/user/<name>`): a CERN login may map to a
  suffixed EOS area (e.g. `zhao` vs `zhao1`).
- `validate`/`salvage` are UAF-only (they act on the local `run_*/` partials).
- `merge` runs where the store is mounted (UAF for ceph; LPC for EOS).
- If LPC condor complains about the OS, add `+DesiredOS = "EL7"` to
  `submit.templ.sub` (usually unnecessary with `+SingularityImage`).
