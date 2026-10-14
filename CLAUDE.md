# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aspen Open Jets (AOJ) Processing — turns 2016 CMS Open Data (JetHT) into ML-ready HDF5 jet samples. Two physics stages plus a batch-orchestration layer:

1. **Stage 1 — PFNano production** (`cmsRun`): MINIAOD → NanoAOD ROOT, adding AK8-jet Particle Flow constituents.
2. **Stage 2 — HDF5 creation** (`H5_maker.py`): NanoAOD → HDF5, applying trigger / quality / jet selections.
3. **Batch orchestration** — two drivers:
   - **`condor/` (current, preferred)**: HTCondor pipeline, one job per small chunk doing PFNano + H5 end-to-end on the worker. See `condor/README.md`.
   - **`step1.sh` → `step2.sh` → `step3.sh` (legacy)**: local parallel run under a shell/tmux; it lost everything when the session died, which is why `condor/` replaced it. Its `run_*/` and `split_*/` leftovers are still consumed by `condor/validate.sh` + `salvage.sh`.

This repo (`AOJProcessing`) is the analysis code, git-tracked at `CMSSW_10_6_30/src/AOJProcessing` inside a CMSSW release area. The external CMSSW packages it depends on (`PhysicsTools/PFNano`, `PhysicsTools/NanoAODTools`) are cloned siblings under `src/` and are not part of this repo.

## Environment & Build

Everything runs inside the CMSSW_10_6_30 CMS Open Data environment (Docker/VM). `H5_maker.py`/`Utils.py` are **Python 2** (use `xrange`, run under CMSSW's python) and import the NanoAODTools data model directly.

```bash
cd CMSSW_10_6_30/src/
cmsenv      # set up CMSSW environment — required before any command below
scram b     # build the release + PFNano C++ plugin
```

CMSSW_10_6_30 is `slc7_amd64_gcc700`. On el8/el9 hosts (FNAL LPC, UAF) build and run interactively inside the EL7 container: `cmssw-el7 -B /uscms_data` (LPC — the bind is needed or `~/nobackup` dangles). `condor_submit` exists only on the bare host, so `condor/run.sh` runs from **outside** the container; its local helpers (validate/salvage/merge) enter `cmssw-el7` by themselves.

There is no test suite. To check a change, run Stage 1 on one input file with `maxEvents=N`, then run Stage 2 on the output. For the batch path, `condor/run.sh smoke` runs one file end to end.

The release was built elsewhere and moved to EOS. If `cmsRun` fails with `No module named PFNano...` or `CMSSW_BASE` points at an old path, run `scram b ProjectRename` in `src/` (worker jobs already do this). The container cannot see the Claude session scratch directory; put local test output under `/uscms_data` or EOS instead.

## HTCondor pipeline (`condor/`)

```bash
cd condor
./run.sh pack                 # tar the built release (+ this repo, cert) -> $STORE_XRD/cmssw/  (re-run after ANY code change)
./run.sh smoke [2016G]        # one-file end-to-end test -> h5/out_smoke_sub000.h5
./run.sh scratch              # pack + regenerate all of 2016G+2016H
./run.sh salvage-and-submit   # UAF: reuse complete legacy run_*/ NanoAOD, submit only bad/missing chunks
./run.sh jobs && ./run.sh submit   # rebuild joblist from regen_lists.txt, then submit
condor_release <cluster>      # retry HELD (failed) jobs
./run.sh merge                # where the store is mounted (UAF): data -> merged.h5
./run.sh merge QCD_Pt300      # one MC sample -> merged_QCD_Pt300.h5 (never merge across MC samples)
```
`merge.sh` refuses to overwrite an existing output (`H5_merge.py` truncates its output first, and `merged.h5` is the source of the published shards) unless `AOJ_MERGE_FORCE=1`. Merge a sample only after all its jobs have finished.

On LPC, **jobs cannot be submitted from `/eos`**: the schedds don't see the EOS FUSE mount, so every job goes HELD with `Cannot access initial working directory`. All submissions go through `submit.sh`. On LPC it copies `job.sh`, `submit.sub`, `joblist.txt` and the needed `chunks/` to `SUBMIT_DIR` (default `~/nobackup/aoj_condor`, set in `config.sh`) and submits from there, so **job logs live in `$SUBMIT_DIR/logs`**, not `condor/logs`. On UAF, `SUBMIT_DIR` is empty and jobs are submitted in place. `condor_submit` routes the cluster to one of several schedds (it prints `Attempting to submit jobs to lpcscheddN.fnal.gov`). Plain `condor_q` may show nothing, so query that schedd: `condor_q -name lpcscheddN.fnal.gov <user>`. Stdout comes back only when a job exits; use `condor_tail -name <schedd> <cluster.proc>` to peek at a running job.

How it fits together:
- **`config.sh` holds all site configuration.** It picks `SITE` (`uaf`|`lpc`, auto-detected from the hostname, `AOJ_SITE` overrides) and exports the store path, container image, and job sizing (`FILES_PER_JOB`, `NTHREADS`, `REQUEST_*`). Every other script sources it. Output defaults to UCSD ceph (`root://redirector.t2.ucsd.edu:1095//store/user/zichun/parcel/AOJ`), even for jobs submitted from LPC.
- **`make_jobs.sh`** reads `regen_lists.txt` (`<dataset> <file-list> <label>`), splits each list into `chunks/<label>_subNNN` with `FILES_PER_JOB` files each (the suffix widens to 4 digits only for labels with more than 1000 chunks; an empty list is skipped with a warning), and writes `joblist.txt`. It skips any chunk whose `out_<chunk>.h5` is already on the store (it lists the store through the local mount, or with `xrdfs` under a timeout), so re-running it is idempotent. `AOJ_MISSING=<file>` skips the store listing and queues only the chunk labels in that file, which helps off-site where listing is slow.
- **`gen_submit.sh`** fills the `@PLACEHOLDERS@` in `submit.templ.sub` → `submit.sub` (generated, untracked). Jobs run in the EL7 Singularity image. `on_exit_hold` keeps failed jobs in the queue as HELD.
- **`job.sh`** (worker, `set -e`) does: xrdcp + untar the release tarball → `scram b ProjectRename` → stage MINIAOD inputs to scratch with retries (WAN streaming is ~10× slower) → `cmsRun` → `H5_maker.py` → xrdcp only the `.h5` to `$STORE_XRD/h5/`. The large NanoAOD never leaves the worker. `job.sh` is the Condor executable, shipped from the submit dir and not from the tarball, so edits to it need no re-pack. cmsRun output passes through `thin()`, which keeps every 10000th `Begin processing` line and drops the MC Rivet HTXS warning; without it each job writes about 20 MB of logs. If `cmsRun` crashes (a known rare ParticleNet/DeepBoostedJet segfault) after writing partial output, the job keeps going on that partial output. Data vs MC comes from the dataset label (`QCD*`/`TT*`/`*_mc*`/`MC_*` → MC config, no cert). `TT*` labels also pass `--top_label` to `H5_maker.py`.
- Workers run the **packed tarball**, not your working tree. Any edit to `H5_maker.py`, the configs, or the plugin takes effect on the grid only after `./run.sh pack`. `pack_cmssw.sh` fails early if the golden-JSON cert is missing.

## Legacy local pipeline (`step*.sh`)

Run from the repo dir, in order. Together they process 2016G+2016H end to end (**data only** — `step1.sh` hardcodes the data config; there is no MC/QCD batch script despite the QCD file lists existing):

```bash
./step1.sh   # Stage 1: 16 parallel cmsRun jobs (8 per dataset) -> run_<ds>_NN/nano_data2016.root
./step2.sh   # Stage 2: parallel H5_maker over every run dir    -> out_run_<ds>_NN.h5
./step3.sh   # merge:   H5_merge.py merged.h5 out_run_*.h5       -> merged.h5
```

- `step1.sh`: `NJOBS=8` per dataset, `NTHREADS=8` per job (up to 16×8 = 128 cores). Splits each `file_lists/<ds>.txt` into 8 `split_<ds>/chunk_NN`, then launches one `cmsRun pfnano_data_2016UL_OpenData.py inputFiles_load=<chunk> nThreads=8` per chunk inside its own `run_<ds>_NN/` dir (each writes a fixed `nano_data2016.root`, so per-dir isolation avoids collisions). Waits on all PIDs, exits nonzero if any job fails.
- `step2.sh`: runs `H5_maker.py` on each `run_*/nano_data2016.root` with the run-quality cert, producing `out_run_<ds>_NN.h5`.
- `step3.sh`: `python H5_merge.py merged.h5 out_run_2016G_*.h5 out_run_2016H_*.h5`.

## Manual / single-file commands

### Stage 1 — PFNano (MINIAOD → NanoAOD ROOT)
```bash
# Data: writes nano_data2016.root   (GlobalTag 106X_dataRun2_v37)
cmsRun pfnano_data_2016UL_OpenData.py inputFiles_load=file_lists/2016G.txt nThreads=8

# MC: writes nano_mc2016post.root   (GlobalTag 106X_mcRun2_asymptotic_v17)
cmsRun pfnano_mc_2016UL_OpenData.py inputFiles_load=file_lists/QCD_Pt300.txt nThreads=8

# Single file (testing). inputFiles= takes ROOT paths directly; inputFiles_load= reads a list file.
cmsRun pfnano_data_2016UL_OpenData.py inputFiles=root://eospublic.cern.ch//eos/opendata/cms/...
```
`nThreads` (VarParsing option, default 4) sets `numberOfThreads`; `maxEvents=N` limits events (default -1 = all). The MC output is `nano_mc2016post.root` (UL16 post-VFP), not `nano_mc2016.root`.

### Stage 2 — HDF5 (NanoAOD → HDF5)
```bash
# Data: -j JSON cert required for run-quality filtering
python H5_maker.py -i nano_data2016.root -o out.h5 --sample_type data -j Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt

# MC: no cert. --top_label (ttbar) records per-jet top_match in gen_info without dropping jets;
# --gen_match P_ID instead KEEPS ONLY gen-matched jets
python H5_maker.py -i nano_mc2016post.root -o out.h5 --sample_type MC --top_label
```
Flags: `-i/--input`, `-o/--output` (default `test.h5`), `--sample_type {data,MC}` (exact, else exits), `-j/--json`, `-y/--year` (2016), `-n/--nEvents` (default -1; **counts saved jets, not events**), `--gen_match` (default 0; only active when >0), `--top_label`.

### Merge
```bash
python H5_merge.py merged.h5 file1.h5 file2.h5 ...   # argv[1]=output, argv[2:]=inputs
```

## Architecture

### Data flow
`MINIAOD` → `pfnano_*_OpenData.py` (cmsRun) → `nano_{data2016,mc2016post}.root` → `H5_maker.py` → per-job `.h5` → `H5_merge.py` → `merged.h5`

### `pfnano_*_OpenData.py` (cmsDriver-generated CMSSW configs)
Process `cms.Process('NANO', Run2_2016, run2_nanoAOD_106Xv2)`. Apply stock `nanoAOD_customize{Data,MC}` then PFNano's `PFnano_customize{Data,MC}_AK8JetsOnly` (`addPFCands(..., onlyAK8=True)`) — only AK8-jet PF candidates are stored, not all event PF. Output datatier is NANOAODSIM for both; `numberOfStreams=0`, `numberOfConcurrentLuminosityBlocks=1`, ROOT IMT disabled.

### `H5_maker.py`
- **`NanoReader`** — event loop reading NanoAOD via the `PhysicsTools.NanoAODTools` data model. Selections, in order:
  - **JSON run filter** (data, when `-j` given): applied before the loop, through the per-file entry list that `preSkim` builds.
  - **Trigger**: OR of **85** active HLT paths (AK8PFJet / PFHT / DiPFJetAve / PFJet / CaloJet500; 2 VBF paths commented out).
  - Trigger/filter names that have no branch in the input NanoAOD are dropped before the loop (`avail_triggers`/`avail_filters`), because `readBranch` raises on a missing branch. The HLT menu differs between eras, so this is expected and does not change the selection. Keep this guard when you add triggers.
  - **Quality**: AND of 8 flags — `Flag_goodVertices`, `globalSuperTightHalo2016Filter`, `HBHENoiseFilter`, `HBHENoiseIsoFilter`, `EcalDeadCellTriggerPrimitiveFilter`, `BadPFMuonFilter`, `eeBadScFilter`, `CSCTightHaloFilter`.
  - **Jet** (per AK8 FatJet): `pt > 300`, `jetId == 6` (tight + tightLepVeto), `|eta| < 2.5`.
  - **Gen-match** (MC, when `--gen_match > 0`): `6` → ttbar path (`get_ttbar_gen_parts`, requires `check_matching == 2` for either top); any other ID (24=W, 23=Z, 25=H) → V+jets path (`get_vjets_gen_parts`, `check_matching == 1`). Only matched jets are saved.
- **`Outputer`** — buffers `batch_size=5000` jets, then flushes to HDF5. Four datasets, plus `gen_info` for MC:

  | dataset | shape | dtype | gzip | contents |
  |---|---|---|---|---|
  | `event_info` | (N, 3) | int64 | no | run, luminosityBlock, event |
  | `jet_kinematics` | (N, 4) | float32 | yes | pt, eta, phi, msoftdrop |
  | `jet_tagging` | (N, 13) | float32 | yes | nConstituents, tau1–4, ParticleNet {H4q,Hbb,Hcc,QCD,T,W,Z}vsQCD, PN mass |
  | `PFCands` | (N, 150, 11) | float32 | yes | per constituent: Px,Py,Pz,E,d0,d0Err,dz,dzErr,charge,pdgId,puppiWeight |
  | `gen_info` (MC only) | (N, 5) | float32 | yes | genWeight, top_match (0/1/2 = none/W/top; −1 = not computed), FatJet hadronFlavour, nBHadrons, nCHadrons. Column names are in the `columns` attribute, a numpy array of byte strings |

  Root attributes give the normalization. `nEventsProcessed` counts the events the loop reads (MC: all of them; data: those left by the `-j` JSON pre-skim), and MC files also get `sumGenWeight`; both are taken before the trigger, filter and jet cuts. Normalize each MC sample with `xsec × lumi / sumGenWeight`. These come from the event loop, not the NanoAOD `Runs` tree, so they stay correct for the partial NanoAOD a segfaulted job leaves behind.

  Note `event_info` is **not** gzip-compressed; the others are. PF candidates: up to 150 per jet (truncated to the first 150 **before** sorting), then sorted by descending pT; Px/Py/Pz/E are built from a `PtEtaPhiMVector`.

### `Utils.py` (MC gen-matching)
`get_ttbar_gen_parts` / `get_vjets_gen_parts` find final-copy tops/Ws and their decay fermions/b-quarks via status flags + mother links. `check_matching(jet, f1, f2, b)` returns 0/1/2 (no / W / top) using ΔR<0.8 cone matching (`ang_dist` handles φ wrap). PDG IDs: top=6, H=25, W=24, Z=23, b=5.

### `H5_merge.py`
Copies the first input as the base, including dataset and root attributes, then concatenates the rest dataset by dataset. It skips any input whose key set differs from the base, so MC (which has `gen_info`) never merges into data. The root attributes `nEventsProcessed` and `sumGenWeight` are **summed**. Datasets named `*_eff` or with first-dim 1 get a row-count-weighted average instead of concatenation. `H5_maker.py` writes no such datasets, but beware: a per-job file holding exactly one jet would trip this rule.

### `file_lists/`
Input ROOT-file lists (one eospublic xrootd path per line):
- **Data:** `2016G.txt` (1236), `2016H.txt` (1309).
- **MC:** UL16 post-VFP `RunIISummer20UL16MiniAODv2`, `106X_mcRun2_asymptotic_v17-v1`. The lists contain only files actually present on eospublic.
  - `QCD_Pt{300,470,600,800,1000,1400,1800,2400,3200}.txt` cover the pT-hat bins 300to470 … 3200toInf. `QCD_Pt2400` has 1 file because only 1 of the official 76 is on disk.
  - `TTToHadronic.txt` (2434) and `TTToSemiLeptonic.txt` (3192).

Each sample directory has an official `file-indexes/*.txt` next to its `MINIAODSIM` folder, and the Open Data API (`opendata.cern.ch/api/records/?q=title:"<dataset>"`) gives event counts. `step1.sh` and `run.sh scratch` cover only 2016G/2016H. To run MC through Condor, put lines such as `QCD_Pt300 file_lists/QCD_Pt300.txt QCD_Pt300` in `regen_lists.txt`.

## Notes
- `.gitignore` excludes `*.h5 *.txt *.root *.swp *.un~`, but the golden-JSON cert and `file_lists/*.txt` are **force-tracked**: workers need them inside the packed tarball. Use `git add -f` for any new `.txt` that has to ship. `condor/.gitignore` excludes the generated `chunks/`, `logs/`, `submit.sub`, `joblist.txt`, and `*_lists.txt`.
- `run_*/` and `split_*/` hold legacy `step1.sh` output, not source. `validate.sh` assumes `run_<ds>_NN` came from `split_<ds>/chunk_NN`.
- `step1.sh` relies on the `nThreads` VarParsing option in both `pfnano_*` configs.
