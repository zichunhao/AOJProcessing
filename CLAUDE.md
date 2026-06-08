# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aspen Open Jets (AOJ) Processing — turns 2016 CMS Open Data (JetHT) into ML-ready HDF5 jet samples. Two physics stages plus a batch-orchestration layer:

1. **Stage 1 — PFNano production** (`cmsRun`): MINIAOD → NanoAOD ROOT, adding AK8-jet Particle Flow constituents.
2. **Stage 2 — HDF5 creation** (`H5_maker.py`): NanoAOD → HDF5, applying trigger / quality / jet selections.
3. **Batch driver** (`step1.sh` → `step2.sh` → `step3.sh`): runs the full thing in parallel over the 2016G + 2016H datasets and merges the result.

This repo (`AOJProcessing`) is the analysis code, git-tracked at `CMSSW_10_6_30/src/AOJProcessing` inside a CMSSW release area. The external CMSSW packages it depends on (`PhysicsTools/PFNano`, `PhysicsTools/NanoAODTools`) are cloned siblings under `src/` and are not part of this repo.

## Environment & Build

Everything runs inside the CMSSW_10_6_30 CMS Open Data environment (Docker/VM). `H5_maker.py`/`Utils.py` are **Python 2** (use `xrange`, run under CMSSW's python) and import the NanoAODTools data model directly.

```bash
cd CMSSW_10_6_30/src/
cmsenv      # set up CMSSW environment — required before any command below
scram b     # build the release + PFNano C++ plugin
```

## Full pipeline (batch driver)

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

# MC: no cert; optional --gen_match P_ID requires the jet to be gen-matched
python H5_maker.py -i nano_mc2016post.root -o out.h5 --sample_type MC --gen_match 6
```
Flags: `-i/--input`, `-o/--output` (default `test.h5`), `--sample_type {data,MC}` (exact, else exits), `-j/--json`, `-y/--year` (2016), `-n/--nEvents` (default -1; **counts saved jets, not events**), `--gen_match` (default 0; only active when >0).

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
  - **Trigger**: OR of **85** active HLT paths (AK8PFJet / PFHT / DiPFJetAve / PFJet / CaloJet500; 2 VBF paths commented out).
  - **Quality**: AND of 8 flags — `Flag_goodVertices`, `globalSuperTightHalo2016Filter`, `HBHENoiseFilter`, `HBHENoiseIsoFilter`, `EcalDeadCellTriggerPrimitiveFilter`, `BadPFMuonFilter`, `eeBadScFilter`, `CSCTightHaloFilter`.
  - **Jet** (per AK8 FatJet): `pt > 300`, `jetId == 6` (tight + tightLepVeto), `|eta| < 2.5`.
  - **JSON run filter** (data, when `-j` given): `preSkim` builds the per-file entry list.
  - **Gen-match** (MC, when `--gen_match > 0`): `6` → ttbar path (`get_ttbar_gen_parts`, requires `check_matching == 2` for either top); any other ID (24=W, 23=Z, 25=H) → V+jets path (`get_vjets_gen_parts`, `check_matching == 1`). Only matched jets are saved.
- **`Outputer`** — buffers `batch_size=5000` jets, then flushes to HDF5. Four datasets:

  | dataset | shape | dtype | gzip | contents |
  |---|---|---|---|---|
  | `event_info` | (N, 3) | int64 | no | run, luminosityBlock, event |
  | `jet_kinematics` | (N, 4) | float32 | yes | pt, eta, phi, msoftdrop |
  | `jet_tagging` | (N, 13) | float32 | yes | nConstituents, tau1–4, ParticleNet {H4q,Hbb,Hcc,QCD,T,W,Z}vsQCD, PN mass |
  | `PFCands` | (N, 150, 11) | float32 | yes | per constituent: Px,Py,Pz,E,d0,d0Err,dz,dzErr,charge,pdgId,puppiWeight |

  Note `event_info` is **not** gzip-compressed; the other three are. PF candidates: up to 150 per jet (truncated to the first 150 **before** sorting), then sorted by descending pT; Px/Py/Pz/E are built from a `PtEtaPhiMVector`.

### `Utils.py` (MC gen-matching)
`get_ttbar_gen_parts` / `get_vjets_gen_parts` find final-copy tops/Ws and their decay fermions/b-quarks via status flags + mother links. `check_matching(jet, f1, f2, b)` returns 0/1/2 (no / W / top) using ΔR<0.8 cone matching (`ang_dist` handles φ wrap). PDG IDs: top=6, H=25, W=24, Z=23, b=5.

### `H5_merge.py`
Copies the first input as the base, then concatenates the rest dataset-by-dataset (skips any input whose key set differs from the base). Datasets named `*_eff` or with first-dim 1 get a row-count-weighted average instead of concatenation — but the current `H5_maker.py` writes no such datasets, so in practice the merge is a plain concatenation.

### `file_lists/`
Input ROOT-file lists (one path per line): `2016G.txt` (1236), `2016H.txt` (1309), `QCD_Pt300.txt` (1087), `QCD_Pt470.txt` (1067), `QCD_Pt600.txt` (1431). The batch scripts use only 2016G/2016H.

## Notes
- `.gitignore` excludes `*.h5 *.txt *.root *.swp *.un~` — so file lists, the cert `.txt`, all ROOT/HDF5 outputs, and `split_*/chunk_*` are untracked. The generated `run_*/` and `split_*/` dirs are working data, not source.
- The unstaged diff on the two `pfnano_*` configs adds the `nThreads` VarParsing option (replacing a hardcoded `4`); `step1.sh` depends on it (`nThreads=8`).
