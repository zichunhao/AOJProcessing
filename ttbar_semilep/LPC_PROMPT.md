# Brief for the LPC session: MC for top jets (training + semileptonic-ttbar evaluation)

> **Update 2026-10-26 (UAF), read first.** (1) **Only public CMS Open Data**: every file list is on eospublic
> (the CMS-grid `/store` lists of single-top tW, W+jets and MuEnriched QCD are converted); request the "on demand"
> files on the record pages first (`python3 ttbar_semilep/samples.py --status`). (2) The CR samples are
> `ttbar_semilep/samples.py` (adds W/Z(qq)+jets, DY, VV, ttZ, QCD pT-hat 120-300; the e-channel QCD is the inclusive
> pT-hat QCD, the HT-binned QCD is retired). (3) Rerun every CR sample, MC and data, with the format-3 code
> (`ttbar_semilep_v3.tar.gz`, the pre-filter on) into `<store>/ttbar_skim_v3/`; the recipe is in
> `ttbar_semilep/README.md` ("CR samples, outputs and production"). `ttbar_skim/` stays as the format-2 reference.

You are processing CMS Open Data MC for the Aspen Open Jets (AOJ) project (repo `AOJProcessing`, see its
`CLAUDE.md`). A UAF session did the R&D for a **top-enriched AOJ jet sample from real 2016G+H data, selected with
semileptonic-ttbar tag-and-probe**. This brief hands over what was decided and what MC is needed. Read it fully
before starting; ask the user before anything marked *(ask first)*.

## 0. Get the code

The R&D code is `AOJProcessing/ttbar_semilep/` (selection, maker, summary, tests, `README.md` with all numbers)
on branch **`ttbar-semilep`** of `git@github.com:zichunhao/AOJProcessing.git`:

```bash
git fetch origin && git checkout ttbar-semilep      # or: git checkout origin/ttbar-semilep -- ttbar_semilep
```

`ttbar_semilep/` is self-contained: it does not need `event_level/`. `aoj_common.py` is a copy of the AOJ
trigger/filter lists, checked against `H5_maker.py` by a test, and `py3env.sh` is its own Python 3 environment.

Then run the tests (Python 3 through `ttbar_semilep/py3env.sh`: LCG_105a on el7/el9; on an el8 host it re-runs
itself inside `cmssw-el9`):

```bash
cd CMSSW_10_6_30/src/AOJProcessing
bash ttbar_semilep/py3env.sh ttbar_semilep/tests/test_ttbar.py     # expect: N passed, 0 failed (real-file tests may SKIP)
```

Set `AOJ_TT_TEST_MC` (a NanoAOD/PFNano MC file with GenPart) and `AOJ_TT_TEST_MC_ERA` (`2016postVFP`) so the
real-file tests (vectorised selection == independent per-event loop, truth sanity) run on an LPC file.

## 1. What was decided (do not change without asking)

**Selection** (`ttbar_selection.py`, default `--preset an`): the HIG-24-010 ttbar tag-and-probe implementation,
adapted to 2016 UL and to AOJ. The probe is always an AOJ jet (pT > 300, |eta| < 2.5, jetId == 6, as `H5_maker.py`).

| stage | requirement |
|---|---|
| trigger | 2016: mu `HLT_IsoMu24 \|\| HLT_IsoTkMu24`; e `HLT_Ele27_WPTight_Gsf \|\| HLT_Ele32_WPTight_Gsf` (Ele32_WPTight does not exist in Run2016G) |
| filters_json | the 8 AOJ MET filters; golden JSON for data |
| one_lepton | exactly one tag lepton pT > 55, \|eta\| < 2.4 (mu tightId + pfRelIso04 < 0.15; e mvaFall17V2Iso_WP80, ECAL gap excluded); no other isolated lepton with pT > 30 |
| met | MET > 50 |
| one_probe | exactly one AK8 jet with pT > 300, \|eta\| < 2.5, and it is an AOJ jet. (The analysis note vetoes extra AK8 > 160; with our 300 GeV probe we veto > 300: ~3x the events at the same top purity. `--preset an160` = the note's 160.) |
| dphi | dphi(tag lepton, probe) > 2.0 |
| btag | >= 1 AK4 jet pT > 40, \|eta\| < 2.4, DeepJet medium (UL16 post-VFP 0.2489), dR(b, probe) > 1.5, dR(b, lepton) > 0.5 |
| tau32 | probe tau32 < 0.5 (the final sample uses it) |

**Truth definition** ("fully merged top"): the b, q and q' of a hadronic top all within dR < 0.8 of the jet axis.
This is the JetClass signal definition, `H5_maker.py --gen_match 6` (`Utils.check_matching == 2`), and category
`t` (= 3) of `ttbar_maker.py`'s `truth` table (other categories: `W` = q q' only, `bq`, `other`).

**Scale factors available** (POG correctionlib, `/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/`,
2016postVFP_UL): muon trigger `NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight` (binned |eta| x pT,
~0.95-0.99 above 55 GeV), muon ID/iso; electron ID (`wp80iso`) and reco. **No electron trigger SF** exists.

## 2. Reference numbers from the R&D (use them to validate your outputs)

Small UL16 NanoAODv9 subsets, normalised to 2016G+H = 16.39 fb^-1, default selection, mu + e, after tau32:

| probe pT | data events (projected) | fully merged t | merged W | b+q | other top | W+jets | QCD |
|---|---|---|---|---|---|---|---|
| 300-400 | 775 | 0.41 | 0.17 | 0.14 | 0.12 | 0.01 | 0.15 +- 0.15 (one MC event) |
| 400-500 | 599 | 0.83 | 0.06 | 0.06 | 0.05 | 0.01 | 0 |
| > 500 | 207 | 0.83 | 0.04 | 0.05 | 0.08 | 0.01 | 0 |

- TTToSemiLeptonic: the full selection keeps 1.6e-4 (mu) + 1.2e-4 (e) of all events.
- ttbar (+ single top) is ~96 % (~99 %) of the selected events, not counting the one high-weight QCD MC event.
- Data/MC is ~0.95 at the btag stage and ~0.8 after tau32 (low data statistics), with no SFs and no top-pT
  reweighting.
- **QCD is MC-statistics limited**: 0-1 raw QCD MC events survive tau32. The full QCD samples are the main thing
  the R&D could not do.

## 3. Task A: training MC for fine-tuning (fully merged tops vs QCD)

Decision: the signal label is **fully merged tops only**. Partially merged ttbar jets are **not** background; drop
them, or give merged W their own class if the model is multi-class. Train without the lepton tag and without
tau32/mSD cuts (those cuts are applied identically to data and MC at evaluation time), and reweight pT/eta
between classes when training.

- Signal: `TTToSemiLeptonic` (+ `TTToHadronic` for more statistics), UL16 post-VFP (Open Data records 67992,
  67840; MiniAODv2). Run the normal jet-level chain (Stage 1 MC config + `H5_maker.py --sample_type MC`) with
  **`--gen_match 6`**: it writes exactly the fully merged AOJ jets, in the usual AOJ format (incl. `PFCands`).
  `condor/job.sh` does not pass `--gen_match` today: add it for the ttbar samples (e.g. per dataset), keeping
  jet mode byte-identical for everything else.
- *(ask first)* If the user wants merged-W / partial classes too: instead of `--gen_match`, write all AOJ jets
  with their truth category (`ttbar_selection.gen_truth` gives it; it needs a small "all AOJ jets, no tag
  selection" writer). Ask before building it.
- Background: QCD AOJ jets, `H5_maker.py --sample_type MC` without gen matching (QCD file lists exist in
  `file_lists/QCD_Pt*.txt`; `event_level/opendata_filelist.py`, where available, builds MC lists from Open Data records).

## 4. Task B: MC for the tag-and-probe evaluation and purity

Run `ttbar_semilep/ttbar_maker.py` on the **Stage-1 PFNano output of each MC job, in the same Condor job, before
the NanoAOD is deleted** (`job.sh` removes it after `H5_maker`). Push the small output next to the jet-level one,
e.g. `<store>/ttbar/tt_<chunk>.h5`. Run it on the worker with `ttbar_semilep/py3env.sh`, which uses the LCG_105a
centos7 view inside the EL7 job image (`condor/job_event.sh`, if you have it, does the same for the event-level tools).

```bash
bash ttbar_semilep/py3env.sh ttbar_semilep/ttbar_maker.py -i nano_mc2016post.root -o tt_<chunk>.h5 \
     --sample_type MC --era 2016postVFP --trigger lepton_or_jetht --store one_probe
```

`--store one_probe` keeps every event up to the probe stage, and `ttbar_tag/stage` records how far it got.
`--trigger lepton_or_jetht` keeps the lepton-trigger route and the JetHT route; the bits `trig_mu`/`trig_el`/
`trig_jetht` let you separate them offline. The output has the probe in AOJ jet-level format (`event_info`,
`jet_kinematics`, `jet_tagging`, `PFCands`), `ttbar_tag`, `truth`, `gen_weight`, a per-channel `cutflow` and the
attribute `sum_genweight` (Runs `genEventSumw`; use it for normalisation).

Samples (UL16 post-VFP, RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17). Cross sections are the ones the
R&D used (13 TeV). The ttbar and single-top values are the standard CMS numbers (sigma_tt = 831.76 pb) but were
quoted from memory, not re-checked; check them, together with the QCD ones, in XSDB or your own tables:

| group | samples (Open Data record) | sigma [pb] |
|---|---|---|
| ttbar | TTToSemiLeptonic (67992), TTTo2L2Nu (67800), TTToHadronic (67840) | 365.34, 88.29, 377.96 |
| single top | ST_tW_top / antitop_5f_NoFullyHadronicDecays (64894 / 64838); ST_t-channel_top / antitop_4f_InclusiveDecays (64758 / 64658); ST_s-channel_4f_leptonDecays (64634) | 19.47 each; 136.02 / 80.95; 3.36 |
| W+jets | WJetsToLNu_HT-{70To100,100To200,200To400,400To600,600To800,800To1200,1200To2500,2500ToInf} (69720-69734) | 1275, 1270, 336.8, 44.64, 11.6, 4.898, 1.142, 0.02678, each x 1.21 (NNLO k-factor; from HH4b `xsecs.json`) |
| QCD (mu) | QCD_Pt-{120To170,...,1000}_MuEnrichedPt5 (Open Data records around 63145-63237; look up each bin) | Run-2 XSDB sigma x filter eff. **Verify**: the R&D quoted 170-300 8654.49, 300-470 797.35, 470-600 79.03, 600-800 25.10, 800-1000 4.707, 1000+ 1.621 from memory |
| QCD (e) | QCD_Pt EMEnriched + QCD_Pt bcToE (records in the same range; look up), or QCD_HT{300..Inf} madgraphMLM | HT: 347700, 30330, 6412, 1118, 108.5, 21.94 (HH4b `xsecs.json`); EM/bcToE: look up |

**Priority:** ttbar and single top first (they set the purity), then W+jets, then the **full** QCD samples (the
open question). DY and diboson are expected below 1 %; skip them unless cheap.

Then summarise with the R&D tools:
- `ttbar_semilep/ttbar_summary.py`: yields and truth composition per pT bin.
- `ttbar_semilep/rnd/purity.py`, `compare.py`, `breakdown.py`: cutflow, tt/(tt+QCD), top purity. They read a
  `samples.json` built by `rnd/make_jobs.py` (DAS NanoAODv9 subsets), so adapt the paths or the loader to your
  outputs. Don't trust them blindly.

Compare against section 2 and report differences.

## 5. Data (*ask first*; the user scoped this session to MC)

When asked:
- **Lepton route:** SingleMuon (30513, 30546) and SingleElectron (30512, 30545) 2016G/H MINIAOD, ~18.6 TB and
  ~4,950 files. Stage-1 data config + `ttbar_maker.py --sample_type data -j Cert_...json --trigger lepton`.
  Take mu from SingleMuon and e from SingleElectron; the channels are exclusive by offline lepton content.
- **JetHT route:** ~2/3 of the final probes (91 % at pT > 400) also fire the AOJ JetHT OR, i.e. they are already
  AOJ jets. They can be picked out of existing AOJ by running `ttbar_maker.py --trigger jetht` on the Open Data
  JetHT NanoAODv9 (30525, 30558) and matching run/lumi/event + dR to `merged.h5`.

## 6. Pitfalls found during the R&D

- `HLT_Ele32_WPTight_Gsf` is absent in Run2016G: never use it alone for 2016.
- LCG_105a has no el8 build: `ttbar_semilep/py3env.sh` re-runs itself in `cmssw-el9` on el8 hosts (set
  `AOJ_PY3ENV_NO_CONTAINER=1` to disable). Inside the EL7 job image it uses the centos7 view.
- Remote reads from eospublic time out with uproot's default 30 s: `ttbar_maker.py --timeout` (default 600 s).
  Staging is how `job.sh` already handles MINIAOD.
- MC normalisation = `gen_weight * xsec * lumi / sum(sum_genweight of the processed files)`. Use only the files
  actually processed.
- Keep `--store one_probe` (not `btag`/`tau32`) so stage-by-stage purity can be redone offline.

## 7. Report back to the user

- Which samples ran: events, files, `sum_genweight`, store paths. Failures and what you did about them.
- The default-selection composition table (section 2 format) with your full MC, and the QCD level with its raw
  MC count.
- For Task A: number of fully merged top jets and QCD jets written per pT bin.
- Anything you had to change in `job.sh` / `config.sh` / `ttbar_semilep/`, as a short diff summary. Jet mode must
  stay byte-identical for samples that do not use the new options.
