# Top tagging with AOJ: plan (pretrain → fine-tune → evaluate)

Status 2026-10-25. Models: RINO and baselines, all trained on the same sets.

| stage | data | status |
|---|---|---|
| 0. Pretraining | AOJ real data (JetHT 2016G+H), no labels | done |
| 1. Fine-tuning | CMS UL16 MC: fully merged tops (TTToHadronic) vs QCD | sets built and checked (`finetune_top_20261021_*`: all AOJ jets, every weight 1) |
| 2a. MC test | held-out TTToHadronic + QCD; TTToSemiLeptonic as a second top sample | in the fine-tuning sets |
| 2b. Top-enriched region | real data (SingleMuon, SingleElectron 2016G+H) vs MC, semileptonic ttbar tag-and-probe | MC: `an` and the looser `skim` done and validated; **data not processed** |

## 0. Pretraining (done)
- AOJ: JetHT 2016G+H, AOJ selection (below), 176.6M jets, per-jet random split 100:5:10 (`DATASET.md`).
- The evaluation-region probes overlap with it (about 2/3 of the final probes also fire the JetHT trigger,
  i.e. are AOJ jets). That is intended: pretraining on the unlabeled target domain is the point. No exclusion.

## 1. Fine-tuning (MC only)

**AOJ selection** (the only cuts, identical to the pretraining data; `H5_maker.py`): the 85-path JetHT trigger
OR and the 8 MET filters (also applied to MC), AK8 pT > 300 GeV, |eta| < 2.5, jetId == 6. No lepton, MET, b-tag,
tau32 or mSD cut. **tau32 is not used in fine-tuning** (it is only an evaluation-region cut).

| | |
|---|---|
| Signal | TTToHadronic (UL16 post-VFP, Open Data 67840), AK8 jets with `top_match == 2` (b, q, q' within dR < 0.8 of the jet): 1.30M jets in 1.14M events. Both tops of an event are kept when both are merged (14 % of events, 25 % of jets). |
| Background | QCD_Pt_{300..3200} pythia8 (9 pT-hat bins, 475M AOJ jets), no gen matching. Cross sections: Open Data records. |
| Jets per event | all AOJ jets, as in pretraining (JetClass-II also keeps every jet): every merged top (both when both are merged; in 97 % of those events they are the two leading jets) and every QCD jet. `jet_rank` (0 = leading) and `n_jets` are stored. Leading: 66 % of the TTToHadronic tops, 85 % of the TTToSemiLeptonic tops, 66 % (lnpt) / 61 % (lnpt-lnmsd) of the QCD, 61 % of the AOJ data jets. Leading-only would cut the train tops by 1/3 (910k -> 604k); `--jets leading` builds that variant. |
| Not used | Partially merged ttbar jets (top_match 0/1); TTToSemiLeptonic (test only). |
| Splits | per event, u = splitmix64((lumi << 32) \| event) / 2^64: train u < 0.7, val 0.7-0.8, test >= 0.8, for TTToHadronic and for QCD. No veto (TTToHadronic is left out of the evaluation MC instead). |
| Second test | all TTToSemiLeptonic fully merged tops (863k jets); its background is the `test` QCD. |
| Class ratio | 1:1 in every split. |
| Weights | none: every jet has weight 1, train unweighted (top: \|genWeight\| is constant, 2.4 % negative; keep or drop the sign). `w_phys` gives physical yields. |
| Size | train 0.91M top + 0.91M QCD (lnpt-lnmsd: 0.89M + 0.89M), val 0.13M + 0.13M, test 0.26M + 0.26M. |

### Kinematic sampling: two versions
Flat per class (GloParT: uniform in mSD x ln pT for every class, arXiv:2604.09809) is impossible with SM tops:
they leave most of the plane empty (0 tops at mSD > 200 GeV below pT 400, almost none below 60 GeV), and only
72 of 1.3M tops would survive even in mSD 140-200 x pT 300-1000. GloParT has dedicated Z' -> tt samples with
the top mass itself varied. So instead **QCD is sampled to match the top distribution**:

| set | QCD matched to the top in | the tagger cannot use |
|---|---|---|
| `finetune_top_20261021_lnpt` | ln pT (60 bins 300-2500 GeV + overflow) x \|eta\| (0 / 0.8 / 1.6 / 2.5) | pT, eta (it can use the mass) |
| `finetune_top_20261021_lnpt-lnmsd` | the same x ln mSD (40 bins 20-300 GeV + mSD < 20 + mSD > 300) | pT, eta, mass (mass-decorrelated) |

How the sampling works (`finetune/make_finetune_set.py`):
- **Tops: no sampling**, except in the cells short of QCD (below). Every other fully merged top of the split is
  kept, with weight 1.
- **QCD: per-jet accept/reject, not event weights.** Each QCD jet i in the split's pool gets a keep probability
  `p_i = N_QCD x f_top(cell) x w_i / W_QCD(cell)`. Here f_top is the top fraction in the jet's cell,
  w_i = genWeight x sigma_bin / sumGenWeight_bin, and W_QCD is the summed w in the cell. A deterministic per-jet
  random number (hash of lumi, event and the jet's rank) keeps the jet if it is below p_i. Kept jets have
  weight 1.
- Because p_i is proportional to w_i, the pT-hat bins mix by cross section inside every cell. Two jets of one
  QCD event are decided independently, but always in the same split.
- **Cells short of QCD** (p_i > 1: fewer QCD MC jets than the tops need, only in the ln mSD version, at low pT
  and top-like mass): keep a fraction 1 / max_cell(p_i) of the cell's tops (per-jet hash) and scale the cell's
  p_i by the same fraction (`--qcd-scarce drop-top`), so every jet has weight 1 and the pT-hat mix in the cell
  stays the cross-section one. lnpt: none. lnpt-lnmsd: 71 of 7,686 cells; 15,059 train tops dropped (1.7 %),
  99 % of them at pT 310-430 GeV and mSD 160-260 GeV; a short cell keeps >= 22 % of its tops. There every jet of
  the highest-weight pT-hat bin (QCD_Pt300) is used. The 20261018 set kept those tops and gave 5.5 % of the
  train QCD a weight up to 4.6 instead; like JetClass-II, which only down-samples (weights <= 1), now no jet
  stands for more than itself.
- **Match quality** (`check_finetune_set.py`, train): pT fractions agree to 0.1 % in both sets. lnpt-lnmsd:
  mSD agrees to 0.3 % abs. except a within-bin residual at the top peak (140-160 GeV: 15.6 % top vs 16.3 % QCD;
  180-200 GeV: 26.4 % vs 25.5 %), from the 12 GeV-wide ln-mSD bins there; finer mSD bins would reduce it.
- Why sampling rather than weights: per-jet physical weights span ~1e5 between pT-hat bins, which makes
  training noisy.

### Other fine-tuning choices
- **Label-fraction study** (suggested): nested subsets of `train` by the same event hash (e.g. u < 0.007,
  0.07, 0.7 -> 10k, 100k, 910k tops). RINO and baselines then see identical labels.
- **Reference tagger:** `jet_tagging` carries CMS ParticleNet `particleNet_TvsQCD` for every jet (MC and data).

## 2a. MC test (top vs QCD)
- `test` (TTToHadronic + QCD): ROC and QCD rejection at eps_top = 30 / 50 / 70 %, in pT bins 300-400, 400-500,
  500-600, 600-800, > 800. Fix the working points here (at fixed QCD efficiency) and reuse them in data.
- `test_semilep`: the same metrics on TTToSemiLeptonic tops against the same QCD (a second ttbar sample).

## 2b. Top-enriched region: data vs MC (`ttbar_semilep/`)

| | |
|---|---|
| Data | SingleMuon (mu channel, records 30513 / 30546) + SingleElectron (e channel, 30512 / 30545), 2016G+H MINIAOD, golden JSON, 16.39 fb^-1. **Not processed:** ~18.6 TB, ~4,950 files -> Stage-1 PFNano + `ttbar_maker.py`, at LPC. |
| MC | TTToSemiLeptonic, TTTo2L2Nu, single top (tW, t-channel, s-channel), W+jets (8 HT bins), QCD MuEnriched (mu) and HT-binned QCD (e), `--store one_probe`. Two passes at LPC: `<store>/ttbar/` with the `an` preset (done), and `<store>/ttbar_skim/` with the looser `skim` preset (more statistics, and an anti-isolated sideband for a data-driven QCD check; 50 % done on 2026-10-18: ttbar and single top finished, W+jets partial, QCD started). `an` is recovered exactly from `skim` offline, so `ttbar_skim/` becomes the evaluation MC and the final cuts are chosen on it (statistics vs purity). **Not TTToHadronic** (0.03 % of the region), so the evaluation MC is independent of fine-tuning. |
| Normalization | gen_weight x sigma x L / sum(sum_genweight). sigma: ttbar 831.76 pb x BR (NNLO), single top NLO, W+jets x 1.21, QCD from the Open Data records. |
| Truth (MC) | per probe: fully merged t (the fine-tuning signal definition) / merged W / b+q / other; + single top, W+jets, QCD. |
| Sampling | none: all events, physical weights. |

**Selection** (default `an` preset, the HIG-24-010 analysis-note ttbar CR adapted to 2016 and AOJ; the `skim`
preset loosens it to: the `an` OR the high-pT lepton triggers (Mu50, TkMu50, Ele115), a tight-ID lepton with
pT > 30 and relIso < 1, no extra-lepton veto, MET > 20, any number of AK8 jets > 300 with the probe = the leading
AOJ jet at dR(lepton) > 0.8; dphi / b-tag / tau32 as below):

| stage | cut |
|---|---|
| trigger | mu: HLT_IsoMu24 \|\| HLT_IsoTkMu24; e: HLT_Ele27_WPTight_Gsf \|\| HLT_Ele32_WPTight_Gsf |
| filters, JSON | the 8 AOJ MET filters; golden JSON (data) |
| one lepton | exactly one tight lepton, pT > 55, \|eta\| < 2.4 (mu tightId, relIso < 0.15; e MVA WP80 iso, no gap); no other isolated lepton > 30 |
| MET | > 50 GeV |
| one probe | exactly one AK8 jet with pT > 300, \|eta\| < 2.5, and it is an AOJ jet (jetId == 6) |
| dphi | dphi(lepton, probe) > 2.0 |
| b-tag | >= 1 AK4 jet pT > 40, \|eta\| < 2.4, DeepJet medium (0.2489), dR(b, probe) > 1.5, dR(b, lepton) > 0.5 |
| tau32 | probe tau32 < 0.5 (evaluation only) |

Expected with the `an` cuts (full MC, mu + e, 16.39 fb^-1; the skim yields are still to be computed):

| stage | MC events | fully merged t | merged W | b+q | other | W+jets | QCD |
|---|---|---|---|---|---|---|---|
| b-tag | 29.5k | 0.07 | 0.16 | 0.19 | 0.48 | 0.09 | 0.02 |
| tau32 | 1,728 | 0.65 | 0.12 | 0.11 | 0.11 | 0.013 | 0.003 |

(t/W/bq/other include single top, 4 %. Data projected from the R&D subsets: ~1,580 events after tau32, ~1,000
merged tops; data/MC ~0.91 without corrections.)

**Skim MC (done 2026-10-21) and the evaluation selection.** `ttbar_skim/` is complete (29 samples, 7,445 chunks)
and consistent with `ttbar/`: every chunk has the same n_entries and sum_genweight, and `an` recovered from the
skim gives exactly the same events at the dphi / b-tag / tau32 stages with identical probe pT and tau32
(`ttbar_semilep/rnd/skim_vs_an.py`). Loosening `an` step by step (`rnd/skim_variants.py`, MC, TTToHadronic
excluded), after tau32:

| selection | pT > 300: MC evts / merged t / purity | pT > 400: MC evts / merged t / purity | QCD (pT > 300) |
|---|---|---|---|
| V0 `an` | 1,726 / 1,122 / 0.65 | 918 / 739 / 0.81 | 0.3 % |
| V1 = V0 without the extra-AK8 veto | 2,973 / 2,057 / 0.69 | 1,936 / 1,547 / 0.80 | 0.3 % |
| V2 = V1 + isolated lepton pT > 30 | 3,555 / 2,450 / 0.69 | 2,283 / 1,826 / 0.80 | 0.3 % |
| V3 = V2 + MET > 20 | 4,193 / 2,866 / 0.68 | 2,612 / 2,083 / 0.80 | 0.4 % |
| V4 = full skim (non-isolated leptons, Mu50 / Ele115) | 5,353 / 3,304 / 0.62 | 3,222 / 2,409 / 0.75 | 6.4 % |

At the b-tag stage (before tau32) V3 has 66k MC events and 5.3k merged tops (8 %), against 29k and 2.0k for V0.
**Proposal:** evaluate with V3 (2.6x the merged tops of `an` at the same purity), keep V0 as a cross-check, and
use the non-isolated part of V4 only as a QCD sideband. With extra AK8 jets allowed, the probe is the leading AOJ
jet at dR(lepton) > 0.8. Lepton pT 30-55 sits at the electron trigger turn-on (Ele27/32); an electron pT > 35
cut is a possible refinement.

**Measurement (proposal, to confirm):**
- Score distributions, data vs MC stacked by truth category, at the tau32 stage (65 % merged tops).
- Scale factor eps_data / eps_MC for fully merged tops at each working point, per pT bin (300-400, 400-500,
  > 500), from a pass/fail template fit with the category normalizations floating, at the **b-tag stage**
  (no tau32: tau32 is correlated with any top tagger, so an SF measured after the cut only applies to tops
  that pass it). The mSD of the probe is the natural fit variable for the lnpt set. For the lnpt-lnmsd set
  the score is decorrelated from mSD, so mSD can also serve as a sideband/control.

**Corrections:**
- available (POG correctionlib, 2016postVFP_UL): mu trigger `NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight`,
  mu ID/iso, e ID (wp80iso) and reco.
- **no electron trigger SF**: do the SF fit in the mu channel, use e as a cross-check (or measure the e trigger
  efficiency with an orthogonal dataset later).
- **not stored yet by `ttbar_maker.py`** (needs a change + MC rerun before data): `Pileup_nTrueInt` (pileup
  reweighting), the selected AK4 jets' pt / eta / hadronFlavour / DeepJet score (b-tag SFs), the electron
  supercluster eta (e SFs), both generator tops' pT (top-pT reweighting, optional).
- In a pass/fail fit with floating normalizations these mostly cancel; for score-shape comparisons they matter.

## Message for the LPC session

> Fine-tuning sets are built on UAF from your outputs; nothing needed for them. For the evaluation:
> 1. **Push** the `ttbar_semilep` v2 code (the `skim` preset in `cmssw/ttbar_semilep_v2.tar.gz`) to
>    `origin/ttbar-semilep` before any further change, so both sites work from the same code.
> 2. **Cross sections:** the QCD MuEnriched values in LPC_PROMPT.md were 16-33 % too high. Use the Open Data
>    record values (`cross_section.total_value`, filter efficiency included), pb: Pt-120To170 21180,
>    170To300 7011, 300To470 620.3, 470To600 59.08, 600To800 18.20, 800To1000 3.275, 1000 1.079; QCD_HT
>    300 324700, 500 30350, 700 6425, 1000 1117, 1500 108.3, 2000 21.94. Inclusive QCD_Pt bins: 300 6832,
>    470 551.9, 600 156.5, 800 26.25, 1000 7.475, 1400 0.6482, 1800 0.08746, 2400 0.005235, 3200 0.0001352.
>    Keep ttbar 831.76 pb x BR, single top NLO, W+jets x 1.21.
> 3. **Leave TTToHadronic out of the tag-and-probe MC** (0.03 % of the region); it is the fine-tuning signal.
> 4. The skim pass is finished and validated on UAF. Now **extend `ttbar_maker.py`** (new columns only; keep
>    the existing ones): `Pileup_nTrueInt` (MC), `PV_npvs`; for the AK4 jets used by the b-tag stage (pT > 40,
>    |eta| < 2.4, dR cuts; up to 8 per event): pt, eta, hadronFlavour (MC), btagDeepFlavB; the tight electron's
>    `deltaEtaSC`; MC: both generator tops' pT. Process the data with it, and rerun only TTToSemiLeptonic,
>    TTTo2L2Nu, single top and W+jets with it (QCD is <= 2 % of the region; it can stay as it is).
> 5. **Then process the data**: SingleMuon 30513 / 30546 and SingleElectron 30512 / 30545 (2016G/H MINIAOD),
>    Stage-1 data config + `ttbar_maker.py --sample_type data -j Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt
>    --trigger lepton_or_jetht --store one_probe --preset skim` (the same preset as the MC; `an` is
>    recovered from it offline). Take mu events from SingleMuon and e events from SingleElectron.
> 6. QCD_Pt2400: its normalization is right (it is normalized by its own sumGenWeight); only its statistics
>    are low. No action.

## Decisions
| | decision |
|---|---|
| A. kinematics | two versions: `lnpt` and `lnpt-lnmsd` (QCD matched to the top; flat per class is impossible with SM tops) |
| B. tau32 | not in fine-tuning. Evaluation: distributions after tau32; SF at the b-tag stage (proposal) |
| C. LPC | message above: push v2, new xsecs, no TTToHadronic, finish the skim, extend the maker, then data (`skim`) + non-QCD MC rerun |
| D. pretraining overlap | intended, no exclusion |
| veto | none (TTToHadronic out of the evaluation MC) |
| jets per event | every AOJ jet, as in pretraining (proposed and built; `--jets leading` for the leading-only variant) |
| QCD-short cells | drop tops there, so every weight is 1 (not QCD weights > 1) |

## Datasets on the store (`/ceph/cms/store/user/zichun/parcel/AOJ/`)
The current fine-tuning sets are also on the Nautilus PVC: `/rinovol-central/AOJ/finetune_top_20261021_*`.

| directory | content | status |
|---|---|---|
| `finetune_top_20261021_lnpt` | QCD matched in ln pT x \|eta\|; all AOJ jets, every weight 1, `jet_rank` / `n_jets` | **current**, checked |
| `finetune_top_20261021_lnpt-lnmsd` | QCD matched in ln pT x ln mSD x \|eta\|; 1.7 % of the tops dropped in QCD-short cells, every weight 1 | **current**, checked |
| `finetune_top_20261018_*` | `lnpt`: the same jets as `20261021_lnpt`, no `jet_rank` / `n_jets`; `lnpt-lnmsd`: every top kept, scarce QCD with weight up to 4.6 | deleted 2026-10-25 |
| `finetune_top_20261017_*` | `lnpt-msd` (20 ln-pT bins x linear mSD), `pteta` (20 GeV pT x \|eta\|, 2:1 test QCD), `pteta-split90-semileptest` (90/10); all with the veto | deleted 2026-10-25 |
| `ttbar/` | tag-and-probe MC (`an`, one_probe) | done; superseded by `ttbar_skim/` (identical `an` events) |
| `ttbar_skim/` | tag-and-probe MC (`skim`, one_probe, ttbar_semilep v2), 12 GB | done 2026-10-21, validated; the evaluation MC |
| `eval_events_TT.h5` | (run, lumi, event) of the tag-and-probe MC events | not used as a veto; `check_finetune_set.py` reports the overlap |
