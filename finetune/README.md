# Fine-tuning sets: fully merged top jets vs QCD (CMS UL16 MC)

Supervised top-vs-QCD jets in the same per-jet format as the AOJ skimmed shards (`DATASET.md`), for fine-tuning
models pretrained on AOJ. Built by `finetune/make_finetune_set.py`, checked by `finetune/check_finetune_set.py`;
the overall plan (fine-tuning and evaluation) is `finetune/PLAN.md`.

Location: `/ceph/cms/store/user/zichun/parcel/AOJ/finetune_top_<yyyymmdd>_<setting>/` (UCSD ceph), copied to
`/rinovol-central/AOJ/finetune_top_<yyyymmdd>_<setting>/` (Nautilus PVC `rinovol-central`, ns `cms-ml`), each
with this README, `summary.json` (counts, binning, cross sections) and `qcd_pass1.npz` (the QCD histograms the
sampling used).

| set (`<setting>` = `--match`) | QCD matched to the top in | the tagger cannot use |
|---|---|---|
| `finetune_top_20261021_lnpt` | ln pT (60 bins 300-2500 GeV + overflow) x \|eta\| (0 / 0.8 / 1.6 / 2.5) | pT, eta |
| `finetune_top_20261021_lnpt-lnmsd` | the same x ln mSD (40 bins 20-300 GeV + mSD < 20 + mSD > 300) | pT, eta, mass |

Both sets take every AOJ jet of an event (`--jets all`), and every jet has `weight` = 1 (`--qcd-scarce drop-top`).

| split | top `lnpt` | QCD `lnpt` | top `lnpt-lnmsd` | QCD `lnpt-lnmsd` | top source |
|---|---|---|---|---|---|
| `train/` | 909,927 | 910,128 | 894,868 | 894,218 | TTToHadronic, events with u < 0.7 |
| `val/` | 130,550 | 130,859 | 128,288 | 128,086 | TTToHadronic, 0.7 <= u < 0.8 |
| `test/` | 260,339 | 260,266 | 256,255 | 256,306 | TTToHadronic, u >= 0.8: the held-out test set |
| `test_semilep/` | 863,022 | (use `test/qcd`) | 863,022 | (use `test/qcd`) | TTToSemiLeptonic: a second ttbar sample, never trained on |

`lnpt-lnmsd` has 1.7 % fewer tops: where the MC has fewer QCD jets of top-like mass than the tops need (mostly
pT 310-430 GeV, mSD 160-260 GeV), part of the tops is dropped rather than weighting QCD up (below).

## Selection and labels
- Every jet passes the AOJ selection of `H5_maker.py` (the JetHT trigger OR and 8 MET filters, also on MC; AK8
  pT > 300, |eta| < 2.5, jetId == 6), as the pretraining data does. No other cut (no tau32 / mSD / lepton).
- **All AOJ jets of an event**, as in the pretraining data (`H5_maker.py` keeps every jet that passes): every
  fully merged top and every QCD jet, whatever its pT rank. `jet_rank` (0 = the event's leading AOJ jet) and
  `n_jets` record it (table below).
- `top` = fully merged top: the b, q and q' of one hadronic top within dR < 0.8 of the jet (`top_match == 2`).
  Both tops of a TTToHadronic event are kept when both are merged (14 % of the events, 25 % of the jets; in 97 %
  of these events they are the two leading jets). Partially merged tops are not used.
- `qcd` = QCD_Pt_{300..3200}_TuneCP5 pythia8 jets, no gen matching.
- Splits are per event: u = splitmix64((luminosityBlock << 32) | event) / 2^64 (`u01()` in the maker), for
  TTToHadronic and for QCD (pools train u < 0.7, val 0.7-0.8, test >= 0.8). No event is in two splits. No veto:
  TTToHadronic is left out of the tag-and-probe evaluation MC instead.

| jets | leading / 2nd / 3rd+ AOJ jet of the event |
|---|---|
| top, TTToHadronic (train) | 66 / 33 / 0.7 % |
| top, TTToSemiLeptonic (`test_semilep`) | 85 / 15 / 0.4 % |
| QCD `lnpt` (train) | 66 / 34 / 0.4 % |
| QCD `lnpt-lnmsd` (train) | 61 / 39 / 0.5 % |
| AOJ data (pretraining; 12M-jet sample) | 61 / 39 / 0.8 % |

## Sampling and weights
- QCD is sampled per jet (accept/reject): jet i is kept with probability
  p_i = N_qcd x f_top(cell) x w_i / sum_cell(w), where f_top is the split's top fraction in the jet's cell and
  w_i = genWeight x sigma(pT-hat bin) / sumGenWeight(bin). So QCD has the top's distribution in the cells, and
  inside each cell the pT-hat bins are mixed by their cross sections (Open Data record values). The decision
  uses a deterministic per-jet hash.
- **Cells short of QCD** (p_i > 1 for some jet: fewer QCD MC jets than the tops need): only a fraction
  1 / max_cell(p_i) of the cell's tops is kept (per-jet hash), and the cell's p_i are scaled by the same fraction.
  No jet needs a weight, and the pT-hat mix in the cell stays the cross-section one. `lnpt`: no such cell.
  `lnpt-lnmsd`: 71 of its 7,686 cells (train). Dropped: 15,059 train tops (1.7 %), 2,262 val, 4,084 test; 99 %
  of them at pT 310-430 GeV and mSD 160-260 GeV, the rest in a few cells at mSD > 245 GeV or at pT 800-1030 GeV.
  A short cell keeps >= 22 % of its tops (78 % on average). There the QCD MC is used up: every jet of the
  highest-weight pT-hat bin (QCD_Pt300 in 69 of the 71 cells) is kept.
- All other tops are kept; `test_semilep` tops are never dropped.
- Classes are balanced: QCD/top = 1.000 / 1.002 / 1.000 (`lnpt`) and 0.999 / 0.998 / 1.000 (`lnpt-lnmsd`) in
  train / val / test, i.e. equal up to the sampling fluctuation.
- `w_phys` = physical weight, jets expected in 16.39 fb^-1 (2016G+H), divided by the keep probability (QCD
  sampling, dropped tops): sum over a split estimates the full-sample yield (top 7.07e4 from TTToHadronic,
  3.42e4 from TTToSemiLeptonic). For QCD, only the cells that contain tops are sampled, so its sum covers only
  those: 1.95e8 in `lnpt`, 1.7-1.9e8 in `lnpt-lnmsd` (fewer cells have tops; few jets with large `w_phys` make it
  noisy). Use it for the natural QCD spectrum or a physical top/QCD mix in the top phase space.
- `gen_info[:, 0]` = genWeight. For the top samples |genWeight| is constant (316.5 / 303.4); 2.4 % of the jets
  have a negative one. A jet takes its event's weight; the two tops of an event are not down-weighted.
- `summary.json` -> `qcd_keep_probability`: max p_i = 1 (to float rounding; `p_above_1` counts the jets at
  1 + 2e-16).

## Training with it
- **Loss weights: none.** `weight` = 1 for every jet: train unweighted (plain mean loss, uniform shuffling).
  Never train with `w_phys` (physical normalization, for evaluation) or the genWeight sign.
- **Leading jets only:** `jet_rank == 0` (2/3 of the jets) is closest to the evaluation probe (the leading AOJ
  jet away from the lepton), e.g. for a cross-check of the test metrics. Its QCD is not matched to its tops on
  its own (`lnpt-lnmsd`: 61 % of the QCD vs 66 % of the tops are leading), so to train on leading jets only,
  build a set with `--jets leading`.
- **Feature normalization:** use the pretraining (AOJ) normalization, not statistics of this set.
- **Label-fraction subsets:** nested by the event hash, e.g. train jets with u01 < 0.007 / 0.07 / 0.7 (1 % /
  10 % / all of train), so every model sees the same labelled jets.

```python
import glob, h5py, numpy as np

def u01(event_info):
    """Per-event number in [0, 1): the split (train < 0.7 <= val < 0.8 <= test) and nested subsets."""
    k = (event_info[:, 1].astype(np.uint64) << np.uint64(32)) | event_info[:, 2].astype(np.uint64)
    with np.errstate(over="ignore"):
        k = k + np.uint64(0x9E3779B97F4A7C15)
        k = (k ^ (k >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        k = (k ^ (k >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        k = k ^ (k >> np.uint64(31))
    return (k >> np.uint64(11)).astype(np.float64) / float(1 << 53)

def load(setdir, split):
    """All jets of a split (top + QCD); test_semilep takes its QCD from test/."""
    parts = []
    for cls in ("top", "qcd"):
        d = "test" if (split == "test_semilep" and cls == "qcd") else split
        for p in sorted(glob.glob("%s/%s/%s_*.h5" % (setdir, d, cls))):
            with h5py.File(p, "r") as f:
                parts.append({k: f[k][:] for k in ("PFCands", "jet_kinematics", "label", "jet_rank", "event_info")})
    out = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    out["mask"] = out["PFCands"][..., 0] > 0          # real constituents (padding rows are all zero)
    return out
```
(The full train split is ~1.8M jets x 150 x 16 floats = 17 GB in memory: read shards lazily for training.)

## Datasets (one row per jet)
| dataset | shape | notes |
|---|---|---|
| `PFCands` | (N, 150, 16) float32 | as in `DATASET.md` (`columns` attribute); padding rows all zero, mask `PFCands[..., 0] > 0` |
| `jet_kinematics` | (N, 4) | pt, eta, phi, msoftdrop |
| `jet_tagging` | (N, 13) | nConstituents, tau1-4, ParticleNet scores (incl. `particleNet_TvsQCD`), PN mass |
| `event_info` | (N, 3) int64 | run, luminosityBlock, event |
| `gen_info` | (N, 5) | genWeight, top_match, hadronFlavour, nBHadrons, nCHadrons |
| `label` | (N,) int8 | 1 top, 0 QCD |
| `sample` | (N,) int8 | index into the `names` attribute (TTToHadronic, TTToSemiLeptonic, QCD_Pt300, ...) |
| `jet_rank`, `n_jets` | (N,) int8 | pT rank of the jet among the AOJ jets of its event (0 = leading), and their number. NanoAOD rounds pT, so two jets of an event can have the same pT: the earlier one in the file ranks first |
| `weight`, `w_phys` | (N,) float32 | see above (`weight` = 1 for every jet) |

Shards hold up to 1M jets, chunked 128 jets x 150 x 16 for random access. Jets are in source order and the
classes are in separate files (`top_NNNN.h5`, `qcd_NNNN.h5`): interleave and shuffle at load time.

## Notes
- Flat per class in (mSD, ln pT), as GloParT samples (arXiv:2604.09809), is impossible with SM tops, which
  leave most of that plane empty; GloParT uses dedicated Z' -> tt samples with a varied top mass. Matching QCD
  to the top gives the same decorrelation for a binary tagger without losing tops.
- In `lnpt-lnmsd`, QCD jets with mSD < 20 GeV (half of all QCD jets) are as rare as top jets there (0.3 %).
- QCD_Pt2400 has 1 of its 76 files on eospublic: its normalization is right, its statistics are not.
- Earlier sets, deleted on 2026-10-25: `finetune_top_20261018_lnpt` had exactly the jets of `20261021_lnpt`,
  without `jet_rank` / `n_jets`; `finetune_top_20261018_lnpt-lnmsd` kept every top and gave the scarce QCD
  (5.5 % of the train QCD) `weight` up to 4.6 instead (`--qcd-scarce weight`); `finetune_top_20261017_*` were
  older still. See `PLAN.md`.
