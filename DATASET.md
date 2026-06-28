# AOJ skimmed dataset — access guide (for writing a dataloader)

Self-contained reference for the **Aspen Open Jets** ML dataset: boosted AK8 jets
from 2016 CMS Open Data (JetHT, eras 2016G+2016H), skimmed to per-particle ML
features and split train/val/test. **~176.6 M jets**, predominantly **QCD**
(real data: `pt > 300 GeV`, `|eta| < 2.5`, tight jet ID, certified lumisections).

## Location
Produced on the **UCSD ceph**, under `/ceph/cms/store/user/zichun/parcel/AOJ/`:

| path | what |
|---|---|
| `skimmed/train/train_NNNN.h5` | 77 shards, 153,531,392 jets (86.96%) |
| `skimmed/val/val_NNNN.h5`     | 4 shards, 7,680,203 jets (4.35%) |
| `skimmed/test/test_NNNN.h5`   | 8 shards, 15,346,822 jets (8.69%) |
| `merged.h5` | full un-sharded, un-skimmed source (raw `Px,Py,Pz,E` constituents) |
| `h5/out_*.h5` | the 1273 per-chunk producer outputs (raw) |

Each shard holds **≤ 2,000,000 jets** (last shard per split is smaller).
**For Nautilus training the `skimmed/` tree must be on the `rinovol-central` PVC**
(60 TB, ns `cms-ml`, mounted at `/rinovol-central`); the code lives at
`/rinovol-central/AOJProcessing`. Copy the data with the usual ceph→PVC pattern
(interactive pod + `kubectl cp`, or an xrootd/rsync job).

## Shard schema
One row = one jet. Column names are also stored in `dset.attrs["columns"]`.

**`PFCands` (N, 150, 16) float32** — up to 150 PF constituents per jet, **pT-sorted
descending**, zero-padded. Column index → name (read `dset.attrs["columns"]` to be safe):
```
 0 pt        4 deta       8 dR         12 d0Err
 1 eta       5 dphi       9 pdgId      13 dz
 2 phi       6 logpt     10 charge     14 dzErr
 3 E         7 logptrel  11 d0         15 puppiWeight
```
- `deta = eta - jet_eta`, `dphi = Δφ(phi, jet_phi)` (wrapped to (-π,π]), `logptrel = log(pt/jet_pt)`, `dR = sqrt(deta²+dphi²)`.
- **Padding:** jets with < 150 constituents have trailing all-zero rows. Real-constituent
  mask = `PFCands[..., 0] > 0` (pt > 0). `n_real = mask.sum(-1)` (≤ 150).

**`jet_kinematics` (N, 4) float32** — `pt, eta, phi, msoftdrop` (msoftdrop = -1 means no valid soft-drop mass).

**`jet_tagging` (N, 13) float32** — `nConstituents, tau1, tau2, tau3, tau4,
particleNet_H4qvsQCD, particleNet_HbbvsQCD, particleNet_HccvsQCD, particleNet_QCD,
particleNet_TvsQCD, particleNet_WvsQCD, particleNet_ZvsQCD, particleNet_mass`.
Reference taggers / soft labels. `nConstituents` is the **full** count (can exceed 150).

(`event_info` = run/lumi/event was dropped during the skim.)

## Split
Per-**jet** random assignment, **seed 42**, ratio 100:5:10. Jets keep source order
within each split (the split is a random subset, not a shuffle) → **shuffle `train`
at load time**.

## Loading (h5py)
```python
import h5py, glob, numpy as np
shards = sorted(glob.glob("/rinovol-central/.../skimmed/train/train_*.h5"))
with h5py.File(shards[0]) as f:
    cols = f["PFCands"].attrs["columns"].split(",")   # feature names, in order
    pf   = f["PFCands"][:]          # (n, 150, 16)
    jk   = f["jet_kinematics"][:]   # (n, 4)
    jt   = f["jet_tagging"][:]      # (n, 13)
mask = pf[..., 0] > 0               # (n,150) real-constituent mask; padding -> False
```
Counts per shard: `h5py.File(s)["jet_kinematics"].shape[0]`.

## Dataloader notes
- Iterate shards (lazy `h5py` reads, e.g. `f["PFCands"][i0:i1]`); shuffle `train`
  across shards and within batches. val/test order doesn't matter.
- Feed `mask` to set-based models (Transformer/ParticleNet/ParT ignore padded slots).
- Normalization: `pt/E` are raw GeV (heavy tails) — `logpt/logptrel` are provided for
  that; standardize per-feature using **train**-set mean/std; `d0,dz` and their errors
  have long tails (consider clipping). `eta,phi,deta,dphi,dR,charge,puppiWeight` are
  already O(1).
- Labels: this is **real, predominantly-QCD data** (unlabeled). For supervised tagging,
  pair with gen-matched MC signal (`H5_maker.py --gen_match {6:top,24:W,23:Z,25:H}`),
  or use the ParticleNet scores in `jet_tagging` as soft/reference labels. For
  self-supervised / anomaly / domain-adaptation, use as-is.

## Provenance / regeneration
Repo: `github.com/zichunhao/AOJProcessing`. Pipeline: `H5_maker.py` (MINIAOD→per-jet
HDF5 via PFNano), `H5_merge.py` (merge), `H5_skim_split.py` (this skim+split+shard;
re-runnable with different `--ratios`/`--jets-per-shard`/`--seed`). Batch production:
`condor/`. See `CLAUDE.md`.
