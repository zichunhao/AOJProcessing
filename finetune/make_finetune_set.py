#!/usr/bin/env python3
"""Fine-tuning set for top tagging: fully merged top jets vs QCD jets, in the AOJ skimmed-shard format.

    bash ttbar_semilep/py3env.sh finetune/make_finetune_set.py --store STORE -o OUTDIR [--plan] [--workers 48]

Inputs (STORE = the AOJ store root; all jets pass the AOJ selection of H5_maker.py: JetHT trigger OR,
8 MET filters, AK8 pT > 300, |eta| < 2.5, jetId == 6; no other cut):
  signal train/val/test  merged_TTToHadronic_topmatched.h5      (top_match == 2: b, q, q' of one top within dR < 0.8)
  signal test_semilep    merged_TTToSemiLeptonic_topmatched.h5  (a different sample: never used in training)
                         merged_<sample>.h5 (all AOJ jets) gives the pT rank of each top in its event
  background        h5/out_QCD_Pt<bin>_sub*.h5             (9 pT-hat bins, per-job files)
Jets: every AOJ jet of an event is a candidate, as in the AOJ pretraining data (H5_maker.py keeps all of them):
every fully merged top (both tops of a TTToHadronic event) and every QCD jet. --jets leading keeps only the
leading AOJ jet of each event, in both classes.
Splits are per EVENT (both tops of a TTToHadronic event, or all QCD jets of an event, share a split), from
  u = splitmix64((luminosityBlock << 32) | event) / 2^64   (u01() below; MC has one run number)
  TTToHadronic: train u < 0.7, val 0.7 <= u < 0.8, test u >= 0.8 (the in-distribution test set); with --veto,
                the events of a veto list are dropped from train and val (default none: TTToHadronic is left
                out of the tag-and-probe MC instead)
  TTToSemiLeptonic: all in test_semilep (no veto)
  QCD: pools train u < 0.7, val 0.7-0.8, test u >= 0.8; test_semilep has no QCD of its own: its background is
       test/qcd (QCD with top-like mass at low pT is too scarce in the MC for two separate test pools)
QCD sampling: in each split, QCD jets are drawn from that split's pool so that their distribution in the --match
cells equals the split's signal, with N_qcd = --qcd-ratio x N_top (test splits: --qcd-test-ratio):
  --match lnpt        ln pT (60 bins 300-2500, 3.6 % wide, + overflow) x |eta| (0-0.8-1.6-2.5): the classes share pT and eta;
                      the tagger can use the jet mass
  --match lnpt_lnmsd  the same x ln mSD (40 bins 20-300 GeV + mSD < 20 + mSD > 300): the classes also share the
                      mass, so the tagger is mass-decorrelated
  --match lnpt_msd    20 ln-pT bins x linear 10 GeV mSD bins 20-260 (GloParT's binning) x |eta| (finetune_top_v3)
  --match pt_eta      pT (20 GeV ...) x |eta| (0.25 bins) (finetune_top_v2)
(Flat per class, as GloParT samples, is impossible with SM tops: they leave most of the (pT, mSD) plane empty.)
Within a cell the pT-hat bins are mixed by cross section: jet i is kept with probability
  p_i = N_qcd * f_top(cell) * w_i / sum_{pool, cell} w,   w_i = genWeight * sigma_bin / sumGenWeight_bin,
decided by a second per-jet hash (deterministic). Where the MC has too few QCD jets for this (p_i > 1 in a cell):
  --qcd-scarce drop-top  (default) keep only a fraction 1 / max_cell(p_i) of the cell's tops (per-jet hash) and
                         scale the cell's p_i by the same fraction: every jet has weight 1, and the pT-hat mix in
                         the cell stays the cross-section one
  --qcd-scarce weight    keep every top; QCD jets with p_i > 1 are kept with weight p_i (the 20261018 sets)
test_semilep tops are never dropped. Per-jet outputs:
  weight   training weight: 1, or p_i for the capped QCD jets of --qcd-scarce weight
  w_phys   physical weight [jets in --lumi]: genWeight * sigma * L / sumGenWeight / (pool fraction * keep probability),
           so sum(w_phys) over a split estimates the full-sample yield (natural top / QCD mix and spectrum)
  jet_rank pT rank of the jet among the AOJ jets of its event (0 = leading); n_jets = AOJ jets in the event
Output: OUTDIR/<split>/<top|qcd>_NNNN.h5 with PFCands (N,150,16) (H5_skim_split.py features), jet_kinematics,
jet_tagging, event_info, gen_info, label (1 top, 0 QCD), sample (code; names in the attribute), weight, w_phys,
jet_rank, n_jets; OUTDIR/summary.json. Classes are in separate files, in source order: shuffle at load time.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from H5_skim_split import skim_pf, PF_COLS, JK_COLS, JT_COLS  # noqa: E402

# Open Data record cross sections (GenXSecAnalyzer, pb): QCD_Pt_<bin>_TuneCP5_13TeV_pythia8 UL16 MiniAODv2
QCD_XS = {"QCD_Pt300": 6832., "QCD_Pt470": 551.9, "QCD_Pt600": 156.5, "QCD_Pt800": 26.25, "QCD_Pt1000": 7.475,
          "QCD_Pt1400": 0.6482, "QCD_Pt1800": 0.08746, "QCD_Pt2400": 0.005235, "QCD_Pt3200": 0.0001352}
TT_XS = {"TTToHadronic": 377.96, "TTToSemiLeptonic": 365.34}        # 831.76 pb x BR (NNLO+NNLL)
SAMPLES = list(TT_XS) + list(QCD_XS)                                # sample code = index
SPLITS = ["train", "val", "test", "test_semilep"]
QCD_POOL = {"train": (0.0, 0.7), "val": (0.7, 0.8), "test": (0.8, 1.0)}
QCD_SPLITS = [s for s in SPLITS if s in QCD_POOL]                   # test_semilep: background = test/qcd
SIG_POOL = {"train": (0.0, 0.7), "val": (0.7, 0.8), "test": (0.8, 1.0), "test_semilep": (0.0, 1.0)}
SIG_SRC = {"train": "TTToHadronic", "val": "TTToHadronic", "test": "TTToHadronic", "test_semilep": "TTToSemiLeptonic"}
TRAINED = ("train", "val")                                          # the splits the veto applies to
_ETA_FINE = np.arange(0, 2.5001, 0.25)
_ETA_FINE[-1] = 2.5001
BINNINGS = {
    "pt_eta": dict(pt=np.r_[np.arange(300, 1000, 20), np.arange(1000, 1500, 50), np.arange(1500, 2000, 100),
                            2000, 2500, 3000, 4000, 14000].astype(float),
                   msd=np.array([-1e9, 1e9]), eta=_ETA_FINE),
    "lnpt_msd": dict(pt=np.r_[np.exp(np.linspace(np.log(300.), np.log(2500.), 21)), 14000.],
                     msd=np.r_[-1e9, np.arange(20., 261., 10.), 1e9], eta=np.array([0., 0.8, 1.6, 2.5001])),
    "lnpt": dict(pt=np.r_[np.exp(np.linspace(np.log(300.), np.log(2500.), 61)), 14000.],
                 msd=np.array([-1e9, 1e9]), eta=np.array([0., 0.8, 1.6, 2.5001])),
    "lnpt_lnmsd": dict(pt=np.r_[np.exp(np.linspace(np.log(300.), np.log(2500.), 61)), 14000.],
                       msd=np.r_[-1e9, np.exp(np.linspace(np.log(20.), np.log(300.), 41)), 1e9],
                       eta=np.array([0., 0.8, 1.6, 2.5001])),
}
BLOCK = 20000
GEN_COLS = ["genWeight", "top_match", "hadronFlavour", "nBHadrons", "nCHadrons"]


def splitmix(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def to01(x):
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def evkey(ev):
    return (ev[:, 1].astype(np.uint64) << np.uint64(32)) | ev[:, 2].astype(np.uint64)


def u01(ev):
    """Per-event uniform number in [0, 1): the split."""
    return to01(splitmix(evkey(ev)))


def v01(ev):
    """Per-jet uniform number (event key + rank of the jet in its event): the QCD keep decision."""
    k = evkey(ev)
    start = np.r_[True, k[1:] != k[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(len(k)), 0))
    rank = (np.arange(len(k)) - first).astype(np.uint64)
    with np.errstate(over="ignore"):
        return to01(splitmix(splitmix(k) ^ ((rank + np.uint64(1)) * np.uint64(0xD6E8FEB86659FD93))))


def event_rank(ev, pt):
    """pT rank of each jet among the AOJ jets of its event (0 = leading), and the number of AOJ jets in the event.
    The jets of an event are consecutive rows of an H5_maker output (one FatJet loop per event)."""
    if len(ev) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    new = np.r_[True, np.any(ev[1:] != ev[:-1], axis=1)]
    gid = np.cumsum(new) - 1
    start = np.flatnonzero(new)
    order = np.lexsort((-pt, gid))
    rank = np.empty(len(pt), np.int64)
    rank[order] = np.arange(len(pt)) - start[gid[order]]
    return rank, np.diff(np.r_[start, len(pt)])[gid]


def ncell(B):
    return (len(B["pt"]) - 1) * (len(B["msd"]) - 1) * (len(B["eta"]) - 1)


def cell(jk, B):
    """Flat cell index of each jet (jet_kinematics rows: pt, eta, phi, msoftdrop) in binning B."""
    i = np.clip(np.digitize(jk[:, 0], B["pt"]) - 1, 0, len(B["pt"]) - 2)
    m = np.clip(np.digitize(jk[:, 3], B["msd"]) - 1, 0, len(B["msd"]) - 2)
    e = np.clip(np.digitize(np.abs(jk[:, 1]), B["eta"]) - 1, 0, len(B["eta"]) - 2)
    return (i * (len(B["msd"]) - 1) + m) * (len(B["eta"]) - 1) + e


def in_pool(u, pool):
    return (u >= pool[0]) & (u < pool[1])


def colnames(d):
    return [c.decode() if isinstance(c, bytes) else str(c) for c in np.atleast_1d(d.attrs["columns"])]


def label_of(path):
    return re.sub(r"_sub\d+\.h5$", "", os.path.basename(path))[len("out_"):]


# ------------------------------------------------------------------------------------- QCD pass 1
def qcd_hist(args):
    """Sum of genWeight per (split pool, cell), jet count, and the file's sumGenWeight."""
    path, B, jets = args
    H = np.zeros((len(QCD_SPLITS), 2, ncell(B)))
    with h5py.File(path, "r") as f:
        sgw = float(f.attrs["sumGenWeight"])
        n = f["jet_kinematics"].shape[0]
        if n:
            jk, ev = f["jet_kinematics"][:], f["event_info"][:]
            w = f["gen_info"][:, colnames(f["gen_info"]).index("genWeight")].astype(np.float64)
            u = u01(ev)
            c = cell(jk, B)
            sel = event_rank(ev, jk[:, 0])[0] == 0 if jets == "leading" else np.ones(n, bool)
            for s, name in enumerate(QCD_SPLITS):
                m = in_pool(u, QCD_POOL[name]) & sel
                H[s, 0] += np.bincount(c[m], weights=w[m], minlength=ncell(B))
                H[s, 1] += np.bincount(c[m], minlength=ncell(B))
    return path, sgw, n, H


# ------------------------------------------------------------------------------------- QCD pass 2
def qcd_select(args):
    """Keep-probabilities -> the kept jets of one file, skimmed, per split."""
    path, norm, scale, B, jets, scarce = args   # norm = sigma / sumGenWeight_bin; scale[s] = N_s f_top(cell) / Hq_s(cell)
    out, st = {}, dict(p_above_1=0, p_max=0.0)
    with h5py.File(path, "r") as f:
        n = f["jet_kinematics"].shape[0]
        if n == 0:
            return path, out, st
        jk_all, ev = f["jet_kinematics"][:], f["event_info"][:]
        gi_all = f["gen_info"][:]
        w = gi_all[:, colnames(f["gen_info"]).index("genWeight")].astype(np.float64) * norm
        u, v = u01(ev), v01(ev)
        rank, nj = event_rank(ev, jk_all[:, 0])
        c = cell(jk_all, B)
        p = np.zeros(n)
        split = np.full(n, -1)
        for s, name in enumerate(QCD_SPLITS):
            m = in_pool(u, QCD_POOL[name])
            if jets == "leading":
                m &= rank == 0
            p[m] = scale[s][c[m]] * w[m]
            split[m] = s
        st = dict(p_above_1=int((p > 1).sum()), p_max=float(p.max()))
        keep = (v < p) & (split >= 0)
        if not keep.any():
            return path, out, st
        rows = np.where(keep)[0]
        pf = []
        for b0 in range(0, n, BLOCK):
            r = rows[(rows >= b0) & (rows < b0 + BLOCK)]
            if len(r):
                pf.append(f["PFCands"][b0:min(b0 + BLOCK, n)][r - b0])
        pf = np.concatenate(pf)
        jt = f["jet_tagging"][:][rows]
    jk, gi = jk_all[rows], gi_all[rows]
    pk, sk = p[rows], split[rows]
    for s, name in enumerate(QCD_SPLITS):
        m = sk == s
        if not m.any():
            continue
        pool_frac = QCD_POOL[name][1] - QCD_POOL[name][0]
        out[name] = dict(PFCands=skim_pf(pf[m], jk[m]), jet_kinematics=jk[m], jet_tagging=jt[m], event_info=ev[rows][m],
                         gen_info=gi[m], label=np.zeros(m.sum(), np.int8),
                         sample=np.full(m.sum(), SAMPLES.index(label_of(path)), np.int8),
                         weight=(np.maximum(pk[m], 1.0) if scarce == "weight" else np.ones(m.sum())).astype(np.float32),
                         w_phys_unit=(w[rows][m] / (np.minimum(pk[m], 1.0) * pool_frac)),   # x lumi later
                         jet_rank=rank[rows][m].astype(np.int8), n_jets=nj[rows][m].astype(np.int8))
    return path, out, st


# ------------------------------------------------------------------------------------- writing
class Writer(object):
    """Append per-jet arrays, rolling to a new <prefix>_NNNN.h5 every `per` jets."""
    COLS = {"PFCands": PF_COLS, "jet_kinematics": JK_COLS, "jet_tagging": JT_COLS,
            "event_info": ["run", "luminosityBlock", "event"], "gen_info": GEN_COLS}

    def __init__(self, outdir, prefix, per, attrs):
        self.dir, self.prefix, self.per, self.attrs = outdir, prefix, per, attrs
        os.makedirs(outdir, exist_ok=True)
        self.idx, self.f, self.n, self.total = -1, None, 0, 0

    def _roll(self):
        if self.f:
            self.f.close()
        self.idx += 1
        self.n = 0
        self.f = h5py.File(os.path.join(self.dir, "%s_%04d.h5" % (self.prefix, self.idx)), "w")
        for k, v in self.attrs.items():
            self.f.attrs[k] = v

    def _create(self, k, a):
        chunk = (128,) + a.shape[1:] if a.ndim == 3 else (8192,) + a.shape[1:]
        d = self.f.create_dataset(k, shape=(0,) + a.shape[1:], maxshape=(None,) + a.shape[1:], dtype=a.dtype,
                                  chunks=chunk, compression="gzip")
        if k in self.COLS:
            d.attrs["columns"] = ",".join(self.COLS[k])
        if k == "sample":
            d.attrs["names"] = ",".join(SAMPLES)
        return d

    def write(self, batch):
        N = len(batch["label"])
        off = 0
        while off < N:
            if self.f is None or self.n >= self.per:
                self._roll()
            take = min(self.per - self.n, N - off)
            for k, a in batch.items():
                d = self.f[k] if k in self.f else self._create(k, a)
                n0 = d.shape[0]
                d.resize(n0 + take, axis=0)
                d[n0:] = a[off:off + take]
            self.n += take
            self.total += take
            off += take

    def close(self):
        if self.f:
            self.f.close()


def signal_jets(store, src):
    """Small datasets of the topmatched file, with the pT rank and the number of AOJ jets in the event of each jet,
    taken from merged_<src>.h5 (all AOJ jets of the sample; the topmatched file is its top_match == 2 subset)."""
    with h5py.File(os.path.join(store, "merged_%s_topmatched.h5" % src), "r") as f:
        d = dict(ev=f["event_info"][:], jk=f["jet_kinematics"][:], sumGenWeight=float(f.attrs["sumGenWeight"]),
                 nEventsProcessed=int(f.attrs["nEventsProcessed"]))
    with h5py.File(os.path.join(store, "merged_%s.h5" % src), "r") as f:
        ev, jk = f["event_info"][:], f["jet_kinematics"][:]
        t = f["gen_info"][:, colnames(f["gen_info"]).index("top_match")] == 2
    rank, nj = event_rank(ev, jk[:, 0])
    assert np.array_equal(ev[t], d["ev"]) and np.array_equal(jk[t], d["jk"]), \
        "merged_%s_topmatched.h5 is not the top_match == 2 subset of merged_%s.h5" % (src, src)
    d["rank"], d["n_jets"] = rank[t], nj[t]
    return d


def signal_rows(store, veto_path, B, jets):
    """Per split: (source file, sample, row indices) of the signal jets, their histograms in the cells of B, the
    sample info, and the signal_jets() arrays per sample."""
    veto = {}
    if veto_path:
        with h5py.File(veto_path, "r") as fv:
            veto = {k: evkey(fv[k][:]) for k in fv if not k.endswith("_stage")}
    rows, hist, info, sig = {}, {}, {}, {}
    for name in SPLITS:
        src = SIG_SRC[name]
        if src not in sig:
            sig[src] = signal_jets(store, src)
            info[src] = dict(sumGenWeight=sig[src]["sumGenWeight"], nEventsProcessed=sig[src]["nEventsProcessed"],
                             jets=len(sig[src]["ev"]), leading=int((sig[src]["rank"] == 0).sum()))
        ev, jk = sig[src]["ev"], sig[src]["jk"]
        m = in_pool(u01(ev), SIG_POOL[name])
        if jets == "leading":
            m &= sig[src]["rank"] == 0
        dropped = 0
        if src in veto and name in TRAINED:          # test splits are never trained on: keep all of them
            v = np.isin(evkey(ev), veto[src])
            dropped = int((m & v).sum())
            m &= ~v
        rows[name] = (os.path.join(store, "merged_%s_topmatched.h5" % src), src, np.where(m)[0])
        hist[name] = np.bincount(cell(jk[m], B), minlength=ncell(B)).astype(float)
        info["%s_%s_dropped_eval_events" % (name, src)] = dropped
    return rows, hist, info, sig


def write_signal(writer, path, src, rows, keep_p, pool_frac, lumi, store_info, sig):
    """Write the signal jets `rows` of `path`; keep_p = their keep probability (w_phys is divided by it)."""
    norm = TT_XS[src] * lumi * 1e3 / store_info[src]["sumGenWeight"] / pool_frac
    with h5py.File(path, "r") as f:
        n = f["jet_kinematics"].shape[0]
        for b0 in range(0, n, BLOCK):
            sel = (rows >= b0) & (rows < b0 + BLOCK)
            r = rows[sel] - b0
            if not len(r):
                continue
            sl = slice(b0, min(b0 + BLOCK, n))
            jk, gi = f["jet_kinematics"][sl][r], f["gen_info"][sl][r]
            writer.write(dict(PFCands=skim_pf(f["PFCands"][sl][r], jk), jet_kinematics=jk,
                              jet_tagging=f["jet_tagging"][sl][r], event_info=f["event_info"][sl][r], gen_info=gi,
                              label=np.ones(len(r), np.int8), sample=np.full(len(r), SAMPLES.index(src), np.int8),
                              weight=np.ones(len(r), np.float32),
                              w_phys=(gi[:, 0].astype(np.float64) * norm / keep_p[sel]).astype(np.float32),
                              jet_rank=sig["rank"][rows[sel]].astype(np.int8),
                              n_jets=sig["n_jets"][rows[sel]].astype(np.int8)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", default="/ceph/cms/store/user/zichun/parcel/AOJ")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--veto", default=None, help="event veto list for train/val, e.g. <store>/eval_events_TT.h5 (default: none)")
    ap.add_argument("--qcd-ratio", type=float, default=1.0, help="QCD / top jets in train and val (default 1)")
    ap.add_argument("--qcd-test-ratio", type=float, default=1.0, help="QCD / top jets in test (default 1)")
    ap.add_argument("--lumi", type=float, default=16.39, help="fb^-1 for w_phys (default 2016G+H)")
    ap.add_argument("--jets-per-shard", type=int, default=1000000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--match", choices=sorted(BINNINGS), required=True, help="cells QCD is matched to the top in")
    ap.add_argument("--jets", choices=["all", "leading"], default="all",
                    help="every AOJ jet of an event (default), or only its leading AOJ jet (both classes)")
    ap.add_argument("--qcd-scarce", choices=["drop-top", "weight"], default="drop-top",
                    help="cells with too few QCD MC jets: drop tops there (default; every weight is 1), or keep the "
                         "tops and give those QCD jets weight > 1")
    ap.add_argument("--plan", action="store_true", help="only print the plan (counts, capping); write nothing but the pass-1 cache")
    a = ap.parse_args(argv)
    veto = None if a.veto in (None, "none") else a.veto
    if not a.plan:
        for name in SPLITS:
            if glob.glob(os.path.join(a.outdir, name, "*.h5")):
                sys.exit("%s/%s already has shards: use a new --outdir" % (a.outdir, name))
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()

    B = BINNINGS[a.match]
    rows, sig_hist, sig_info, sig = signal_rows(a.store, veto, B, a.jets)
    for name in SPLITS:
        print("signal %-12s %-17s %8d jets, %d tag-and-probe events dropped" % (
            name, rows[name][1], len(rows[name][2]), sig_info["%s_%s_dropped_eval_events" % (name, rows[name][1])]))

    files = sorted(glob.glob(os.path.join(a.store, "h5", "out_QCD_Pt*_sub*.h5")))
    labels = sorted(set(label_of(p) for p in files), key=lambda l: SAMPLES.index(l))
    cache = os.path.join(a.outdir, "qcd_pass1.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        assert list(z["files"]) == [os.path.basename(p) for p in files], "pass-1 cache is for other files: delete it"
        assert "pools" in z and str(z["pools"]) == json.dumps(QCD_POOL), "pass-1 cache is for other QCD pools: delete it"
        assert "binning" in z and str(z["binning"]) == json.dumps({k: v.tolist() for k, v in B.items()}), \
            "pass-1 cache is for another binning: delete it"
        assert (str(z["jets"]) if "jets" in z else "all") == a.jets, "pass-1 cache is for another --jets: delete it"
        sgw, njet, H = z["sgw"], z["njet"], z["H"]
    else:
        print("QCD pass 1: %d files" % len(files))
        with Pool(a.workers) as pool:
            res = sorted(pool.imap_unordered(qcd_hist, [(p, B, a.jets) for p in files]), key=lambda r: r[0])
        sgw = np.array([r[1] for r in res])
        njet = np.array([r[2] for r in res])
        H = np.stack([r[3] for r in res])
        np.savez(cache, files=np.array([os.path.basename(p) for p in files]), sgw=sgw, njet=njet, H=H,
                 pools=np.array(json.dumps(QCD_POOL)), binning=np.array(json.dumps({k: v.tolist() for k, v in B.items()})),
                 jets=np.array(a.jets))
        print("  pass 1 done (%.0f s)" % (time.time() - t0))
    lab = np.array([label_of(p) for p in files])
    norm = {l: QCD_XS[l] / sgw[lab == l].sum() for l in labels}
    fnorm = np.array([norm[l] for l in lab])
    Hq = np.einsum("f,fsc->sc", fnorm, H[:, :, 0])              # physical sum w per pool and cell
    target = {name: (a.qcd_ratio if name in TRAINED else a.qcd_test_ratio) * len(rows[name][2]) for name in QCD_SPLITS}
    scale, plan, keepc = [], {}, {}
    for s, name in enumerate(QCD_SPLITS):
        f_top = sig_hist[name] / sig_hist[name].sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.where(Hq[s] > 0, target[name] * f_top / Hq[s], 0.0)
        empty = (f_top > 0) & (Hq[s] <= 0)
        # per sample: MC jets and their mean weight per cell (genWeight ~ constant within a pythia pT-hat bin),
        # hence the largest QCD keep probability of each cell
        nj = {l: H[lab == l][:, s, 1].sum(0) for l in labels}
        wj = {l: np.where(nj[l] > 0, H[lab == l][:, s, 0].sum(0) / np.maximum(nj[l], 1), 0.0) * norm[l] for l in labels}
        pmax = np.max([np.where(nj[l] > 0, sc * wj[l], 0.0) for l in labels], axis=0)
        kc = np.ones(ncell(B))
        if a.qcd_scarce == "drop-top":      # short cells: keep a fraction 1 / pmax of their tops, and of their QCD
            kc = np.where(pmax > 1, 1.0 / np.maximum(pmax, 1e-300), 1.0)
            kc[empty] = 0.0
        keepc[name] = kc
        sc = sc * kc
        scale.append(sc)
        per = {l: float((nj[l] * np.minimum(sc * wj[l], 1.0)).sum()) for l in labels}
        capped_w = float(sum((nj[l] * np.maximum(sc * wj[l] - 1.0, 0.0)).sum() for l in labels))
        tops = float((sig_hist[name] * kc).sum())
        short = (pmax > 1) & (f_top > 0)
        plan[name] = dict(tops=int(sig_hist[name].sum()), tops_expected_kept=tops, short_cells=int(short.sum()),
                          tops_in_short_cells=float(sig_hist[name][short].sum()),
                          target=target[name] * tops / sig_hist[name].sum(), expected_kept=per,
                          expected_kept_total=sum(per.values()), capped_shortfall=capped_w,
                          top_fraction_in_cells_without_qcd=float(f_top[empty].sum()))
        print("QCD %-5s tops %8d -> %9.0f kept (%d short cells)  QCD target %9.0f  expected kept %9.0f  "
              "shortfall (carried by weight > 1) %7.0f  top in cells without QCD %.2e" % (
                  name, sig_hist[name].sum(), tops, short.sum(), plan[name]["target"], sum(per.values()), capped_w,
                  f_top[empty].sum()))
        print("    by sample: " + ", ".join("%s %.0f" % (l, v) for l, v in per.items()))
    # tops of the short cells: per-jet keep decision (test_semilep has no QCD of its own and is never sampled)
    vt = {src: v01(d["ev"]) for src, d in sig.items()}
    for name in SPLITS:
        path, src, r = rows[name]
        kp = keepc[name][cell(sig[src]["jk"][r], B)] if name in keepc else np.ones(len(r))
        k = vt[src][r] < kp
        rows[name] = (path, src, r[k], kp[k])
        sig_info["%s_tops_dropped" % name] = int((~k).sum())
        print("signal %-12s %8d jets kept, %d dropped in short cells" % (name, k.sum(), (~k).sum()))
    if a.plan:
        return 0

    attrs = dict(doc=__doc__, lumi_fb=a.lumi, qcd_xsec_pb=json.dumps(QCD_XS), tt_xsec_pb=json.dumps(TT_XS),
                 qcd_pools=json.dumps(QCD_POOL), signal_pools=json.dumps(SIG_POOL), signal_source=json.dumps(SIG_SRC),
                 veto=str(veto), qcd_ratio=a.qcd_ratio, qcd_test_ratio=a.qcd_test_ratio, match=a.match, jets=a.jets,
                 qcd_scarce=a.qcd_scarce, binning=json.dumps({k: v.tolist() for k, v in B.items()}))
    writers = {(name, c): Writer(os.path.join(a.outdir, name), c, a.jets_per_shard, dict(attrs, split=name, cls=c))
               for name in SPLITS for c in ("top", "qcd") if c == "top" or name in QCD_SPLITS}
    for name in SPLITS:
        path, src, r, kp = rows[name]
        write_signal(writers[(name, "top")], path, src, r, kp, SIG_POOL[name][1] - SIG_POOL[name][0], a.lumi, sig_info,
                     sig[src])
        print("  wrote %d %s top jets (%.0f s)" % (writers[(name, "top")].total, name, time.time() - t0), flush=True)
    kept = {(name, l): 0 for name in QCD_SPLITS for l in labels}
    capped = {name: 0 for name in QCD_SPLITS}
    pstat = dict(p_above_1=0, p_max=0.0)
    tasks = [(p, norm[label_of(p)], scale, B, a.jets, a.qcd_scarce) for p in files]
    with Pool(a.workers) as pool:
        for k, (p, out, st) in enumerate(pool.imap(qcd_select, tasks, chunksize=1)):
            pstat["p_above_1"] += st["p_above_1"]
            pstat["p_max"] = max(pstat["p_max"], st["p_max"])
            for name, b in out.items():
                b["w_phys"] = (b.pop("w_phys_unit") * a.lumi * 1e3).astype(np.float32)
                writers[(name, "qcd")].write(b)
                kept[(name, label_of(p))] += len(b["label"])
                capped[name] += int((b["weight"] > 1).sum())
            if k % 100 == 0:
                print("  QCD %d/%d files, kept %s (%.0f s)" % (k, len(files), {n: writers[(n, "qcd")].total for n in QCD_SPLITS},
                                                                time.time() - t0), flush=True)
    for w in writers.values():
        w.close()
    summary = dict(match=a.match, jets=a.jets, qcd_scarce=a.qcd_scarce, veto=veto,
                   binning={k: v.tolist() for k, v in B.items()}, plan=plan, signal=sig_info,
                   written={"%s/%s" % k: dict(jets=w.total, shards=w.idx + 1) for k, w in writers.items()},
                   qcd_kept_by_sample={name: {l: kept[(name, l)] for l in labels} for name in QCD_SPLITS},
                   qcd_jets_with_weight_above_1=capped, qcd_keep_probability=pstat,
                   qcd_sumGenWeight={l: float(sgw[lab == l].sum()) for l in labels},
                   qcd_files={l: int((lab == l).sum()) for l in labels}, qcd_xsec_pb=QCD_XS, tt_xsec_pb=TT_XS,
                   lumi_fb=a.lumi, seconds=time.time() - t0)
    with open(os.path.join(a.outdir, "summary.json"), "w") as fo:
        json.dump(summary, fo, indent=1)
    print(json.dumps(summary["written"], indent=1))
    print("done (%.0f s)" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
