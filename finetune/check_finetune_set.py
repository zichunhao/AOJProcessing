#!/usr/bin/env python3
"""Checks of a make_finetune_set.py output.

    bash ttbar_semilep/py3env.sh finetune/check_finetune_set.py OUTDIR [--store STORE]

- schema: datasets, shapes, column attributes; PFCands columns = the AOJ skimmed shards'; padding rows all zero
- counts per split / class / sample, and against summary.json
- labels: top jets have top_match == 2 and come from TTToHadronic (train/val/test) or TTToSemiLeptonic (test_semilep)
- splits: no event in two splits (per sample); no train/val event in the tag-and-probe veto list (if one was used)
- weights: all 1 for --qcd-scarce drop-top; jet_rank / n_jets of every top against merged_<sample>.h5, and of the
  random QCD jets below against their per-job file
- kinematics: weighted (pT, |eta|) shape of QCD vs top per split, and sum(w_phys) vs the full-sample yields
- PFCands of a few random jets recomputed from the source files
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import make_finetune_set as mf  # noqa: E402
from H5_skim_split import skim_pf  # noqa: E402

SMALL = ["jet_kinematics", "event_info", "gen_info", "label", "sample", "weight", "w_phys"]
OPT = ["jet_rank", "n_jets"]                     # sets from 20261021 on


def load(outdir, split, cls):
    files = sorted(glob.glob(os.path.join(outdir, split, "%s_*.h5" % cls)))
    d = {k: [] for k in SMALL + OPT}
    for p in files:
        with h5py.File(p, "r") as f:
            n = f["label"].shape[0]
            for k in SMALL + [k for k in OPT if k in f]:
                assert f[k].shape[0] == n, (p, k)
                d[k].append(f[k][:])
            assert f["PFCands"].shape == (n, 150, 16) and f["jet_tagging"].shape == (n, 13), p
            assert f["PFCands"].attrs["columns"] == ",".join(mf.PF_COLS), p
            assert f["sample"].attrs["names"] == ",".join(mf.SAMPLES), p
            pf = f["PFCands"][:min(n, 2000)]
            pad = pf[..., 0] == 0
            assert not np.abs(pf[pad]).any(), "%s: non-zero padding rows" % p
    return files, {k: np.concatenate(v) if v else np.zeros(0) for k, v in d.items() if v or k in SMALL}


def rank_key(ev, jk):
    """64-bit hash of (event, jet pT, eta, phi): finds a jet in its source file (NanoAOD rounds pT, so the two jets
    of an event can have the same pT)."""
    b = np.ascontiguousarray(jk[:, :3], dtype=np.float32).view(np.uint32).astype(np.uint64)
    with np.errstate(over="ignore"):
        return mf.splitmix(mf.splitmix(mf.evkey(ev) ^ b[:, 0]) ^ ((b[:, 1] << np.uint64(32)) | b[:, 2]))


def source_ranks(store, src):
    """Sorted jet keys, pT rank and number of AOJ jets in the event, of every fully merged top of merged_<src>.h5."""
    with h5py.File(os.path.join(store, "merged_%s.h5" % src), "r") as f:
        ev, jk = f["event_info"][:], f["jet_kinematics"][:]
        t = f["gen_info"][:, mf.GEN_COLS.index("top_match")] == 2
    rank, nj = mf.event_rank(ev, jk[:, 0])
    k = rank_key(ev[t], jk[t])
    o = np.argsort(k)
    return k[o], rank[t][o], nj[t][o]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--store", default="/ceph/cms/store/user/zichun/parcel/AOJ")
    a = ap.parse_args(argv)
    summ = json.load(open(os.path.join(a.outdir, "summary.json")))
    ok = True
    D, SRC_RANK = {}, {}
    for s in mf.SPLITS:
        for c in ("top", "qcd"):
            if c == "qcd" and s not in mf.QCD_SPLITS:
                D[(s, c)] = D[("test", c)]           # test_semilep's background is test/qcd
                continue
            files, d = load(a.outdir, s, c)
            D[(s, c)] = d
            n = len(d["label"])
            exp = summ["written"]["%s/%s" % (s, c)]["jets"]
            print("%-12s %-3s %8d jets in %d shards (summary %d)" % (s, c, n, len(files), exp))
            ok &= n == exp
            ok &= bool((d["label"] == (1 if c == "top" else 0)).all())
            names = np.array(mf.SAMPLES)[d["sample"].astype(int)]
            u, k = np.unique(names, return_counts=True)
            print("      samples: " + ", ".join("%s %d" % x for x in zip(u, k)))
            if c == "top":
                tm = d["gen_info"][:, mf.GEN_COLS.index("top_match")]
                ok &= bool((tm == 2).all()) and set(u) == {mf.SIG_SRC[s]}
            else:
                ok &= all(x.startswith("QCD") for x in u)
            if summ.get("qcd_scarce") == "drop-top":
                ok &= bool((d["weight"] == 1).all())
            if "jet_rank" in d:
                r, nj = d["jet_rank"].astype(int), d["n_jets"].astype(int)
                fr = np.bincount(np.minimum(r, 2), weights=d["weight"], minlength=3) / d["weight"].sum()
                print("      jet_rank leading / 2nd / 3rd+: %s; n_jets 1 / 2 / 3+: %s" % (
                    " / ".join("%.3f" % x for x in fr),
                    " / ".join("%.3f" % x for x in np.bincount(np.minimum(nj, 3), minlength=4)[1:] / len(nj))))
                ok &= bool(((r >= 0) & (r < nj)).all())
                if summ.get("jets") == "leading":
                    ok &= bool((r == 0).all())
                if c == "top":
                    if mf.SIG_SRC[s] not in SRC_RANK:
                        SRC_RANK[mf.SIG_SRC[s]] = source_ranks(a.store, mf.SIG_SRC[s])
                    K, R, NJ = SRC_RANK[mf.SIG_SRC[s]]
                    k = rank_key(d["event_info"], d["jet_kinematics"])
                    i = np.clip(np.searchsorted(K, k), 0, len(K) - 1)
                    good = (K[i] == k) & (R[i] == r) & (NJ[i] == nj)
                    print("      jet_rank, n_jets equal to merged_%s.h5: %d of %d" % (mf.SIG_SRC[s], good.sum(), len(k)))
                    ok &= bool(good.all())
    # splits are disjoint in events (per sample) and train/val are free of the tag-and-probe events
    with h5py.File(os.path.join(a.store, "eval_events_TT.h5"), "r") as fv:
        veto = mf.evkey(fv["TTToHadronic"][:])
    for c in ("top", "qcd"):
        keys = {}
        sp = mf.SPLITS if c == "top" else mf.QCD_SPLITS
        for s in sp:
            d = D[(s, c)]
            keys[s] = set(zip(d["sample"].tolist(), mf.evkey(d["event_info"]).tolist()))
        for i, s1 in enumerate(sp):
            for s2 in sp[i + 1:]:
                inter = len(keys[s1] & keys[s2])
                print("%s: events in both %s and %s: %d" % (c, s1, s2, inter))
                ok &= inter == 0
    for s in ("train", "val"):
        bad = int(np.isin(mf.evkey(D[(s, "top")]["event_info"]), veto).sum())
        print("top %s: jets from tag-and-probe events: %d%s" % (s, bad, "" if summ.get("veto") else " (no veto requested)"))
        ok &= bad == 0 or not summ.get("veto")
    # kinematic match: QCD (weight) vs top, per split, in the make_finetune_set cells and in coarse pT bins
    for s in mf.SPLITS:
        t, q = D[(s, "top")], D[(s, "qcd")]
        B = {k: np.array(v) for k, v in summ.get("binning", {}).items()} or mf.BINNINGS["pt_eta"]
        nc = mf.ncell(B)
        ht = np.bincount(mf.cell(t["jet_kinematics"], B), minlength=nc).astype(float)
        hq = np.bincount(mf.cell(q["jet_kinematics"], B), weights=q["weight"], minlength=nc)
        ht, hq = ht / ht.sum(), hq / hq.sum()
        frac = lambda d, col, edges: np.round(np.histogram(d["jet_kinematics"][:, col], edges,
                                                           weights=d["weight"])[0] / d["weight"].sum(), 4).tolist()
        pt_e, m_e = [300, 400, 500, 600, 800, 1000, 14000], [-1e9, 20, 60, 100, 140, 160, 180, 200, 260, 1e9]
        print("%-5s max cell |diff| %.2e (%d cells); weight > 1: %d QCD jets (max %.2f); QCD/top = %.3f" % (
            s, np.abs(ht - hq).max(), nc, int((q["weight"] > 1).sum()), q["weight"].max(), q["weight"].sum() / len(t["label"])))
        print("      pT fractions  top %s\n                    qcd %s" % (frac(t, 0, pt_e), frac(q, 0, pt_e)))
        print("      mSD fractions top %s\n                    qcd %s   (mSD edges %s)" % (frac(t, 3, m_e), frac(q, 3, m_e),
                                                                                  m_e[1:-1]))
        print("      sum w_phys: top %.4g, qcd %.4g (jets in %.2f fb^-1)" % (t["w_phys"].sum(), q["w_phys"].sum(),
                                                                         summ["lumi_fb"]))
    # PFCands of random jets vs the source (signal: topmatched files; QCD: per-job files)
    rng = np.random.RandomState(1)
    for s in mf.SPLITS:
        for c in ("top", "qcd"):
            if c == "qcd" and s not in mf.QCD_SPLITS:
                continue
            files = sorted(glob.glob(os.path.join(a.outdir, s, "%s_*.h5" % c)))
            with h5py.File(files[0], "r") as f:
                n = f["label"].shape[0]
                for r in sorted(rng.choice(n, 3, replace=False)):
                    ev, jk, pf, smp = f["event_info"][r], f["jet_kinematics"][r], f["PFCands"][r], mf.SAMPLES[f["sample"][r]]
                    src = (os.path.join(a.store, "merged_%s_topmatched.h5" % smp) if c == "top"
                           else sorted(glob.glob(os.path.join(a.store, "h5", "out_%s_sub*.h5" % smp))))
                    found = False
                    for p in ([src] if c == "top" else src):
                        with h5py.File(p, "r") as g:
                            e = g["event_info"][:]
                            m = np.where((e[:, 1] == ev[1]) & (e[:, 2] == ev[2]))[0]
                            for i in m:
                                if np.allclose(g["jet_kinematics"][i], jk):
                                    ref = skim_pf(g["PFCands"][i:i + 1], g["jet_kinematics"][i:i + 1])[0]
                                    found = np.array_equal(ref, pf)
                                    if c == "qcd" and "jet_rank" in f:   # pT ties: the earlier row ranks first
                                        ptm = g["jet_kinematics"][:][m, 0]
                                        rk = int((ptm > jk[0]).sum() + ((ptm == jk[0]) & (m < i)).sum())
                                        found &= rk == f["jet_rank"][r] and len(m) == f["n_jets"][r]
                                    break
                        if found or len(m):
                            break
                    print("%-5s %-3s row %7d (%s, lumi %d, event %d): PFCands (QCD: and jet_rank) as in the source: %s" % (
                        s, c, r, smp, ev[1], ev[2], found))
                    ok &= found
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
