#!/usr/bin/env python3
"""Standalone reader for the SF inputs (sf_inputs_<date>_<selection>/, one <label>.h5 per sample): numpy + h5py only.

    from sf_reader import SFSet
    s = SFSet("/ceph/cms/store/user/zichun/parcel/AOJ/sf_inputs_20261120_iso30_met20")
    for b in s.batches(sel=("iso30_met20", "btag"), batch_size=4096):
        x, mask = b["PFCands"], b["mask"]                 # (B,150,16) float32, (B,150) bool: the model input
        score = model(x, mask)
        s.add_scores("rino", b, score)                   # kept per (sample, row), aligned with the files
    s.save_scores("rino", OUTDIR)                        # OUTDIR/rino/<label>.npy, NaN for rows not scored

Each batch holds the rows of one sample (no shuffling: evaluation only) that pass every flag in `sel` (columns of
the `sel` dataset: an, an_anyak8, iso30, iso30_met20, skim_iso, dphi, btag, tau32), with
  PFCands, mask, label (1 merged top, 0 other MC, -1 data), weight (nominal), channel (1 mu, 2 e), pt, eta, msd,
  category (truth: 0 other, 1 b+q, 2 W, 3 t; -1 data), sample (the label), row (index in <label>.h5)
and, on request (`extra`), any dataset of the file ("weights", "ttbar_tag", "jet_tagging", "ak4", ...).
Rows are read in blocks of `block` consecutive rows, so a selective `sel` reads only what it needs.

Run this file to self-test: python sf_reader.py SETDIR
"""
import glob
import os
import sys

import numpy as np
import h5py


class SFSet(object):
    def __init__(self, setdir):
        self.dir = setdir
        self.files = dict((os.path.basename(p)[:-3], p) for p in sorted(glob.glob(os.path.join(setdir, "*.h5"))))
        self.labels = list(self.files)
        self._scores = {}

    def info(self, label):
        """The file attributes of a sample (group, channels, xsec_pb, sum_genweight, lumi_fb, files, ...)."""
        with h5py.File(self.files[label], "r") as f:
            return dict(f.attrs.items())

    def mask(self, label, sel):
        """Boolean mask over the rows of <label>.h5: every flag in sel set."""
        with h5py.File(self.files[label], "r") as f:
            if "sel" not in f:
                return np.zeros(0, bool)
            cols = f["sel"].attrs["columns"].split(",")
            fl = f["sel"][:]
        m = np.ones(len(fl), bool)
        for s in sel:
            m &= fl[:, cols.index(s)] > 0
        return m

    def count(self, sel, labels=None):
        """Selected rows per sample."""
        return dict((l, int(self.mask(l, sel).sum())) for l in (labels or self.labels))

    def batches(self, sel=("iso30_met20", "btag"), batch_size=4096, labels=None, extra=(), block=8192):
        for label in labels or self.labels:
            m = self.mask(label, sel)
            rows = np.flatnonzero(m)
            if not len(rows):
                continue
            with h5py.File(self.files[label], "r") as f:
                tc = f["ttbar_tag"].attrs["columns"].split(",")
                ch_col = tc.index("channel")
                buf = []
                for b0 in range(0, len(m), block):
                    r = rows[(rows >= b0) & (rows < b0 + block)]
                    if not len(r):
                        continue
                    sl = slice(b0, min(b0 + block, len(m)))
                    loc = r - b0
                    jk = f["jet_kinematics"][sl][loc]
                    d = dict(PFCands=f["PFCands"][sl][loc], label=f["label"][sl][loc], weight=f["weight"][sl][loc],
                             channel=f["ttbar_tag"][sl][loc][:, ch_col].astype(np.int8), pt=jk[:, 0], eta=jk[:, 1],
                             msd=jk[:, 3], row=r,
                             category=(f["truth"][sl][loc][:, 0].astype(np.int8) if "truth" in f
                                       else np.full(len(r), -1, np.int8)))
                    for k in extra:
                        d[k] = f[k][sl][loc]
                    buf.append(d)
                    while sum(len(x["row"]) for x in buf) >= batch_size:
                        yield self._emit(label, buf, batch_size)
                while buf and sum(len(x["row"]) for x in buf):
                    yield self._emit(label, buf, batch_size)

    @staticmethod
    def _emit(label, buf, n):
        cat = dict((k, np.concatenate([x[k] for x in buf])) for k in buf[0])
        out = dict((k, v[:n]) for k, v in cat.items())
        rest = dict((k, v[n:]) for k, v in cat.items())
        buf[:] = [rest] if len(rest["row"]) else []
        out["mask"] = out["PFCands"][..., 0] > 0
        out["sample"] = label
        return out

    def add_scores(self, name, batch, score):
        """Keep a model's scores of one batch, by (sample, row)."""
        d = self._scores.setdefault(name, {}).setdefault(batch["sample"], {})
        d.update(zip(batch["row"].tolist(), np.asarray(score, np.float32).reshape(-1).tolist()))

    def save_scores(self, name, outdir):
        """OUTDIR/<name>/<label>.npy: one float32 per row of <label>.h5, NaN where no score was added."""
        os.makedirs(os.path.join(outdir, name), exist_ok=True)
        for label, d in self._scores.get(name, {}).items():
            with h5py.File(self.files[label], "r") as f:
                n = f["weight"].shape[0]
            a = np.full(n, np.nan, np.float32)
            a[list(d.keys())] = list(d.values())
            np.save(os.path.join(outdir, name, label + ".npy"), a)


def _selftest(setdir):
    s = SFSet(setdir)
    sel = ("iso30_met20", "btag")
    cnt = s.count(sel)
    n, w, labels = 0, {"data": 0.0, "mc": 0.0}, set()
    for b in s.batches(sel, batch_size=50000):
        assert b["PFCands"].shape[1:] == (150, 16) and b["mask"].shape == b["PFCands"].shape[:2]
        assert np.array_equal(b["row"], np.sort(b["row"]))
        n += len(b["row"])
        labels.add(b["sample"])
        w["data" if b["label"][0] == -1 else "mc"] += float(b["weight"].astype(np.float64).sum())
        s.add_scores("pn", b, np.zeros(len(b["row"])))
    assert n == sum(cnt.values()), (n, sum(cnt.values()))
    print("%d rows of %s in %d samples; data %d events, MC %.1f (nominal weight)" % (n, " & ".join(sel), len(labels), w["data"], w["mc"]))
    return 0


if __name__ == "__main__":
    sys.exit(_selftest(sys.argv[1]))
