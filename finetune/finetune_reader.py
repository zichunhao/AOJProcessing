#!/usr/bin/env python3
"""Standalone reader for the fine-tuning sets (finetune_top_<date>_<setting>/): numpy + h5py only.

    from finetune_reader import FinetuneSet
    fs = FinetuneSet("/rinovol-central/AOJ/finetune_top_20261025_lnpt-msd")
    for epoch in range(n_epochs):
        for batch in fs.batches("train", epoch, batch_size=512, seed=0):
            x, mask, y = batch["PFCands"], batch["mask"], batch["label"]     # (B,150,16), (B,150), (B,)
            ...
    for batch in fs.batches("test", epoch=0, shuffle=False, fields=("PFCands", "label", "jet_kinematics")):
        ...

Each epoch uses every top jet and the QCD jets with epoch_u01(sample, event_info, jet_rank, epoch) < w_sample
(train: a new draw per epoch from a pool of ~5 epochs of QCD; val / test: w_sample = 1, all jets, every epoch).
The draw is deterministic: every model gets the same jets in each epoch. test_semilep has no QCD of its own;
its background is test/qcd (batches("test_semilep") reads both).

Batches mix top and QCD: blocks of `block` consecutive rows are read from all shards of the split, the rows
kept in the epoch are pooled in a buffer of ~`buffer` jets, shuffled, and cut into batches (block order random
within each class, the classes interleaved so that every buffer has the epoch's class mix; in-buffer shuffling;
good enough for jets in source order).

Run this file to self-test: python finetune_reader.py SETDIR
"""
import glob
import os
import sys

import numpy as np
import h5py


def _splitmix(x):
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def _to01(x):
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _evkey(event_info):
    return (event_info[:, 1].astype(np.uint64) << np.uint64(32)) | event_info[:, 2].astype(np.uint64)


def u01(event_info):
    """Per-event number in [0, 1): the split (train < 0.7 <= val < 0.8 <= test); use it for nested label-fraction
    subsets (e.g. train jets with u01 < 0.07 = 10 % of train), which then hold for both classes."""
    return _to01(_splitmix(_evkey(event_info)))


def epoch_u01(sample, event_info, jet_rank, epoch):
    """Per-jet number of one epoch: the jet is used in `epoch` iff this is below its w_sample."""
    with np.errstate(over="ignore"):
        k = _splitmix(_evkey(event_info) ^ (np.asarray(sample).astype(np.uint64) << np.uint64(56)))
        k = _splitmix(k ^ ((np.asarray(jet_rank).astype(np.uint64) + np.uint64(1)) * np.uint64(0x9FB21C651E98DF25)))
        return _to01(_splitmix(k ^ ((np.uint64(epoch) + np.uint64(1)) * np.uint64(0xC2B2AE3D27D4EB4F))))


class FinetuneSet(object):
    def __init__(self, setdir):
        self.setdir = setdir
        self._small = {}

    def shards(self, split):
        """(path, class) of the shards a split reads; test_semilep reads its tops and test/qcd."""
        out = []
        for cls in ("top", "qcd"):
            d = "test" if (split == "test_semilep" and cls == "qcd") else split
            out += [(p, cls) for p in sorted(glob.glob(os.path.join(self.setdir, d, "%s_*.h5" % cls)))]
        return out

    def epoch_mask(self, path, epoch, subset=None):
        """Rows of one shard used in `epoch` (bool array); subset = (lo, hi) keeps only events with lo <= u01 < hi."""
        if path not in self._small:                  # small per-jet arrays, cached (~30 bytes per jet)
            with h5py.File(path, "r") as f:
                self._small[path] = (f["sample"][:], f["event_info"][:], f["jet_rank"][:], f["w_sample"][:])
        sample, ev, rank, ws = self._small[path]
        m = epoch_u01(sample, ev, rank, epoch) < ws
        if subset is not None:
            u = u01(ev)
            m &= (u >= subset[0]) & (u < subset[1])
        return m

    def count(self, split, epoch, subset=None):
        """Jets per class in one epoch."""
        n = {"top": 0, "qcd": 0}
        for p, cls in self.shards(split):
            n[cls] += int(self.epoch_mask(p, epoch, subset).sum())
        return n

    def batches(self, split, epoch, batch_size=512, fields=("PFCands", "label"), shuffle=True, seed=0,
                block=8192, buffer=262144, subset=None, drop_last=False):
        """Yield dicts of numpy arrays (fields + 'mask' = PFCands[..., 0] > 0 when PFCands is read)."""
        rng = np.random.default_rng([seed, epoch])
        blocks = {"top": [], "qcd": []}
        for p, cls in self.shards(split):
            m = self.epoch_mask(p, epoch, subset)
            for b0 in range(0, len(m), block):
                rows = np.flatnonzero(m[b0:b0 + block]) + b0
                if len(rows):
                    blocks[cls].append((p, b0, min(b0 + block, len(m)), rows))
        if shuffle:
            # shuffle within each class, then interleave by kept rows: a top block keeps all its rows, a train QCD
            # block ~1/5, so a plain block shuffle would let the class mix of a buffer swing by ~+-10 %
            per = [[q[i] for i in rng.permutation(len(q))] for q in (blocks["top"], blocks["qcd"])]
            tot = [max(1, sum(len(b[3]) for b in q)) for q in per]
            pos, done, out = [0, 0], [0, 0], []
            while pos[0] < len(per[0]) or pos[1] < len(per[1]):
                c = min((done[c] / float(tot[c]), c) for c in (0, 1) if pos[c] < len(per[c]))[1]
                out.append(per[c][pos[c]])
                done[c] += len(per[c][pos[c]][3])
                pos[c] += 1
            blocks = out
        else:
            blocks = blocks["top"] + blocks["qcd"]
        buf, nbuf = [], 0
        for i, (p, b0, b1, rows) in enumerate(blocks):
            with h5py.File(p, "r") as f:
                buf.append({k: f[k][b0:b1][rows - b0] for k in fields})
            nbuf += len(rows)
            if nbuf >= buffer or i == len(blocks) - 1:
                cat = {k: np.concatenate([b[k] for b in buf]) for k in fields}
                o = rng.permutation(nbuf) if shuffle else np.arange(nbuf)
                stop = nbuf if (i == len(blocks) - 1 and not drop_last) else nbuf - nbuf % batch_size
                for s in range(0, stop, batch_size):
                    sel = o[s:s + batch_size]
                    if drop_last and len(sel) < batch_size:
                        break
                    out = {k: v[sel] for k, v in cat.items()}
                    if "PFCands" in out:
                        out["mask"] = out["PFCands"][..., 0] > 0
                    yield out
                rest = o[stop:]
                buf = [{k: v[rest] for k, v in cat.items()}] if len(rest) else []
                nbuf = len(rest)


if __name__ == "__main__":
    fs = FinetuneSet(sys.argv[1])
    for split in ("train", "val", "test", "test_semilep"):
        print(split, "epoch 0:", fs.count(split, 0), "epoch 1:", fs.count(split, 1))
    n, ytot = 0, 0
    for b in fs.batches("val", 0, batch_size=1024, fields=("label",)):
        n += len(b["label"])
        ytot += int(b["label"].sum())
    print("val epoch 0 via batches: %d jets, %d tops" % (n, ytot))
