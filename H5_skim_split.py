"""Skim the AOJ merged.h5 to ML features, split train/val/test, and shard -- one pass.

Per-constituent PFCands [Px,Py,Pz,E,d0,d0Err,dz,dzErr,charge,pdgId,puppiWeight]
are transformed to 16 features (padding rows stay all-zero):
  pt, eta, phi, E, deta, dphi, logpt, logptrel, dR,
  pdgId, charge, d0, d0Err, dz, dzErr, puppiWeight
where deta/dphi are relative to the jet axis and logptrel = log(pt/jet_pt).
jet_kinematics (pt,eta,phi,msoftdrop) and jet_tagging (13) are carried as-is;
event_info is dropped. Per-jet random split (reproducible via --seed). Each split
is written as size-capped shards: <outdir>/<split>/<split>_NNNN.h5.

Usage:
  python H5_skim_split.py -i merged.h5 -o /path/skimmed \
      --ratios 100,5,10 --seed 42 --jets-per-shard 2000000
"""
from __future__ import print_function
import h5py, numpy as np, argparse, os, sys

PF_COLS = ["pt", "eta", "phi", "E", "deta", "dphi", "logpt", "logptrel", "dR",
           "pdgId", "charge", "d0", "d0Err", "dz", "dzErr", "puppiWeight"]
JK_COLS = ["pt", "eta", "phi", "msoftdrop"]
JT_COLS = ["nConstituents", "tau1", "tau2", "tau3", "tau4",
           "particleNet_H4qvsQCD", "particleNet_HbbvsQCD", "particleNet_HccvsQCD",
           "particleNet_QCD", "particleNet_TvsQCD", "particleNet_WvsQCD",
           "particleNet_ZvsQCD", "particleNet_mass"]
KEYS = ["PFCands", "jet_kinematics", "jet_tagging"]


def skim_pf(pf, jk):
    """pf (B,150,11) -> (B,150,16) float32; padding (pt==0) rows stay all-zero."""
    Px, Py, Pz, E = pf[..., 0], pf[..., 1], pf[..., 2], pf[..., 3]
    d0, d0Err, dz, dzErr = pf[..., 4], pf[..., 5], pf[..., 6], pf[..., 7]
    charge, pdgId, puppi = pf[..., 8], pf[..., 9], pf[..., 10]
    pt = np.sqrt(Px * Px + Py * Py)
    real = pt > 0
    pts = np.where(real, pt, 1.0)
    jpt, jeta, jphi = jk[:, 0:1], jk[:, 1:2], jk[:, 2:3]
    eta = np.where(real, np.arcsinh(Pz / pts), 0.0)
    phi = np.where(real, np.arctan2(Py, Px), 0.0)
    deta = np.where(real, eta - jeta, 0.0)
    dphi = np.where(real, (phi - jphi + np.pi) % (2 * np.pi) - np.pi, 0.0)
    logpt = np.where(real, np.log(pts), 0.0)
    logptrel = np.where(real, np.log(pts / jpt), 0.0)
    dR = np.where(real, np.sqrt(deta * deta + dphi * dphi), 0.0)
    out = np.stack([pt, eta, phi, E, deta, dphi, logpt, logptrel, dR,
                    pdgId, charge, d0, d0Err, dz, dzErr, puppi], axis=-1)
    return out.astype(np.float32)


class ShardWriter(object):
    """Append rows across keys, rolling to a new shard file every `per` rows."""
    def __init__(self, outdir, prefix, per, specs):
        self.dir, self.prefix, self.per, self.specs = outdir, prefix, per, specs
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        self.idx, self.f, self.n, self.total, self.nshard = -1, None, 0, 0, 0
        self._roll()

    def _roll(self):
        if self.f:
            self.f.close()
        self.idx += 1
        self.n = 0
        self.nshard = self.idx + 1
        self.f = h5py.File(os.path.join(self.dir, "%s_%04d.h5" % (self.prefix, self.idx)), "w")
        for k in KEYS:
            tail, dt, comp, cols = self.specs[k]
            d = self.f.create_dataset(k, shape=(0,) + tail, maxshape=(None,) + tail,
                                      dtype=dt, chunks=True, compression=comp)
            d.attrs["columns"] = ",".join(cols)

    def write(self, batch):
        N = batch[KEYS[0]].shape[0]
        off = 0
        while off < N:
            take = min(self.per - self.n, N - off)
            for k in KEYS:
                d = self.f[k]
                n0 = d.shape[0]
                d.resize(n0 + take, axis=0)
                d[n0:] = batch[k][off:off + take]
            self.n += take
            self.total += take
            off += take
            if self.n >= self.per:
                self._roll()

    def close(self):
        if self.f:
            self.f.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--outdir", default="skimmed")
    ap.add_argument("--ratios", default="100,5,10", help="train,val,test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--jets-per-shard", type=int, default=2000000)
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    edges = np.cumsum(ratios) / sum(ratios)
    names = ["train", "val", "test"]

    fin = h5py.File(args.input, "r")
    N = fin["jet_kinematics"].shape[0]
    print("input %s : %d jets" % (args.input, N))
    print("ratios=%s seed=%d jets/shard=%d -> %s" %
          (ratios, args.seed, args.jets_per_shard, args.outdir))

    specs = {
        "PFCands":        ((150, 16),                  np.float32, fin["PFCands"].compression,        PF_COLS),
        "jet_kinematics": (fin["jet_kinematics"].shape[1:], np.float32, fin["jet_kinematics"].compression, JK_COLS),
        "jet_tagging":    (fin["jet_tagging"].shape[1:],    np.float32, fin["jet_tagging"].compression,    JT_COLS),
    }
    writers = {nm: ShardWriter(os.path.join(args.outdir, nm), nm, args.jets_per_shard, specs)
               for nm in names}

    rng = np.random.RandomState(args.seed)
    for bi, start in enumerate(range(0, N, args.batch)):
        end = min(start + args.batch, N)
        assign = np.digitize(rng.random_sample(end - start), edges[:-1])
        pf = skim_pf(fin["PFCands"][start:end], fin["jet_kinematics"][start:end])
        jk = fin["jet_kinematics"][start:end]
        jt = fin["jet_tagging"][start:end]
        for i, nm in enumerate(names):
            m = assign == i
            if not m.any():
                continue
            writers[nm].write({"PFCands": pf[m], "jet_kinematics": jk[m], "jet_tagging": jt[m]})
        if bi % 50 == 0:
            print("  %d/%d (%.1f%%)  train=%d(%dsh) val=%d(%dsh) test=%d(%dsh)" %
                  (end, N, 100.0 * end / N,
                   writers["train"].total, writers["train"].nshard,
                   writers["val"].total, writers["val"].nshard,
                   writers["test"].total, writers["test"].nshard))
            sys.stdout.flush()

    for w in writers.values():
        w.close()
    fin.close()
    tot = sum(writers[nm].total for nm in names)
    print("DONE: total=%d" % tot)
    for nm in names:
        print("  %-5s %d jets in %d shard(s) (%.2f%%)" %
              (nm, writers[nm].total, writers[nm].nshard, 100.0 * writers[nm].total / tot))


if __name__ == "__main__":
    main()
