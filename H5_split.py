"""Split a jet HDF5 (the AOJ merged.h5) into train/val/test by ratio.

Per-jet random assignment, streamed in batches (memory-light), reproducible via
--seed. Each output keeps the same datasets/dtypes/compression as the input.

Note: jets keep their source order within each split (the split is a random
*subset*, not a shuffle) -- shuffle the train set at load time as usual.

Usage:
  python H5_split.py -i merged.h5 -o /path/out --ratios 100,5,10 --seed 42
  -> writes <out>/train.h5, <out>/val.h5, <out>/test.h5
"""
from __future__ import print_function
import h5py, numpy as np, argparse, os, sys


def append_h5(dset, data):
    n0 = dset.shape[0]
    dset.resize(n0 + data.shape[0], axis=0)
    dset[n0:] = data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--outdir", default=".")
    ap.add_argument("--ratios", default="100,5,10", help="train,val,test (any positive numbers)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=50000)
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    assert len(ratios) == 3, "need exactly 3 ratios: train,val,test"
    edges = np.cumsum(ratios) / sum(ratios)          # [f_train, f_train+f_val, 1.0]
    names = ["train", "val", "test"]

    fin = h5py.File(args.input, "r")
    keys = list(fin.keys())
    N = fin[keys[0]].shape[0]
    fr = (edges[0], edges[1] - edges[0], 1.0 - edges[1])
    print("input %s : %d rows, keys=%s" % (args.input, N, keys))
    print("ratios=%s -> fractions train/val/test = %.4f/%.4f/%.4f  seed=%d" %
          (ratios, fr[0], fr[1], fr[2], args.seed))

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    outs = {}
    for nm in names:
        f = h5py.File(os.path.join(args.outdir, nm + ".h5"), "w")
        for k in keys:
            ds = fin[k]
            f.create_dataset(k, shape=(0,) + ds.shape[1:], maxshape=(None,) + ds.shape[1:],
                             dtype=ds.dtype, chunks=True, compression=ds.compression)
        outs[nm] = f

    rng = np.random.RandomState(args.seed)
    counts = [0, 0, 0]
    for bi, start in enumerate(range(0, N, args.batch)):
        end = min(start + args.batch, N)
        assign = np.digitize(rng.random_sample(end - start), edges[:-1])   # 0/1/2
        batch = {k: fin[k][start:end] for k in keys}
        for i, nm in enumerate(names):
            m = assign == i
            c = int(m.sum())
            if c == 0:
                continue
            for k in keys:
                append_h5(outs[nm][k], batch[k][m])
            counts[i] += c
        if bi % 50 == 0:
            print("  %d / %d (%.1f%%)  train=%d val=%d test=%d" %
                  (end, N, 100.0 * end / N, counts[0], counts[1], counts[2]))
            sys.stdout.flush()

    for f in outs.values():
        f.close()
    fin.close()
    tot = float(sum(counts))
    print("DONE: total=%d  train=%d (%.2f%%)  val=%d (%.2f%%)  test=%d (%.2f%%)" %
          (int(tot), counts[0], 100 * counts[0] / tot, counts[1], 100 * counts[1] / tot,
           counts[2], 100 * counts[2] / tot))


if __name__ == "__main__":
    main()
