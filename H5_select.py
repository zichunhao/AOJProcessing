"""Copy an AOJ jet HDF5 file keeping only the gen-matched jets, e.g. the fully merged tops of a ttbar sample.

    python H5_select.py merged_TTToHadronic.h5 merged_TTToHadronic_top.h5 --top_match 2

--top_match 2 keeps the jets whose gen_info top_match is 2 (b, q and q' of one hadronic top within
dR < 0.8): exactly the jets `H5_maker.py --gen_match 6` writes, from a file made with --top_label.
--top_match 1 keeps the W-matched ones. Every dataset with one row per jet is filtered in chunks;
dataset and root attributes are copied. nEventsProcessed and sumGenWeight still describe the whole
processed sample, so xsec x lumi / sumGenWeight stays the right normalization.
Runs under python 2 (CMSSW, like H5_merge.py) and python 3.
"""
from __future__ import print_function

import argparse
import os
import sys

import h5py
import numpy as np

CHUNK = 20000          # rows per read: ~130 MB of PFCands (150 x 11 float32 per jet)


def column_names(ds):
    return [c.decode() if isinstance(c, bytes) else str(c) for c in np.atleast_1d(ds.attrs["columns"])]


def main(argv=None):
    p = argparse.ArgumentParser(description="keep only gen-matched jets of an AOJ jet HDF5 file")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--top_match", type=int, required=True, choices=[1, 2],
                   help="keep jets with gen_info top_match == this (2 = fully merged top, 1 = W)")
    p.add_argument("--force", action="store_true", help="overwrite an existing output")
    a = p.parse_args(argv)
    if os.path.exists(a.output) and not a.force:
        sys.exit("H5_select: %s exists (use --force to overwrite)" % a.output)

    with h5py.File(a.input, "r") as fin:
        if "gen_info" not in fin:
            sys.exit("H5_select: %s has no gen_info (data, or MC made without gen info)" % a.input)
        gi = fin["gen_info"]
        icol = column_names(gi).index("top_match")
        n = gi.shape[0]
        rows = [k for k in fin if isinstance(fin[k], h5py.Dataset) and fin[k].shape[:1] == (n,)]
        other = [k for k in fin if k not in rows]
        if other:
            sys.exit("H5_select: datasets without one row per jet: %s" % other)
        tm = gi[:, icol]
        if (tm < 0).any():
            sys.exit("H5_select: top_match not computed for %d jets (file made without --top_label?)"
                     % int((tm < 0).sum()))

        with h5py.File(a.output, "w") as fout:
            for k in rows:
                d = fin[k]
                fout.create_dataset(k, shape=(0,) + d.shape[1:], maxshape=(None,) + d.shape[1:], dtype=d.dtype,
                                    chunks=True, compression=d.compression, compression_opts=d.compression_opts)
                for attr in d.attrs:
                    fout[k].attrs[attr] = d.attrs[attr]
            kept = 0
            for start in range(0, n, CHUNK):
                m = tm[start:start + CHUNK] == a.top_match
                nk = int(m.sum())
                if nk == 0:
                    continue
                for k in rows:
                    out = fout[k]
                    out.resize(kept + nk, axis=0)
                    out[kept:kept + nk] = fin[k][start:start + CHUNK][m]
                kept += nk
            for attr in fin.attrs:
                fout.attrs[attr] = fin.attrs[attr]
            fout.attrs["jet_selection"] = "gen_info top_match == %d" % a.top_match
    print("H5_select: kept %d of %d jets (top_match == %d) -> %s" % (kept, n, a.top_match, a.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
