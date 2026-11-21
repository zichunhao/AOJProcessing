#!/usr/bin/env python3
"""Compare ttbar_maker.py outputs of format 4 (L1 pre-firing weights) with format-3 outputs of the same inputs.

    bash ttbar_semilep/py3env.sh ttbar_semilep/prefiring/check_prefiring.py V4_FILE_OR_DIR V3_FILE_OR_DIR [--max-files N]

Files are paired by name (tt_<label>_subNNN.h5). Per pair: format_version 4 vs 3; every dataset and every root
attribute but format_version identical (ttbar_tag: the format-3 columns identical, NaN included, and exactly the
five l1pf_* columns appended); MC: l1pf_* finite, in (0, 1], l1pf_up <= l1pf_nom <= l1pf_dn. Prints the mean
weights per sample and ALL OK, or the failures.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import h5py

NEW = ["l1pf_nom", "l1pf_up", "l1pf_dn", "l1pf_ecal_nom", "l1pf_muon_nom"]


def pairs(a, b):
    if os.path.isfile(a):
        return [(a, b)]
    out = []
    for p in sorted(glob.glob(os.path.join(a, "tt_*_sub*.h5"))):
        q = os.path.join(b, os.path.basename(p))
        if os.path.exists(q):
            out.append((p, q))
    return out


def compare(p4, p3):
    """List of problems of one pair, and the l1pf_nom values (MC)."""
    bad, w = [], None
    with h5py.File(p4, "r") as f4, h5py.File(p3, "r") as f3:
        if int(f4.attrs["format_version"]) != 4 or int(f3.attrs["format_version"]) != 3:
            bad.append("format_version %s / %s" % (f4.attrs["format_version"], f3.attrs["format_version"]))
        for k in set(f3.attrs) | set(f4.attrs):
            if k == "format_version":
                continue
            if k not in f3.attrs or k not in f4.attrs or str(f3.attrs[k]) != str(f4.attrs[k]):
                if k not in ("versions",):
                    bad.append("attribute %s differs" % k)
        names3, names4 = [], []
        f3.visit(names3.append)
        f4.visit(names4.append)
        if sorted(names3) != sorted(names4):
            bad.append("datasets differ: %s" % sorted(set(names3) ^ set(names4)))
        for k in names3:
            if not isinstance(f3[k], h5py.Dataset) or k not in f4:
                continue
            a3, a4 = f3[k][...], f4[k][...]
            if k == "ttbar_tag":
                c3 = f3[k].attrs["columns"].split(",")
                c4 = f4[k].attrs["columns"].split(",")
                if c4 != c3 + NEW:
                    bad.append("ttbar_tag columns: not the format-3 ones + %s" % NEW)
                    continue
                if not np.array_equal(a4[:, :len(c3)], a3, equal_nan=True):
                    bad.append("ttbar_tag: format-3 columns differ")
                w = dict((n, a4[:, c4.index(n)].astype(np.float64)) for n in NEW)
            elif a3.dtype.kind == "f":
                if not np.array_equal(a3, a4, equal_nan=True):
                    bad.append("%s differs" % k)
            elif not np.array_equal(a3, a4):
                bad.append("%s differs" % k)
        mc = "gen_weight" in f4
    nom = None
    if mc and not bad and w is not None:
        for n, x in w.items():
            if len(x) and not (np.isfinite(x).all() and (x > 0).all() and (x <= 1).all()):
                bad.append("%s not finite in (0, 1]" % n)
        if len(w["l1pf_nom"]) and not ((w["l1pf_up"] <= w["l1pf_nom"] + 1e-9).all() and (w["l1pf_dn"] >= w["l1pf_nom"] - 1e-9).all()):
            bad.append("not l1pf_up <= l1pf_nom <= l1pf_dn")
        nom = w["l1pf_nom"]
    return bad, nom


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("v4")
    ap.add_argument("v3")
    ap.add_argument("--max-files", type=int, default=0)
    a = ap.parse_args(argv)
    P = pairs(a.v4, a.v3)
    if a.max_files:
        P = P[:a.max_files]
    if not P:
        sys.exit("no pairs of files with the same name")
    fails, means = [], {}
    for p4, p3 in P:
        bad, nom = compare(p4, p3)
        name = os.path.basename(p4)
        if bad:
            fails.append((name, bad))
            print("FAIL  %s: %s" % (name, "; ".join(bad)))
        lab = re.sub(r"_sub\d+\.h5$", "", name)
        if nom is not None and len(nom):
            m = means.setdefault(lab, [0.0, 0])
            m[0] += nom.sum()
            m[1] += len(nom)
    for lab, (s, n) in sorted(means.items()):
        print("      %-28s mean l1pf_nom %.4f over %d rows" % (lab, s / n, n))
    print("%d pairs compared; " % len(P) + ("ALL OK" if not fails else "%d FAILED" % len(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
