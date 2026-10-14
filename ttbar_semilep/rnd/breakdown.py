#!/usr/bin/env python3
"""Composition of the final selection (stage tau32), mu + e: fractions of all MC by probe truth category.

    python3 breakdown.py OUTDIR [--stage tau32]

Categories: fully merged top (t), merged W (q q'), b + one q, other (not merged) -- for ttbar and single top --
then W+jets and QCD (MuEnriched for mu, HT-binned for e).  +- MC statistics.  Also shows the effect of the
single largest-weight QCD event, since QCD survives with ~0-1 raw MC events.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import purity as P  # noqa: E402

BINS = [(300, 400), (400, 500), (500, np.inf), (300, np.inf), (400, np.inf)]
NAMES = [("t", 3), ("W", 2), ("bq", 1), ("other", 0)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("outdir")
    p.add_argument("--stage", default="tau32")
    p.add_argument("--lumi", type=float, default=16.39)
    a = p.parse_args(argv)
    info = json.load(open(os.path.join(HERE, "samples.json")))
    S = dict((n, P.load_sample(n, i, a.lumi, a.outdir)) for n, i in info.items())
    G = dict((g, P.merge([S[n] for n, i in info.items() if i["group"] == g]))
             for g in ("ttbar", "single_top", "wjets", "qcd_mu", "qcd_ht", "data_mu", "data_el"))
    ix = G["ttbar"]["stages"].index(a.stage)

    def sel(d, ch, lo, hi):
        m = (d["stage"] >= ix) & (d["channel"] == P.CH[ch]) & (d["pt"] > lo) & (d["pt"] <= hi)
        return m & ((d["trig_mu"] > 0) if ch == "mu" else (d["trig_el"] > 0))

    cols = ["top: t", "top: W", "top: bq", "top: other", "W+jets", "QCD"]
    print("stage %s, mu + e, fractions of all MC (+- MC stat); top = ttbar + single top" % a.stage)
    print("%-10s %9s %9s | " % ("pT bin", "data", "MC") + " ".join("%-13s" % c for c in cols) + " | t w/o top QCD event")
    for lo, hi in BINS:
        y, v = {}, {}
        for nm, code in NAMES:
            y[nm] = v[nm] = 0.0
            for g in ("ttbar", "single_top"):
                for ch in ("mu", "el"):
                    m = sel(G[g], ch, lo, hi) & (G[g]["cat"] == code)
                    y[nm] += G[g]["w"][m].sum()
                    v[nm] += (G[g]["w"][m] ** 2).sum()
        mw = [sel(G["wjets"], ch, lo, hi) for ch in ("mu", "el")]
        y["W+jets"] = sum(G["wjets"]["w"][m].sum() for m in mw)
        v["W+jets"] = sum((G["wjets"]["w"][m] ** 2).sum() for m in mw)
        qm, qe = sel(G["qcd_mu"], "mu", lo, hi), sel(G["qcd_ht"], "el", lo, hi)
        qw = np.concatenate([G["qcd_mu"]["w"][qm], G["qcd_ht"]["w"][qe]])
        y["QCD"], v["QCD"] = qw.sum(), (qw ** 2).sum()
        tot = sum(y.values())
        data = G["data_mu"]["w"][sel(G["data_mu"], "mu", lo, hi)].sum() + G["data_el"]["w"][sel(G["data_el"], "el", lo, hi)].sum()
        keys = ["t", "W", "bq", "other", "W+jets", "QCD"]
        cells = ["%5.3f+-%5.3f" % (y[k] / tot, np.sqrt(v[k]) / tot) if tot else "-" for k in keys]
        t_noq = y["t"] / (tot - (qw.max() if len(qw) else 0.0)) if tot else np.nan
        print("%-10s %9.0f %9.0f | %s | %5.3f (QCD raw %d)" % (
            "%d-%s" % (lo, "inf" if not np.isfinite(hi) else int(hi)), data, tot, " ".join("%-13s" % c for c in cells),
            t_noq, len(qw)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
