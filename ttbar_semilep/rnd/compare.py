#!/usr/bin/env python3
"""Compare selection variants (maker output dirs) at the btag and tau32 stages, mu + e combined.

    python3 compare.py out_an out_an_veto300 [--lumi 16.39]

Per variant / stage / probe-pT threshold: data events (projected to 2016G+H, raw in brackets), MC composition
(ttbar, single top, W+jets, QCD = MuEnriched for mu + HT-binned for e) with MC-statistics errors, and the top
purity = fully merged top probes (from ttbar + single top) / all MC.  Uses purity.py's loaders.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import purity as P  # noqa: E402

GROUPS = ["ttbar", "single_top", "wjets", "qcd"]


def frac(num, num_v, rest, rest_v):
    """num / (num + rest) and its error for independent num, rest (variances num_v, rest_v)."""
    tot = num + rest
    if tot <= 0:
        return np.nan, np.nan
    return num / tot, np.sqrt(rest ** 2 * num_v + num ** 2 * rest_v) / tot ** 2


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("outdirs", nargs="+")
    p.add_argument("--lumi", type=float, default=16.39)
    a = p.parse_args(argv)
    info = json.load(open(os.path.join(HERE, "samples.json")))
    print("%-24s %-6s %4s %13s | %-12s %-12s %-12s %-12s | %-12s %-12s %s" % (
        "variant", "stage", "pT>", "data (raw)", "ttbar", "single top", "W+jets", "QCD (raw)", "tt / all", "t+tW / all",
        "TOP purity"))
    for od in a.outdirs:
        S = dict((n, P.load_sample(n, i, a.lumi, od)) for n, i in info.items())
        G = {}
        for grp in ("ttbar", "single_top", "wjets", "qcd_mu", "qcd_ht", "data_mu", "data_el"):
            G[grp] = P.merge([S[n] for n, i in info.items() if i["group"] == grp])
        st_ix = G["ttbar"]["stages"].index
        for stage in ("btag", "tau32"):
            for lo in (300, 400, 500):
                y, v, nraw = {}, {}, {}

                def sel(d, ch):
                    m = (d["stage"] >= st_ix(stage)) & (d["channel"] == P.CH[ch]) & (d["pt"] > lo)
                    return m & ((d["trig_mu"] > 0) if ch == "mu" else (d["trig_el"] > 0))
                for g in ("ttbar", "single_top", "wjets"):
                    ms = [sel(G[g], ch) for ch in ("mu", "el")]
                    y[g] = sum(G[g]["w"][m].sum() for m in ms)
                    v[g] = sum((G[g]["w"][m] ** 2).sum() for m in ms)
                    if g in ("ttbar", "single_top"):
                        y[g + "_t"] = sum(G[g]["w"][m & (G[g]["cat"] == 3)].sum() for m in ms)
                        v[g + "_t"] = sum((G[g]["w"][m & (G[g]["cat"] == 3)] ** 2).sum() for m in ms)
                mq, eq = sel(G["qcd_mu"], "mu"), sel(G["qcd_ht"], "el")
                y["qcd"] = G["qcd_mu"]["w"][mq].sum() + G["qcd_ht"]["w"][eq].sum()
                v["qcd"] = (G["qcd_mu"]["w"][mq] ** 2).sum() + (G["qcd_ht"]["w"][eq] ** 2).sum()
                nraw["qcd"] = int(mq.sum() + eq.sum())
                dm, de = sel(G["data_mu"], "mu"), sel(G["data_el"], "el")
                dy = G["data_mu"]["w"][dm].sum() + G["data_el"]["w"][de].sum()
                dn = int(dm.sum() + de.sum())
                tot = sum(y[g] for g in GROUPS)
                vtot = sum(v[g] for g in GROUPS)
                f_tt = frac(y["ttbar"], v["ttbar"], tot - y["ttbar"], vtot - v["ttbar"])
                tops = y["ttbar"] + y["single_top"]
                f_top = frac(tops, v["ttbar"] + v["single_top"], y["wjets"] + y["qcd"], v["wjets"] + v["qcd"])
                t = y["ttbar_t"] + y["single_top_t"]
                vt = v["ttbar_t"] + v["single_top_t"]
                f_t = frac(t, vt, tot - t, max(vtot - vt, 0.0))

                def sh(g):
                    return "%5.3f+-%5.3f" % (y[g] / tot, np.sqrt(v[g]) / tot) if tot else "-"
                print("%-24s %-6s %4d %7.0f(%4d) | %-12s %-12s %-12s %-6s(%3d) | %5.3f+-%5.3f  %5.3f+-%5.3f  %5.3f+-%5.3f" % (
                    od if (stage == "btag" and lo == 300) else "", stage if lo == 300 else "", lo, dy, dn,
                    sh("ttbar"), sh("single_top"), sh("wjets"), "%5.3f" % (y["qcd"] / tot if tot else np.nan), nraw["qcd"],
                    f_tt[0], f_tt[1], f_top[0], f_top[1], f_t[0], f_t[1]))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
