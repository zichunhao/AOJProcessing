#!/usr/bin/env python3
"""Offline tightening of the final selection: event-level ttbar(+single top) purity vs jet-level merged-top purity.

    python3 variants.py OUTDIR

For each extra cut on top of stage tau32 (mu + e): projected data events, MC fraction of top-quark events
(ttbar + single top), and the fraction of probes that are fully merged tops (b, q, q' within dR < 0.8).
QCD MC has ~0-1 raw events here; the top-event purity is shown with and without it.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
LUMI = 16.39


def load(od, info, name):
    rows, sumw, n_ent = [], 0.0, 0
    for pth in sorted(glob.glob(os.path.join(HERE, od, name + "_*.h5"))):
        with h5py.File(pth, "r") as f:
            tc = f["ttbar_tag"].attrs["columns"].split(",")
            T = f["ttbar_tag"][:]
            d = {k: T[:, i] for i, k in enumerate(tc)}
            d["pt"], d["msd"] = f["jet_kinematics"][:, 0], f["jet_kinematics"][:, 3]
            d["cat"] = (f["truth"][:, f["truth"].attrs["columns"].split(",").index("category")] if "truth" in f
                        else np.full(len(T), -1.0))
            d["gw"] = f["gen_weight"][:] if "gen_weight" in f else np.ones(len(T))
            d["final"] = d["stage"] >= f["cutflow"].attrs["stages"].split(",").index("tau32")
            s = float(f.attrs["sum_genweight"])
            sumw += s if np.isfinite(s) else 0.0
            n_ent += int(f.attrs["n_entries"])
            rows.append(d)
    d = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    i = info[name]
    d["w"] = d["gw"] * (i["n_events_dataset"] / n_ent if i["xsec"] is None else i["xsec"] * LUMI * 1e3 / sumw)
    trig = np.where(d["channel"] == 1, d["trig_mu"], np.where(d["channel"] == 2, d["trig_el"], 0)) > 0
    keep_ch = {"data_mu": 1, "data_el": 2}.get(i["group"])
    d["base"] = d["final"] & trig & ((d["channel"] == keep_ch) if keep_ch else np.ones(len(trig), bool))
    if i["group"] == "qcd_mu":
        d["base"] &= d["channel"] == 1
    if i["group"] == "qcd_ht":
        d["base"] &= d["channel"] == 2
    return i["group"], d


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("outdir")
    a = p.parse_args(argv)
    info = json.load(open(os.path.join(HERE, "samples.json")))
    samples = [load(a.outdir, info, n) for n in info]
    V = [
        ("final (default)", lambda d: d["pt"] > 300),
        ("+ 105 < mSD < 210", lambda d: (d["pt"] > 300) & (d["msd"] > 105) & (d["msd"] < 210)),
        ("+ 140 < mSD < 200", lambda d: (d["pt"] > 300) & (d["msd"] > 140) & (d["msd"] < 200)),
        ("+ tau32 < 0.4", lambda d: (d["pt"] > 300) & (d["tau32"] < 0.4)),
        ("pT > 400", lambda d: d["pt"] > 400),
        ("pT > 400 + 105 < mSD < 210", lambda d: (d["pt"] > 400) & (d["msd"] > 105) & (d["msd"] < 210)),
        ("pT > 400 + 140 < mSD < 200", lambda d: (d["pt"] > 400) & (d["msd"] > 140) & (d["msd"] < 200)),
    ]
    print("%-28s %12s %9s | %-22s %-14s | %s" % ("extra cut on top of final", "data (raw)", "MC",
                                                "top events (no QCD evt)", "merged-top", "merged tops in data"))
    for name, fn in V:
        y = {"top": 0.0, "t": 0.0, "wjets": 0.0, "qcd": 0.0, "data": 0.0}
        vt, nd, qmax = 0.0, 0, 0.0
        for grp, d in samples:
            m = d["base"] & fn(d)
            w = d["w"][m]
            if grp in ("ttbar", "single_top"):
                y["top"] += w.sum()
                y["t"] += d["w"][m & (d["cat"] == 3)].sum()
                vt += (d["w"][m & (d["cat"] == 3)] ** 2).sum()
            elif grp == "wjets":
                y["wjets"] += w.sum()
            elif grp.startswith("qcd"):
                y["qcd"] += w.sum()
                qmax = max(qmax, w.max() if len(w) else 0.0)
            else:
                y["data"] += w.sum()
                nd += int(m.sum())
        tot = y["top"] + y["wjets"] + y["qcd"]
        print("%-28s %6.0f(%4d) %9.0f | %5.3f (%5.3f)          %5.3f+-%5.3f   ~%.0f" % (
            name, y["data"], nd, tot, y["top"] / tot, y["top"] / (tot - qmax), y["t"] / tot, np.sqrt(vt) / tot,
            y["data"] * y["t"] / tot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
