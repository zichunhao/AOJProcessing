#!/usr/bin/env python3
"""Composition of the tag-and-probe selection with the full LPC MC (<store>/ttbar/tt_<label>_sub*.h5).

    python3 full_mc.py STORE_TTBAR_DIR [--cache DIR] [--lumi 16.39]

Reads the small datasets only (no PFCands) and caches them per label in --cache (npz), so a re-run is fast.
Per probe-pT bin and stage (tau32 = final, btag, dphi): MC yield, truth composition of the top-quark samples
(ttbar + single top: fully merged t, merged W, b+q, other), single top and W+jets fractions, QCD with its raw
MC count, and the top-event purity. QCD: MuEnrichedPt5 for the mu channel, HT-binned madgraphMLM for the e
channel (as in the R&D). Cross sections: LPC_PROMPT.md for ttbar / single top / W+jets, Open Data records for QCD.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import h5py

XSEC = {"TTToSemiLeptonic": 365.34, "TTTo2L2Nu": 88.29, "TTToHadronic": 377.96,
        "ST_tW_top": 19.47, "ST_tW_antitop": 19.47, "ST_tch_top": 136.02, "ST_tch_antitop": 80.95, "ST_sch": 3.36}
for _b, _x in zip([70, 100, 200, 400, 600, 800, 1200, 2500], [1275, 1270, 336.8, 44.64, 11.6, 4.898, 1.142, 0.02678]):
    XSEC["WJets_HT%d" % _b] = _x * 1.21                      # NNLO k-factor
# QCD: Open Data record values (cross_section.total_value, GenXSecAnalyzer; MuEnriched includes the filter efficiency)
for _b, _x in zip([120, 170, 300, 470, 600, 800, 1000], [21180., 7011., 620.3, 59.08, 18.20, 3.275, 1.079]):
    XSEC["QCD_MuEnr_Pt%d" % _b] = _x
for _b, _x in zip([300, 500, 700, 1000, 1500, 2000], [324700., 30350., 6425., 1117., 108.3, 21.94]):
    XSEC["QCD_HT%d" % _b] = _x
CATS = ["other", "bq", "W", "t"]
BINS = [(300, 400), (400, 500), (500, np.inf), (300, np.inf), (400, np.inf)]


def group(label):
    if label.startswith("TT"):
        return "ttbar"
    if label.startswith("ST"):
        return "single_top"
    if label.startswith("WJets"):
        return "wjets"
    return "qcd_mu" if "MuEnr" in label else "qcd_ht"


def read_label(tt_dir, label, cache):
    npz = os.path.join(cache, label + ".npz") if cache else None
    if npz and os.path.exists(npz):
        z = np.load(npz)
        return {k: z[k] for k in z.files}
    tag, kin, cat, gw, sumw, n_ent = [], [], [], [], [], []
    for p in sorted(glob.glob(os.path.join(tt_dir, "tt_%s_sub*.h5" % label))):
        with h5py.File(p, "r") as f:
            cols, stages = f["ttbar_tag"].attrs["columns"], f["cutflow"].attrs["stages"]
            tag.append(f["ttbar_tag"][:])
            kin.append(f["jet_kinematics"][:])
            cat.append(f["truth"][:, f["truth"].attrs["columns"].split(",").index("category")])
            gw.append(f["gen_weight"][:])
            sumw.append(float(f.attrs["sum_genweight"]))
            n_ent.append(int(f.attrs["n_entries"]))
    z = dict(tag=np.concatenate(tag), kin=np.concatenate(kin), cat=np.concatenate(cat), gw=np.concatenate(gw),
             sumw=np.array(sumw), n_entries=np.array(n_ent), cols=np.array(cols), stages=np.array(stages))
    if npz:
        os.makedirs(cache, exist_ok=True)
        np.savez(npz, **z)
    return z


def load(tt_dir, cache, lumi):
    labels = sorted(set(re.sub(r"_sub\d+\.h5$", "", os.path.basename(p))[3:]
                        for p in glob.glob(os.path.join(tt_dir, "tt_*.h5"))))
    samples, stages = {}, None
    for lab in labels:
        if lab not in XSEC:
            print("skip %s: no cross section" % lab)
            continue
        z = read_label(tt_dir, lab, cache)
        cols = str(z["cols"]).split(",")
        stages = str(z["stages"]).split(",")
        d = {k: z["tag"][:, i] for i, k in enumerate(cols)}
        d["pt"], d["cat"] = z["kin"][:, 0], z["cat"]
        sumw = z["sumw"]
        assert np.isfinite(sumw).all() and (z["n_entries"] > 0).all(), lab
        d["w"] = z["gw"] * XSEC[lab] * lumi * 1e3 / sumw.sum()
        g = group(lab)
        trig = np.where(d["channel"] == 1, d["trig_mu"], np.where(d["channel"] == 2, d["trig_el"], 0)) > 0
        chan = {"qcd_mu": d["channel"] == 1, "qcd_ht": d["channel"] == 2}.get(g, d["channel"] > 0)
        d["ok"], d["group"] = trig & chan, g
        samples[lab] = d
        print("%-18s files=%4d entries=%11d rows=%8d" % (lab, len(sumw), z["n_entries"].sum(), len(d["pt"])))
    return samples, stages


def table(samples, stages, stage, chan=None):
    si = stages.index(stage)
    print("\n== stage >= %s, %s" % (stage, {None: "mu + e", 1: "mu", 2: "e"}[chan]))
    print("%-9s %8s | %6s %6s %6s %6s | %6s %6s | %-17s %-17s | %s" % (
        "probe pT", "MC evts", "t", "W", "bq", "other", "ST", "W+jets", "QCD mu (raw)", "QCD e (raw)", "top purity"))
    for lo, hi in BINS:
        y = dict.fromkeys(CATS + ["st", "wj", "qcd_mu", "qcd_ht"], 0.0)
        raw, var = {"qcd_mu": 0, "qcd_ht": 0}, {"qcd_mu": 0.0, "qcd_ht": 0.0}
        for d in samples.values():
            m = d["ok"] & (d["stage"] >= si) & (d["pt"] > lo) & (d["pt"] <= hi)
            if chan is not None:
                m &= d["channel"] == chan
            w = d["w"][m]
            if d["group"] in ("ttbar", "single_top"):
                for c, name in enumerate(CATS):
                    y[name] += d["w"][m & (d["cat"] == c)].sum()
                if d["group"] == "single_top":
                    y["st"] += w.sum()
            elif d["group"] == "wjets":
                y["wj"] += w.sum()
            else:
                y[d["group"]] += w.sum()
                raw[d["group"]] += int(m.sum())
                var[d["group"]] += (w ** 2).sum()
        tot = sum(y[k] for k in CATS + ["wj", "qcd_mu", "qcd_ht"])
        f = lambda v: v / tot
        q = lambda k: "%5.3f+-%5.3f (%4d)" % (f(y[k]), np.sqrt(var[k]) / tot, raw[k])
        print("%-9s %8.0f | %6.3f %6.3f %6.3f %6.3f | %6.3f %6.3f | %s %s | %.3f" % (
            "%d-%s" % (lo, "inf" if np.isinf(hi) else hi), tot, f(y["t"]), f(y["W"]), f(y["bq"]), f(y["other"]),
            f(y["st"]), f(y["wj"]), q("qcd_mu"), q("qcd_ht"), f(sum(y[k] for k in CATS))))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("tt_dir", help="directory with tt_<label>_sub*.h5 (e.g. <store>/ttbar)")
    p.add_argument("--cache", default=None, help="npz cache directory (default: none)")
    p.add_argument("--lumi", type=float, default=16.39, help="fb^-1 (default 2016G+H)")
    a = p.parse_args(argv)
    samples, stages = load(a.tt_dir, a.cache, a.lumi)
    print("\nweights: gen_weight * xsec * %.2f fb^-1 / sum(sum_genweight); trigger = the tag channel's paths;"
          " t/W/bq/other = truth of ttbar + single top" % a.lumi)
    for st in ("tau32", "btag", "dphi"):
        table(samples, stages, st)
    for ch in (1, 2):
        table(samples, stages, "tau32", ch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
