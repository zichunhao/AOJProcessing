#!/usr/bin/env python3
"""Yields and composition of loosened selections (V0 = 'an' ... V4 = the full skim), from the skim MC.

    python3 skim_variants.py CACHE_DIR      (CACHE_DIR written by skim_vs_an.py)

MC only (16.39 fb^-1), TTToHadronic left out (it is the fine-tuning signal). QCD MuEnriched in the mu channel,
HT-binned QCD in the e channel; cross sections from full_mc.py.
"""
import sys, glob, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import full_mc
C = sys.argv[1]; L = 16.39
CATS = ["other", "bq", "W", "t"]
S = {}
for p in sorted(glob.glob(C + "/*.npz")):
    l = os.path.basename(p)[:-4]
    if l == "TTToHadronic": continue                       # not in the evaluation MC
    z = np.load(p); cols = str(z["cols"]).split(","); T = z["tag"]
    d = {k: T[:, i] for i, k in enumerate(cols)}
    d.update(pt=z["kin"][:, 0], msd=z["kin"][:, 3], cat=z["cat"], g=full_mc.group(l))
    d["w"] = z["gw"] * full_mc.XSEC[l] * L * 1e3 / z["sumw"].sum()
    ch = d["channel"]
    d["trig_an"] = np.where(ch == 1, d["trig_mu_an"], np.where(ch == 2, d["trig_el_an"], 0)) > 0
    d["trig_sk"] = np.where(ch == 1, d["trig_mu"], np.where(ch == 2, d["trig_el"], 0)) > 0
    d["chan_ok"] = {"qcd_mu": ch == 1, "qcd_ht": ch == 2}.get(d["g"], ch > 0)
    S[l] = d
stages = str(np.load(C + "/TTToSemiLeptonic.npz")["stages"]).split(",")
iso30 = lambda d: (d["lep_id_iso"] > 0) & (d["lep_pt"] > 30) & ((d["n_veto_mu"] + d["n_veto_el"]) == 1)
V = [
    ("V0 an (pT_l>55 iso, 1 lep, MET>50, 1 AK8)", lambda d: (d["an_channel"] == d["channel"]) & d["trig_an"] & (d["met_pt"] > 50) & (d["n_ak8"] == 1)),
    ("V1 = V0 w/o extra-AK8 veto",                lambda d: (d["an_channel"] == d["channel"]) & d["trig_an"] & (d["met_pt"] > 50)),
    ("V2 = V1 + iso lepton pT > 30",              lambda d: iso30(d) & d["trig_an"] & (d["met_pt"] > 50)),
    ("V3 = V2 + MET > 20",                        lambda d: iso30(d) & d["trig_an"] & (d["met_pt"] > 20)),
    ("V4 = full skim (non-iso, Mu50/Ele115)",     lambda d: d["trig_sk"]),
]
for st in ("btag", "tau32"):
    si = stages.index(st)
    for lo, hi in [(300, 1e9), (400, 1e9)]:
        print("\n== stage >= %s, probe pT > %d, mu + e (MC, %.2f fb^-1; no TTToHadronic)" % (st, lo, L))
        print("%-42s %9s %8s | %5s %5s %5s %5s | %5s %5s | %-17s %-15s" % ("selection", "MC evts", "merged t", "t", "W", "bq", "oth", "ST", "W+j", "QCD mu (raw)", "QCD e (raw)"))
        for name, f in V:
            y = dict.fromkeys(CATS + ["st", "wj", "qcd_mu", "qcd_ht"], 0.0); raw = {"qcd_mu": 0, "qcd_ht": 0}
            for l, d in S.items():
                m = d["chan_ok"] & f(d) & (d["stage"] >= si) & (d["pt"] > lo) & (d["pt"] <= hi)
                w = d["w"][m]
                if d["g"] in ("ttbar", "single_top"):
                    for c, n in enumerate(CATS): y[n] += d["w"][m & (d["cat"] == c)].sum()
                    if d["g"] == "single_top": y["st"] += w.sum()
                elif d["g"] == "wjets": y["wj"] += w.sum()
                else: y[d["g"]] += w.sum(); raw[d["g"]] += int(m.sum())
            tot = sum(y[k] for k in CATS + ["wj", "qcd_mu", "qcd_ht"])
            f_ = lambda v: v / tot if tot else np.nan
            print("%-42s %9.0f %8.0f | %5.3f %5.3f %5.3f %5.3f | %5.3f %5.3f | %5.3f (%5d)     %5.3f (%4d)" % (name, tot, y["t"], f_(y["t"]), f_(y["W"]), f_(y["bq"]),
                  f_(y["other"]), f_(y["st"]), f_(y["wj"]), f_(y["qcd_mu"]), raw["qcd_mu"], f_(y["qcd_ht"]), raw["qcd_ht"]))
