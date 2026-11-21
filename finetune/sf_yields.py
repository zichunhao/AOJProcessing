#!/usr/bin/env python3
"""Data / MC yields and the merged-top fraction of the per-sample SF inputs (make_sf_inputs.py), per channel,
level (btag, tau32) and probe-pT bin.

    bash ttbar_semilep/py3env.sh finetune/sf_yields.py OUTDIR [--sel iso30_met20] [--json out.json]

MC: the nominal weight (`weight`), and for comparison w_norm only and the nominal x w_toppt; merged-top fraction
= MC weight with label 1 (truth category t) / all MC weight; MC composition by process group (nominal weight).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ttbar_semilep"))
import samples as smp  # noqa: E402

PT_BINS = [(300, 400), (400, 500), (500, 2500), (300, 2500)]
GROUP_NAMES = dict(ttbar="tt", single_top="t", wjets_lnu="W(lnu)", dy_ll="DY", wjets_qq="W(qq)", zjets_qq="Z(qq)",
                   diboson="VV", ttv="ttZ", qcd_mu="QCD", qcd="QCD")


def load(outdir, sel):
    rows = {}
    for p in sorted(glob.glob(os.path.join(outdir, "*.h5"))):
        label = os.path.basename(p)[:-3]
        with h5py.File(p, "r") as f:
            if "weight" not in f or f["weight"].shape[0] == 0:
                continue
            sc = f["sel"].attrs["columns"].split(",")
            tc = f["ttbar_tag"].attrs["columns"].split(",")
            wc = f["weights"].attrs["columns"].split(",")
            fl = f["sel"][:].astype(bool)
            m = fl[:, sc.index(sel)]
            W = f["weights"][:][m].astype(np.float64)
            rows[label] = dict(ch=f["ttbar_tag"][:, tc.index("channel")][m], pt=f["jet_kinematics"][:, 0][m],
                               btag=fl[m, sc.index("btag")], tau32=fl[m, sc.index("tau32")],
                               w=f["weight"][:][m].astype(np.float64), w_norm=W[:, wc.index("w_norm")],
                               w_toppt=W[:, wc.index("w_toppt")], label=f["label"][:][m],
                               group=smp.SAMPLES[label]["group"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--sel", default="iso30_met20")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    R = load(a.outdir, a.sel)
    res = []
    for lev in ("btag", "tau32"):
        print("\n== %s & %s, probe in 20 < mSD < 500 (data: events; MC: sum of weights, %s)" % (
            a.sel, lev, "nominal weight"))
        print("%-4s %-10s %7s %9s %13s %7s %7s | %9s %9s | %s" % ("ch", "pT [GeV]", "data", "MC", "data/MC", "f_t",
                                                                  "MC t", "d/MC norm", "d/MC +top", "MC composition"))
        for chn, cn in ((1, "mu"), (2, "e")):
            for lo, hi in PT_BINS:
                y = dict(data=0, mc=0.0, mc_norm=0.0, mc_toppt=0.0, t=0.0, groups={})
                for label, r in R.items():
                    m = r[lev] & (r["ch"] == chn) & (r["pt"] > lo) & (r["pt"] < hi)
                    if r["group"] == "data":
                        y["data"] += int(m.sum())
                        continue
                    y["mc"] += r["w"][m].sum()
                    y["mc_norm"] += r["w_norm"][m].sum()
                    y["mc_toppt"] += (r["w"][m] * r["w_toppt"][m]).sum()
                    y["t"] += r["w"][m & (r["label"] == 1)].sum()
                    g = GROUP_NAMES.get(r["group"], r["group"])
                    y["groups"][g] = y["groups"].get(g, 0.0) + r["w"][m].sum()
                ratio = y["data"] / y["mc"] if y["mc"] else float("nan")
                err = ratio / np.sqrt(y["data"]) if y["data"] else float("nan")
                comp = " ".join("%s %.3f" % (g, v / y["mc"]) for g, v in sorted(y["groups"].items(), key=lambda kv: -kv[1])
                                if y["mc"] and v / y["mc"] >= 0.005)
                print("%-4s %-10s %7d %9.1f %6.3f+-%.3f %7.3f %7.1f | %9.3f %9.3f | %s" % (
                    cn, "%d-%s" % (lo, hi if hi < 2500 else ""), y["data"], y["mc"], ratio, err, y["t"] / y["mc"] if y["mc"] else float("nan"),
                    y["t"], y["data"] / y["mc_norm"] if y["mc_norm"] else float("nan"),
                    y["data"] / y["mc_toppt"] if y["mc_toppt"] else float("nan"), comp))
                res.append(dict(sel=a.sel, level=lev, channel=cn, pt=[lo, hi], **y))
    print("\nd/MC norm: MC with w_norm only (no corrections); d/MC +top: nominal x w_toppt (top-pT reweighting). "
          "f_t: merged-top fraction of the MC; MC t: merged tops (MC weight). Uncertainty: data statistics only.")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
