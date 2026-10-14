"""Fraction of lepton-route ttbar probes (data) whose event also fired the AOJ JetHT OR, i.e. that are AOJ jets."""
import glob, sys, h5py, numpy as np
od = sys.argv[1]
for stage_name in ("btag", "tau32"):
    for lo in (300, 400, 500):
        n = nj = 0
        for pd, pat in (("mu", "sm"), ("el", "se")):
            for p in glob.glob("%s/%s[GH]_*.h5" % (od, pat)):
                with h5py.File(p, "r") as f:
                    tc = f["ttbar_tag"].attrs["columns"].split(","); T = f["ttbar_tag"][:]
                    c = {k: T[:, i] for i, k in enumerate(tc)}
                    st = f["cutflow"].attrs["stages"].split(",").index(stage_name)
                    m = (c["stage"] >= st) & (c["channel"] == (1 if pd == "mu" else 2)) & (f["jet_kinematics"][:, 0] > lo)
                    n += m.sum(); nj += (m & (c["trig_jetht"] > 0)).sum()
        print("%-6s pT>%d: %4d data events, %4d also fired the AOJ JetHT OR (%.0f%%)" % (stage_name, lo, n, nj, 100. * nj / max(n, 1)))
