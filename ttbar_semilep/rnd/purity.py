#!/usr/bin/env python3
"""How tight is the HIG-24-010-like selection, and how pure?  Reads out/*.h5 + samples.json (make_jobs.py).

    python3 purity.py [--lumi 16.39]

Prints (1) the cut flow of every MC group and of data (projected to 2016G+H) with per-cut efficiencies,
(2) per level (probe / btag / HIG-24-010 = btag + tau32 / + dphi(l,J) > 2) and probe-pT threshold: yields,
ttbar / (ttbar + QCD), ttbar / all MC, top purity (fully merged top probes from ttbar + single top / all MC)
and data / MC.  QCD = MuEnriched (mu channel) or HT-binned (el channel; the other one is shown as a cross-check).
MC weight = genWeight * xsec * lumi / sum(genEventSumw of the processed files of that sample); data weight =
events in the dataset / events processed (per sample).  Uncertainties: MC / data statistics only.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ttbar_selection as ts  # noqa: E402

CH = {"mu": 1, "el": 2}


def load_sample(name, info, lumi, outdir):
    paths = sorted(glob.glob(os.path.join(HERE, outdir, name + "_*.h5")))
    if not paths:
        return None
    rows, sumw, n_ent, cf = [], 0.0, 0, None
    for p in paths:
        with h5py.File(p, "r") as f:
            tc = f["ttbar_tag"].attrs["columns"].split(",")
            T = f["ttbar_tag"][:]
            d = {k: T[:, i] for i, k in enumerate(tc)}
            d["pt"], d["msd"] = f["jet_kinematics"][:, 0], f["jet_kinematics"][:, 3]
            if "truth" in f:
                d["cat"] = f["truth"][:, f["truth"].attrs["columns"].split(",").index("category")]
            d["gw"] = f["gen_weight"][:] if "gen_weight" in f else np.ones(len(T))
            rows.append(d)
            s = float(f.attrs["sum_genweight"])
            sumw += s if np.isfinite(s) else 0.0
            n_ent += int(f.attrs["n_entries"])
            c = f["cutflow/w"][:]
            cf = c if cf is None else cf + c
            stages = f["cutflow"].attrs["stages"].split(",")
    d = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    d["stages"] = stages
    if info["xsec"] is None:
        norm = info["n_events_dataset"] / n_ent
    else:
        norm = info["xsec"] * lumi * 1e3 / sumw
    d["w"] = d["gw"] * norm
    d["cutflow"] = cf * norm
    d["n_files"], d["n_entries"], d["norm"] = len(paths), n_ent, norm
    if "cat" not in d:
        d["cat"] = np.full(len(d["w"]), -1.0)
    return d


def merge(ds):
    ds = [d for d in ds if d is not None]
    if not ds:
        return None
    keys = ["channel", "trig_mu", "trig_el", "n_b", "pass_tau32", "dphi_lep_probe", "pt", "msd", "cat", "w"]
    keys += ["stage"] if all("stage" in d for d in ds) else []
    out = {k: np.concatenate([d[k] for d in ds]) for k in keys}
    out["stages"] = ds[0]["stages"]
    out["cutflow"] = sum(d["cutflow"] for d in ds)
    out["n_files"] = sum(d["n_files"] for d in ds)
    out["n_entries"] = sum(d["n_entries"] for d in ds)
    return out


def levels(d):
    if "stage" in d:                     # stage-based (files with the dphi stage)
        st, ix = d["stage"], d["stages"].index
        return {"probe (1 lep, MET, 1 AK8)": st >= ix("one_probe"), "+ dphi(l,J) > 2": st >= ix("dphi"),
                "+ btag": st >= ix("btag"), "+ tau32": st >= ix("tau32")}
    b = d["n_b"] >= 1                    # older outputs (paper-like selection, no stage column)
    t = b & (d["pass_tau32"] > 0)
    return {"probe (1 lep, MET, 1 AK8)": np.ones(len(b), bool), "+ btag": b, "+ tau32": t,
            "+ tau32 + dphi(l,J) > 2": t & (d["dphi_lep_probe"] > 2)}


def ys(d, m):
    return float(d["w"][m].sum()), float(np.sqrt((d["w"][m] ** 2).sum())), int(m.sum())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--lumi", type=float, default=16.39)
    p.add_argument("--outdir", default="out", help="maker outputs (relative to rnd/)")
    a = p.parse_args(argv)
    info = json.load(open(os.path.join(HERE, "samples.json")))
    S = dict((n, load_sample(n, i, a.lumi, a.outdir)) for n, i in info.items())
    G = {}
    for grp in ("ttbar", "single_top", "wjets", "qcd_mu", "qcd_ht", "data_mu", "data_el"):
        G[grp] = merge([S[n] for n, i in info.items() if i["group"] == grp])
    G["tt_semilep"] = S["tt"]
    print("processed: " + ", ".join("%s %d files / %.2fM ev" % (g, G[g]["n_files"], G[g]["n_entries"] / 1e6)
                                    for g in G if G[g] is not None and g != "tt_semilep"))
    print("lumi %.2f fb^-1; data projected to 2016G+H by event count\n" % a.lumi)

    # ---------------------------------------------------------------- (1) cut flow
    for ch, j in (("mu", 0), ("el", 1)):
        qcd = "qcd_mu" if ch == "mu" else "qcd_ht"
        data = "data_" + ch
        cols = ["tt_semilep", "ttbar", "single_top", "wjets", qcd, data]
        print("CUT FLOW %s channel (yield; efficiency w.r.t. previous cut)" % ch)
        print("  %-14s" % "cut" + "".join("%24s" % c for c in cols))
        for k, stage in enumerate(G["ttbar"]["stages"]):
            line = "  %-14s" % stage
            for c in cols:
                cf = G[c]["cutflow"][:, j]
                eff = cf[k] / cf[k - 1] if k and cf[k - 1] > 0 else np.nan
                line += "%14.4g (%5.3f)  " % (cf[k], eff) if k else "%14.4g          " % cf[k]
            print(line)
        tt = G["tt_semilep"]["cutflow"][:, j]
        print("  -> TTToSemiLeptonic: HIG-24-010 selection keeps %.2e of all events (%.2e of %s-channel triggered)\n"
              % (tt[-1] / tt[0], tt[-1] / tt[1], ch))

    # ---------------------------------------------------------------- (2) composition per level
    for ch in ("mu", "el"):
        qcd, qcd_alt = ("qcd_mu", "qcd_ht") if ch == "mu" else ("qcd_ht", "qcd_mu")
        print("COMPOSITION %s channel (pT > threshold).  yields: ttbar | single top | W+jets | QCD (%s; alt %s) | data"
              % (ch, qcd, qcd_alt))
        hdr = ("  %-26s %4s %9s %8s %8s %13s %9s  %-15s %-15s %-15s %s"
               % ("level", "pT>", "ttbar", "1top", "W+jets", "QCD (alt)", "data", "tt/(tt+QCD)", "tt/allMC",
                  "TOP purity", "data/MC"))
        print(hdr)
        for lname in levels(G["ttbar"]):
            for lo in (300, 400, 500):
                y = {}
                for g in ("ttbar", "single_top", "wjets", qcd, qcd_alt, "data_" + ch):
                    d = G[g]
                    m = levels(d)[lname] & (d["channel"] == CH[ch]) & (d["pt"] > lo)
                    m &= (d["trig_mu"] > 0) if ch == "mu" else (d["trig_el"] > 0)
                    y[g] = ys(d, m)
                    if g in ("ttbar", "single_top"):
                        y[g + "_t"] = ys(d, m & (d["cat"] == 3))
                tt, st, wj, q = y["ttbar"][0], y["single_top"][0], y["wjets"][0], y[qcd][0]
                mc = tt + st + wj + q
                r1 = tt / (tt + q) if tt + q else np.nan
                e1 = r1 * (1 - r1) * np.hypot(y["ttbar"][1] / tt if tt else 0, y[qcd][1] / q if q else 0)
                r2 = tt / mc if mc else np.nan
                e2 = np.sqrt((r2 * (1 - r2)) ** 2 * ((y["ttbar"][1] / tt) ** 2 if tt else 0)
                             + (r2 ** 2) * (y[qcd][1] ** 2 + y["wjets"][1] ** 2 + y["single_top"][1] ** 2) / mc ** 2) if mc else np.nan
                tp = (y["ttbar_t"][0] + y["single_top_t"][0]) / mc if mc else np.nan
                dd = y["data_" + ch]
                print("  %-26s %4d %9.0f %8.0f %8.0f %6.0f (%5.0f) %5.0f(%3d)  %5.3f+-%5.3f    %5.3f+-%5.3f    %5.3f            %4.2f+-%4.2f"
                      % (lname if lo == 300 else "", lo, tt, st, wj, q, y[qcd_alt][0], dd[0], dd[2], r1, e1, r2, e2, tp,
                         dd[0] / mc if mc else np.nan, dd[0] / mc / np.sqrt(dd[2]) if dd[2] and mc else np.nan))
            print("  " + "-" * (len(hdr) - 2))
        print("  QCD MC statistics (raw events at each level, %s): %s\n" % (qcd, ", ".join(
            "%s %d" % (ln.split(" ")[0] if ln[0] != "+" else ln.split(" ")[1], ys(G[qcd], levels(G[qcd])[ln] & (G[qcd]["channel"] == CH[ch]))[2])
            for ln in levels(G[qcd]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
