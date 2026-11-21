#!/usr/bin/env python3
"""Data / MC control plots of the per-sample SF inputs (make_sf_inputs.py), mu and e side by side.

    bash ttbar_semilep/py3env.sh finetune/sf_plots.py OUTDIR [--sel iso30_met20] [--plots OUTDIR/plots]

At `sel & btag`: event variables (npv, lepton pT / eta, MET, probe pT) stacked by process; probe variables (pT,
mSD, tau32, ParticleNet TvsQCD) stacked by the probe's truth (top samples: merged t / merged W / b+q / other;
the rest: non-top processes); probe variables also at `sel & tau32`. MC: the nominal `weight`; ratio panel:
data / MC with the data statistical uncertainty and the MC statistical band. PDF and PNG.
"""
import argparse
import glob
import os
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ttbar_semilep"))
import samples as smp  # noqa: E402

# categorical slots in fixed order (dataviz reference palette, validated: light surface, 6 slots)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PROCESSES = [("tt", "$t\\bar{t}$", ("ttbar",)), ("st", "single $t$", ("single_top",)),
             ("wj", "W($\\ell\\nu$)+jets", ("wjets_lnu",)), ("dy", "DY($\\ell\\ell$)+jets", ("dy_ll",)),
             ("oth", "$t\\bar{t}$Z, VV, W/Z($qq$)", ("ttv", "diboson", "wjets_qq", "zjets_qq")),
             ("qcd", "QCD", ("qcd", "qcd_mu"))]
TRUTHS = [("t", "merged $t$"), ("W", "merged W"), ("bq", "b + q"), ("other", "other (top samples)"),
          ("nontop", "non-top processes")]
TOP_GROUPS = ("ttbar", "single_top", "ttv")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
EVENT_VARS = [("npv_good", "good primary vertices", np.arange(0, 61, 2), False),
              ("lep_pt", "lepton $p_T$ [GeV]", np.array([30, 40, 50, 60, 70, 80, 100, 120, 150, 200, 300, 500]), True),
              ("lep_eta", "lepton $\\eta$", np.linspace(-2.5, 2.5, 21), False),
              ("met_pt", "$p_T^{miss}$ [GeV]", np.array([20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 500]), True),
              ("probe_pt", "probe $p_T$ [GeV]", np.array([300, 350, 400, 450, 500, 600, 700, 800, 1000, 1500]), True)]
PROBE_VARS = [("probe_pt", "probe $p_T$ [GeV]", np.array([300, 350, 400, 450, 500, 600, 700, 800, 1000, 1500]), True),
              ("probe_msd", "probe $m_{SD}$ [GeV]", np.arange(20, 301, 10), False),
              ("tau32", "probe $\\tau_{32}$", np.linspace(0, 1, 21), False),
              ("pn_tvsqcd", "ParticleNet TvsQCD", np.linspace(0, 1, 21), True)]


def load(outdir, sel):
    rows = []
    for p in sorted(glob.glob(os.path.join(outdir, "*.h5"))):
        label = os.path.basename(p)[:-3]
        with h5py.File(p, "r") as f:
            if "weight" not in f or f["weight"].shape[0] == 0:
                continue
            sc = f["sel"].attrs["columns"].split(",")
            tc = f["ttbar_tag"].attrs["columns"].split(",")
            fl = f["sel"][:].astype(bool)
            m = fl[:, sc.index(sel)] & fl[:, sc.index("btag")]
            T = f["ttbar_tag"][:][m]
            jk = f["jet_kinematics"][:][m]
            g = smp.SAMPLES[label]["group"]
            r = dict(group=g, w=f["weight"][:][m].astype(np.float64), tau32_stage=fl[m, sc.index("tau32")],
                     channel=T[:, tc.index("channel")], probe_pt=jk[:, 0], probe_msd=jk[:, 3],
                     pn_tvsqcd=f["jet_tagging"][:, 9][m])
            for k in ("npv_good", "lep_pt", "lep_eta", "met_pt", "tau32"):
                r[k] = T[:, tc.index(k)]
            r["truth"] = f["truth"][:, 0][m] if "truth" in f else np.full(m.sum(), -1)
            rows.append(r)
    return rows


def classify(r):
    """Per row: process key and truth key."""
    proc = next(k for k, _, gs in PROCESSES if r["group"] in gs) if r["group"] != "data" else "data"
    if r["group"] in TOP_GROUPS:
        truth = np.array(["other", "bq", "W", "t"])[np.clip(r["truth"].astype(int), 0, 3)]
    else:
        truth = np.full(len(r["w"]), "nontop")
    return proc, truth


def panel(ax, rax, data, stacks, mc_w2, bins, logy, xlabel):
    """Stacked MC (list of (label, colour, histogram)), data points, ratio below."""
    centers = 0.5 * (bins[1:] + bins[:-1])
    bottom = np.zeros(len(bins) - 1)
    for label, colour, h in stacks:
        ax.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge", color=colour, label=label,
               edgecolor="#fcfcfb", linewidth=0.6)
        bottom = bottom + h
    mc = bottom
    err = np.sqrt(data)
    ax.errorbar(centers, data, yerr=err, fmt="o", ms=3.5, color=INK, elinewidth=1, label="data")
    ax.set_yscale("log" if logy else "linear")
    if logy:
        ax.set_ylim(max(0.5, 0.3 * min(x for x in data if x > 0) if (data > 0).any() else 0.5), 30 * max(mc.max(), data.max(), 1))
    else:
        ax.set_ylim(0, 1.35 * max(mc.max(), data.max(), 1))
    ax.set_ylabel("events / bin", color=INK)
    ok = mc > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(ok & (data > 0), data / mc, np.nan)
        rerr = np.where(ok & (data > 0), err / mc, 0.0)
        band = np.where(ok, np.sqrt(np.maximum(mc_w2, 0)) / mc, 0.0)
    rax.fill_between(bins, np.r_[1 - band, (1 - band)[-1]], np.r_[1 + band, (1 + band)[-1]], step="post",
                     color="#c3c2b7", alpha=0.6, linewidth=0, label="MC stat.")
    rax.errorbar(centers, ratio, yerr=rerr, fmt="o", ms=3, color=INK, elinewidth=1)
    rax.axhline(1, color=MUTED, linewidth=0.8)
    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel("data / MC", color=INK)
    rax.set_xlabel(xlabel, color=INK)
    for a in (ax, rax):
        a.grid(axis="y", color=GRID, linewidth=0.5)
        a.set_axisbelow(True)
        a.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
        for s in a.spines.values():
            s.set_color("#c3c2b7")
    ax.set_xlim(bins[0], bins[-1])


def figure(rows, var, xlabel, bins, logy, split, stage, sel, lumi, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col", gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
    for col, (chn, cname) in enumerate(((1, "$\\mu$ channel"), (2, "$e$ channel"))):
        data = np.zeros(len(bins) - 1)
        keys = PROCESSES if split == "process" else TRUTHS
        h = dict((k[0], np.zeros(len(bins) - 1)) for k in keys)
        w2 = np.zeros(len(bins) - 1)
        for r in rows:
            m = (r["channel"] == chn) & (r["tau32_stage"] if stage == "tau32" else True)
            x = np.clip(r[var][m], bins[0], bins[-1] - 1e-6 * (bins[-1] - bins[0]))      # overflow into the last bin
            if r["proc"] == "data":
                data += np.histogram(x, bins)[0]
                continue
            w = r["w"][m]
            w2 += np.histogram(x, bins, weights=w * w)[0]
            if split == "process":
                h[r["proc"]] += np.histogram(x, bins, weights=w)[0]
            else:
                tr = r["truth_key"][m]
                for k, _ in TRUTHS:
                    h[k] += np.histogram(x[tr == k], bins, weights=w[tr == k])[0]
        names = [(k[0], k[1]) for k in keys]
        stacks = [(lab, SLOTS[i], h[k]) for i, (k, lab) in reversed(list(enumerate(names)))]  # first entry on top
        ax, rax = axes[0, col], axes[1, col]
        panel(ax, rax, data, stacks, w2, bins, logy, xlabel)
        tot_mc = sum(x[2].sum() for x in stacks)
        ax.set_title("%s   data %d, MC %.0f" % (cname, data.sum(), tot_mc), fontsize=9, color=INK, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index("data")] + [i for i, l in enumerate(labels) if l != "data"][::-1]
    axes[0, 1].legend([handles[i] for i in order], [labels[i] for i in order], fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("CMS Open Data 2016G+H, %.2f fb$^{-1}$ (13 TeV); %s & %s; MC: nominal weight" % (lumi, sel, stage),
                 fontsize=9, color=INK, x=0.07, ha="left")
    fig.patch.set_facecolor("#fcfcfb")
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (path, ext), dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--sel", default="iso30_met20")
    ap.add_argument("--plots", default=None)
    a = ap.parse_args(argv)
    out = a.plots or os.path.join(a.outdir, "plots")
    os.makedirs(out, exist_ok=True)
    rows = load(a.outdir, a.sel)
    for r in rows:
        r["proc"], r["truth_key"] = classify(r)
    with h5py.File(sorted(glob.glob(os.path.join(a.outdir, "*.h5")))[0], "r") as f:
        lumi = float(f.attrs["lumi_fb"])
    for var, xl, bins, logy in EVENT_VARS:
        figure(rows, var, xl, bins, logy, "process", "btag", a.sel, lumi, os.path.join(out, "%s_btag_%s" % (a.sel, var)))
    for stage in ("btag", "tau32"):
        for var, xl, bins, logy in PROBE_VARS:
            figure(rows, var, xl, bins, logy, "truth", stage, a.sel, lumi, os.path.join(out, "%s_%s_%s_truth" % (a.sel, stage, var)))
    print("plots in %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
