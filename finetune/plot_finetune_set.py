#!/usr/bin/env python3
"""Plots of a make_finetune_set.py output, for its README and for the fine-tuning session.

    bash ttbar_semilep/py3env.sh finetune/plot_finetune_set.py OUTDIR [-o OUTDIR/plots]

Reads the small datasets only (no PFCands). QCD is weighted by w_sample, i.e. what one training epoch sees
(val / test / test_semilep: one fixed draw, w_sample = 1; test_semilep is compared with test/qcd). Writes every figure as
<plots>/<name>.pdf and <plots>/png/<name>.png; names ending in _log are the same figure with log y axes:
  kinematics_<split>[_log]   pT, mSD, |eta|, nConstituents, tau32, ParticleNet TvsQCD: top vs QCD (unit area)
  resampling_<split>[_log]   pT, mSD, |eta| of the tops and of QCD before (natural: w_phys, cross-section
                             weighted) and after the resampling (w_sample), and QCD / top (train, test)
  resampling_factor_train    QCD resampled / natural per pT bin and per (pT, mSD) cell: the weight the
                             resampling amounts to
  ptmsd_train                (pT, mSD) density of top and of QCD per epoch, and their ratio per cell
  qcd_composition_train      QCD per epoch by pT-hat sample, vs jet pT
  jet_rank                   fraction of leading AOJ jets vs pT
  roc_reference              ParticleNet TvsQCD and tau32 on test and test_semilep (reference taggers)
  plots.json                 the numbers: counts, QCD rejection at fixed top efficiency per pT bin, resampling factor
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import make_finetune_set as mf  # noqa: E402
from H5_skim_split import JT_COLS  # noqa: E402

TOP, QCD = "#2a78d6", "#eb6834"                                     # categorical slots 1, 2
SAMPLE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ec8f8e", "#b8302f"])
PT_EDGES = np.exp(np.linspace(np.log(300.), np.log(2500.), 31))
MSD_EDGES = np.arange(20., 500.1, 10.)
PT_BINS = [300, 400, 500, 600, 800, 1000, 2500]
WPS = [0.3, 0.5, 0.7]
COLS = ["jet_kinematics", "jet_tagging", "label", "sample", "jet_rank", "w_phys"]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 140,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})


def logy(ax):
    """Log y axis from half the smallest to 3x the largest positive value drawn (stairs and lines)."""
    v = [np.asarray(a.get_data().values, float) for a in ax.patches if hasattr(a, "get_data")]
    v = np.concatenate([x.ravel() for x in v + [np.asarray(l.get_ydata(), float) for l in ax.lines]] or [[0.]])
    v = v[np.isfinite(v) & (v > 0)]
    ax.set_yscale("log")
    if len(v):
        ax.set_ylim(v.min() / 2, v.max() * 3)


def save(fig, path, logy_axes=()):
    """<dir>/<name>.pdf and <dir>/png/<name>.png (path = <dir>/<name>); with logy_axes also <name>_log, the
    same figure with those axes on a log y scale."""
    d, name = os.path.split(path)
    os.makedirs(os.path.join(d, "png"), exist_ok=True)
    for suffix in [""] + (["_log"] if len(logy_axes) else []):
        if suffix:
            for ax in logy_axes:
                logy(ax)
        fig.savefig(os.path.join(d, name + suffix + ".pdf"))
        fig.savefig(os.path.join(d, "png", name + suffix + ".png"))
    plt.close(fig)

def load(setdir, split, cls):
    d = "test" if (split == "test_semilep" and cls == "qcd") else split
    parts = []
    for p in sorted(glob.glob(os.path.join(setdir, d, "%s_*.h5" % cls))):
        with h5py.File(p, "r") as f:
            q = {k: f[k][:] for k in COLS if k in f}
            q["w"] = f["w_sample"][:] if "w_sample" in f else f["weight"][:]
            parts.append(q)
    out = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    jt = out.pop("jet_tagging")
    out["ncons"] = jt[:, JT_COLS.index("nConstituents")]
    t2, t3 = jt[:, JT_COLS.index("tau2")], jt[:, JT_COLS.index("tau3")]
    out["tau32"] = np.where(t2 > 0, t3 / np.maximum(t2, 1e-9), 1.0)
    out["pn"] = jt[:, JT_COLS.index("particleNet_TvsQCD")]
    out["w"] = out["w"].astype(np.float64)
    out["w_phys"] = out["w_phys"].astype(np.float64)
    return out


def logpt(ax):
    ax.set_xscale("log")
    ax.set_xticks([300, 500, 1000, 2000])
    ax.set_xticklabels(["300", "500", "1000", "2000"])
    ax.xaxis.set_minor_formatter(NullFormatter())


def hist(x, edges, w):
    h, _ = np.histogram(x, edges, weights=w)
    return h / max(h.sum(), 1e-30)


def ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((a > 0) & (b > 0), a / b, np.nan)


def step(ax, x, edges, w, color, label, ls="-"):
    ax.stairs(hist(x, edges, w), edges, color=color, lw=1.8, ls=ls, label=label)


def kinematics(setname, split, t, q, path):
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2))
    panels = [("pt", "jet pT [GeV]", PT_EDGES, True), ("msd", "jet mSD [GeV]", MSD_EDGES, False),
              ("abseta", "jet |eta|", np.linspace(0, 2.5, 26), False),
              ("ncons", "nConstituents", np.arange(0, 151, 5), False),
              ("tau32", "tau32 = tau3 / tau2", np.linspace(0, 1, 41), False),
              ("pn", "ParticleNet TvsQCD (CMS)", np.linspace(0, 1, 41), False)]
    for ax, (k, xl, edges, logx) in zip(axs.flat, panels):
        get = lambda d: {"pt": d["jet_kinematics"][:, 0], "msd": d["jet_kinematics"][:, 3],
                         "abseta": np.abs(d["jet_kinematics"][:, 1])}.get(k, d.get(k))
        step(ax, get(t), edges, t["w"], TOP, "top")
        step(ax, get(q), edges, q["w"], QCD, "QCD" + (" (per epoch)" if split == "train" else ""))
        ax.set_xlabel(xl)
        ax.set_ylabel("fraction of jets")
        if logx:
            logpt(ax)
            ax.set_yscale("log")
    axs.flat[0].legend(loc="upper right")
    fig.suptitle("%s - %s: %d top, %.0f QCD%s" % (setname, split, len(t["w"]), q["w"].sum(),
                                                   " per epoch" if split == "train" else ""), color=INK)
    fig.tight_layout()
    save(fig, path, logy_axes=axs.flat)


def resampling(setname, split, t, q, path):
    """Tops and QCD before (natural: w_phys) and after the resampling (w_sample), and QCD / top."""
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.4), gridspec_kw=dict(height_ratios=[2, 1]))
    per = " per epoch" if split == "train" else ""
    for j, (col, xl, edges) in enumerate(((0, "jet pT [GeV]", PT_EDGES), (3, "jet mSD [GeV]", MSD_EDGES),
                                          (1, "jet |eta|", np.linspace(0, 2.5, 26)))):
        a, r = axs[0, j], axs[1, j]
        ht = hist(np.abs(t["jet_kinematics"][:, col]), edges, t["w"])
        hn = hist(np.abs(q["jet_kinematics"][:, col]), edges, q["w_phys"])
        hr = hist(np.abs(q["jet_kinematics"][:, col]), edges, q["w"])
        a.stairs(ht, edges, color=TOP, lw=1.8, label="top (not resampled)")
        a.stairs(hn, edges, color=QCD, lw=1.8, ls="--", label="QCD natural (w_phys)")
        a.stairs(hr, edges, color=QCD, lw=1.8, label="QCD resampled%s (w_sample)" % per)
        r.stairs(ratio(hn, ht), edges, color=QCD, lw=1.6, ls="--")
        r.stairs(ratio(hr, ht), edges, color=QCD, lw=1.6)
        r.axhline(1, color=INK2, lw=0.8)
        logy(r)
        a.set_ylabel("fraction of jets")
        r.set_ylabel("QCD / top")
        r.set_xlabel(xl)
        a.set_xlim(edges[0], edges[-1])
        r.set_xlim(edges[0], edges[-1])
        if col == 0:
            logpt(a)
            logpt(r)
            logy(a)
    axs[0, 0].legend(loc="lower left")
    fig.suptitle("%s - %s: QCD before (natural, cross-section weighted) and after the resampling" % (setname, split),
                 color=INK)
    fig.tight_layout()
    save(fig, path, logy_axes=axs[0])


def resampling_factor(setname, q, path):
    """QCD resampled / natural (both unit area) per pT bin and per (pT, mSD) cell: the weight it amounts to."""
    pt, msd = q["jet_kinematics"][:, 0], q["jet_kinematics"][:, 3]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw=dict(width_ratios=[1, 1.3]))
    f1 = ratio(hist(pt, PT_EDGES, q["w"]), hist(pt, PT_EDGES, q["w_phys"]))
    axs[0].stairs(f1, PT_EDGES, color=QCD, lw=1.8)
    axs[0].axhline(1, color=INK2, lw=0.8)
    logy(axs[0])
    axs[0].set_ylabel("QCD resampled / natural")
    axs[0].set_title("per pT bin", color=INK)
    hr, _, _ = np.histogram2d(pt, msd, (PT_EDGES, MSD_EDGES), weights=q["w"])
    hn, _, _ = np.histogram2d(pt, msd, (PT_EDGES, MSD_EDGES), weights=q["w_phys"])
    f2 = ratio(hr / hr.sum(), hn / hn.sum())
    lim = 10. ** min(3, np.ceil(np.nanmax(np.abs(np.log10(f2)))))
    m = axs[1].pcolormesh(PT_EDGES, MSD_EDGES, np.ma.masked_invalid(f2).T, cmap=DIVERGE, norm=LogNorm(1 / lim, lim))
    fig.colorbar(m, ax=axs[1], label="QCD resampled / natural")
    axs[1].set_ylabel("jet mSD [GeV]")
    axs[1].set_title("per (pT, mSD) cell (display bins)", color=INK)
    axs[1].grid(False)
    for ax in axs:
        logpt(ax)
        ax.set_xlim(PT_EDGES[0], PT_EDGES[-1])
        ax.set_xlabel("jet pT [GeV]")
    fig.suptitle("%s - train QCD: weight the resampling amounts to (> 1: more than in nature, < 1: fewer)" % setname,
                 color=INK)
    fig.tight_layout()
    save(fig, path)
    return dict(pt_edges=np.round(PT_EDGES, 1).tolist(), factor_vs_pt=[None if not np.isfinite(x) else round(float(x), 5)
                                                                      for x in f1])


def ptmsd(setname, t, q, path):
    ht, _, _ = np.histogram2d(t["jet_kinematics"][:, 0], t["jet_kinematics"][:, 3], (PT_EDGES, MSD_EDGES), weights=t["w"])
    hq, _, _ = np.histogram2d(q["jet_kinematics"][:, 0], q["jet_kinematics"][:, 3], (PT_EDGES, MSD_EDGES), weights=q["w"])
    ht, hq = ht / ht.sum(), hq / hq.sum()
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    vmin = max(min(ht[ht > 0].min(), hq[hq > 0].min()), 1e-6)
    for ax, h, title in ((axs[0], ht, "top"), (axs[1], hq, "QCD per epoch")):
        m = ax.pcolormesh(PT_EDGES, MSD_EDGES, np.ma.masked_equal(h, 0).T, cmap=BLUES, norm=LogNorm(vmin, max(ht.max(), hq.max())))
        ax.set_title(title + " (fraction of jets per cell)", color=INK)
        fig.colorbar(m, ax=ax)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((ht > 0) & (hq > 0), hq / ht, np.nan)
    m = axs[2].pcolormesh(PT_EDGES, MSD_EDGES, np.ma.masked_invalid(r).T, cmap=DIVERGE, norm=LogNorm(0.25, 4))
    axs[2].set_title("QCD / top (1 = matched; < 1: QCD short)", color=INK)
    cb = fig.colorbar(m, ax=axs[2], ticks=[0.25, 0.5, 1, 2, 4])
    cb.ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    for ax in axs:
        logpt(ax)
        ax.set_xlabel("jet pT [GeV]")
        ax.set_ylabel("jet mSD [GeV]")
        ax.grid(False)
    fig.suptitle("%s - train, display bins: 30 ln-pT x 10 GeV mSD" % setname, color=INK)
    fig.tight_layout()
    save(fig, path)


def composition(setname, q, path):
    names = np.array(mf.SAMPLES)[q["sample"].astype(int)]
    groups = [("QCD_Pt300", ["QCD_Pt300"]), ("QCD_Pt470", ["QCD_Pt470"]), ("QCD_Pt600", ["QCD_Pt600"]),
              ("QCD_Pt800", ["QCD_Pt800"]), ("QCD_Pt1000+", ["QCD_Pt1000", "QCD_Pt1400", "QCD_Pt1800", "QCD_Pt2400", "QCD_Pt3200"])]
    edges = np.exp(np.linspace(np.log(300.), np.log(2500.), 15))
    tot, _ = np.histogram(q["jet_kinematics"][:, 0], edges, weights=q["w"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bottom = np.zeros(len(tot))
    fr = {}
    for (g, members), col in zip(groups, SAMPLE_COLORS):
        m = np.isin(names, members)
        h, _ = np.histogram(q["jet_kinematics"][m, 0], edges, weights=q["w"][m])
        f = np.where(tot > 0, h / np.maximum(tot, 1e-30), 0)
        fr[g] = np.round(f, 4).tolist()
        ax.bar(edges[:-1], f, width=np.diff(edges) * 0.94, bottom=bottom, align="edge", color=col, label=g,
               edgecolor="#fcfcfb", linewidth=1.0)
        bottom += f
    logpt(ax)
    ax.set_xlabel("jet pT [GeV]")
    ax.set_ylabel("fraction of QCD jets (per epoch)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("%s\ntrain QCD per epoch by pT-hat sample (cross-section mix in each cell)" % setname, color=INK)
    fig.tight_layout()
    save(fig, path)
    return dict(pt_edges=np.round(edges, 1).tolist(), fractions=fr)


def jet_rank(setname, D, path):
    edges = np.exp(np.linspace(np.log(300.), np.log(2500.), 15))
    c = np.sqrt(edges[1:] * edges[:-1])
    fig, ax = plt.subplots(figsize=(7, 4))
    out = {}
    for (key, label, col, ls) in ((("train", "top"), "top, TTToHadronic (train)", TOP, "-"),
                                  (("train", "qcd"), "QCD (train, per epoch)", QCD, "-"),
                                  (("test_semilep", "top"), "top, TTToSemiLeptonic", TOP, "--")):
        d = D[key]
        pt = d["jet_kinematics"][:, 0]
        a, _ = np.histogram(pt, edges, weights=d["w"])
        b, _ = np.histogram(pt[d["jet_rank"] == 0], edges, weights=d["w"][d["jet_rank"] == 0])
        f = np.where(a > 0, b / np.maximum(a, 1e-30), np.nan)
        ax.plot(c, f, color=col, ls=ls, lw=1.8, marker="o", ms=4, label=label)
        out[label] = np.round(f, 3).tolist()
    logpt(ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("jet pT [GeV]")
    ax.set_ylabel("fraction that is the leading AOJ jet")
    ax.legend(loc="lower right")
    ax.set_title("%s\nleading AOJ jet of the event (jet_rank == 0)" % setname, color=INK)
    fig.tight_layout()
    save(fig, path)
    return dict(pt_centers=np.round(c, 1).tolist(), leading_fraction=out)


def roc(st, sq, wt, wq):
    """Top efficiency, QCD efficiency and number of QCD jets passing, for cuts score > x (weighted)."""
    s = np.r_[st, sq]
    y = np.r_[np.ones(len(st)), np.zeros(len(sq))]
    w = np.r_[wt, wq]
    o = np.argsort(-s, kind="stable")
    ct, cq, nq = np.cumsum(w[o] * y[o]), np.cumsum(w[o] * (1 - y[o])), np.cumsum(1 - y[o])
    return ct / ct[-1], cq / cq[-1], nq


def rejection(et, eq, wp):
    i = np.searchsorted(et, wp)
    i = min(i, len(eq) - 1)
    return 1.0 / max(eq[i], 1e-12)


def roc_plot(setname, D, path):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    res = {}
    for split, ls in (("test", "-"), ("test_semilep", "--")):
        t, q = D[(split, "top")], D[(split, "qcd")]
        for name, key, col, sign in (("ParticleNet TvsQCD", "pn", TOP, 1), ("tau32", "tau32", QCD, -1)):
            et, eq, nq = roc(sign * t[key], sign * q[key], t["w"], q["w"])
            sel = nq >= 20                           # at least 20 QCD jets pass: no statistics-limited tail
            ax.plot(et[sel], 1 / eq[sel], color=col, ls=ls, lw=1.8, label="%s, %s" % (name, split))
            r = {"all": {str(wp): round(rejection(et, eq, wp), 2) for wp in WPS}}
            for lo, hi in zip(PT_BINS[:-1], PT_BINS[1:]):
                mt = (t["jet_kinematics"][:, 0] >= lo) & (t["jet_kinematics"][:, 0] < hi)
                mq = (q["jet_kinematics"][:, 0] >= lo) & (q["jet_kinematics"][:, 0] < hi)
                if mt.sum() > 100 and mq.sum() > 100:
                    a, b, _ = roc(sign * t[key][mt], sign * q[key][mq], t["w"][mt], q["w"][mq])
                    r["%d-%d" % (lo, hi)] = {str(wp): round(rejection(a, b, wp), 2) for wp in WPS}
            res["%s / %s" % (name, split)] = r
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("top efficiency")
    ax.set_ylabel("QCD rejection (1 / QCD efficiency)")
    ax.legend(loc="upper right")
    ax.set_title("%s\nreference taggers; QCD = test/qcd; drawn where >= 20 QCD jets pass" % setname, color=INK)
    fig.tight_layout()
    save(fig, path)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("setdir")
    ap.add_argument("-o", "--outdir", default=None)
    a = ap.parse_args(argv)
    out = a.outdir or os.path.join(a.setdir, "plots")
    os.makedirs(out, exist_ok=True)
    setname = os.path.basename(os.path.normpath(a.setdir))
    D = {(s, c): load(a.setdir, s, c) for s in mf.SPLITS for c in ("top", "qcd")}
    summary = {"counts": {"%s/%s" % k: dict(stored=len(d["w"]), per_epoch=round(float(d["w"].sum()), 1))
                          for k, d in D.items()}}
    for s in mf.SPLITS:
        kinematics(setname, s, D[(s, "top")], D[(s, "qcd")], os.path.join(out, "kinematics_%s" % s))
    for s in ("train", "test"):
        resampling(setname, s, D[(s, "top")], D[(s, "qcd")], os.path.join(out, "resampling_%s" % s))
    summary["qcd_resampling_factor_train"] = resampling_factor(setname, D[("train", "qcd")],
                                                               os.path.join(out, "resampling_factor_train"))
    ptmsd(setname, D[("train", "top")], D[("train", "qcd")], os.path.join(out, "ptmsd_train"))
    summary["qcd_composition_train"] = composition(setname, D[("train", "qcd")], os.path.join(out, "qcd_composition_train"))
    summary["jet_rank"] = jet_rank(setname, D, os.path.join(out, "jet_rank"))
    summary["qcd_rejection"] = roc_plot(setname, D, os.path.join(out, "roc_reference"))
    with open(os.path.join(out, "plots.json"), "w") as fo:
        json.dump(summary, fo, indent=1)
    print(json.dumps(summary["qcd_rejection"], indent=1))
    print("plots in %s: %s; png/: %d files" % (out, ", ".join(sorted(f for f in os.listdir(out) if f != "png")),
                                               len(os.listdir(os.path.join(out, "png")))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
