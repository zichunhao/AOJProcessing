#!/usr/bin/env python3
"""Electron trigger scale factor for the CR (Ele27_WPTight | Ele32_WPTight, the isolated electron paths), measured
with the orthogonal-trigger method in the SF samples, since no POG SF exists for it in 2016.

    bash ttbar_semilep/py3env.sh finetune/etrig_sf.py -o el_trig_sf.json [--src <store>/sf_samples_20261026_skim]

Reference events: the tag lepton is an electron with the "an" ID + isolation (lep_id_iso) and pT > 30, the event
has a loose muon above 30 GeV (n_veto_mu >= 1) and fired IsoMu24 | IsoTkMu24 (trig_mu_an), at stage >= one_probe
(an AK8 jet above 300 GeV away from the lepton, MET > 20: the CR topology). Data: SingleMuon 2016G+H (orthogonal
to the CR: these events have two veto leptons). MC: every CR MC sample but the QCD ones, weighted by
genWeight x xsec / sum_genweight. Efficiency = fraction with trig_el_an, in bins of electron pT x abs(SC eta);
SF = eff(data) / eff(MC); a pT row with fewer than MIN_REF data reference events in one of its abs(eta) bins takes
its abs(eta)-inclusive SF in both (the statistics are ~1,550 data events). Uncertainty: data Clopper-Pearson (68 %) and MC statistics, plus the MC closure
|eff(reference) / eff(all) - 1| (the electron efficiency of MC events without the reference requirement) as a
systematic, added in quadrature.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import h5py
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ttbar_semilep"))
import samples as smp  # noqa: E402

PT = [30., 50., 70., 100., 200., 1e5]
ETA = [0., 1.479, 2.5]
MIN_REF = 40
COLS = ["channel", "stage", "lep_id_iso", "lep_pt", "lep_sc_eta", "n_veto_mu", "trig_mu_an", "trig_el_an"]


def read(args):
    """The reference-candidate rows (electron tag with ID + isolation, a loose muon, stage >= one_probe) of one input,
    and its sum_genweight."""
    path, label = args
    with h5py.File(path, "r") as f:
        sw = float(f.attrs["sum_genweight"])
        if f["event_info"].shape[0] == 0:
            return label, sw, None
        c = f["ttbar_tag"].attrs["columns"].split(",")
        T = f["ttbar_tag"][:]
        d = dict((k, T[:, c.index(k)].astype(np.float64)) for k in COLS)
        base = ((d["channel"] == 2) & (d["lep_id_iso"] > 0) & (d["lep_pt"] > 30) & (d["n_veto_mu"] >= 1)
                & (d["stage"] >= f["cutflow"].attrs["stages"].split(",").index("one_probe")))
        d = dict((k, v[base]) for k, v in d.items())
        d["gw"] = f["gen_weight"][:][base].astype(np.float64) if "gen_weight" in f else np.ones(int(base.sum()))
    return label, sw, d


def hist(d, w, m):
    return np.histogram2d(d["lep_pt"][m], np.abs(d["lep_sc_eta"][m]), bins=[PT, ETA], weights=w[m])[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--src", default="/ceph/cms/store/user/zichun/parcel/AOJ/sf_samples_20261026_skim")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    data_labels = [l for l in smp.SAMPLES if l.startswith("SingleMuon_")]
    mc_labels = [l for l, s in smp.SAMPLES.items() if s["cr"] and smp.is_mc(l) and s["group"] not in ("qcd", "qcd_mu")]
    tasks = [(p, l) for l in data_labels + mc_labels for p in sorted(glob.glob(os.path.join(a.src, "tt_%s_sub*.h5" % l)))]
    with Pool(a.workers) as pool:
        R = pool.map(read, tasks, chunksize=8)
    sumw = {}
    for label, sw, _ in R:
        sumw[label] = sumw.get(label, 0.0) + sw
    shape = (len(PT) - 1, len(ETA) - 1)
    H = dict((k, np.zeros(shape)) for k in ("d_ref", "d_pass", "m_ref", "m_pass", "m_ref_w2", "m_pass_w2", "m_all", "m_all_pass"))
    for label, _, d in R:
            if d is None:
                continue
            base = np.ones(len(d["channel"]), bool)
            ref = base & (d["trig_mu_an"] > 0)
            pas = d["trig_el_an"] > 0
            if label in data_labels:
                one = np.ones(len(base))
                H["d_ref"] += hist(d, one, ref)
                H["d_pass"] += hist(d, one, ref & pas)
            else:
                w = d["gw"] * smp.SAMPLES[label]["xsec_pb"] / sumw[label]
                H["m_ref"] += hist(d, w, ref)
                H["m_pass"] += hist(d, w, ref & pas)
                H["m_ref_w2"] += hist(d, w * w, ref)
                H["m_pass_w2"] += hist(d, w * w, ref & pas)
                H["m_all"] += hist(d, w, base)
                H["m_all_pass"] += hist(d, w, base & pas)
    incl = dict(eff_data=H["d_pass"].sum() / H["d_ref"].sum(), eff_mc=H["m_pass"].sum() / H["m_ref"].sum(),
                data_ref=int(H["d_ref"].sum()), eff_mc_all=H["m_all_pass"].sum() / H["m_all"].sum())
    incl["sf"] = incl["eff_data"] / incl["eff_mc"]
    merged = [i for i in range(len(PT) - 1) if H["d_ref"][i].min() < MIN_REF]
    for i in merged:                                   # sparse rows: abs(eta)-inclusive
        for key in H:
            H[key][i, :] = H[key][i].sum()
    n, k = H["d_ref"], H["d_pass"]
    e_d = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    lo = np.where(k > 0, beta.ppf(0.16, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(0.84, k + 1, n - k), 1.0)
    e_m = H["m_pass"] / H["m_ref"]
    # weighted-binomial MC uncertainty: var(eff) = sum over events of w^2 (pass - eff)^2 / (sum w)^2
    var_m = (H["m_pass_w2"] * (1 - e_m) ** 2 + (H["m_ref_w2"] - H["m_pass_w2"]) * e_m ** 2) / H["m_ref"] ** 2
    e_m_all = H["m_all_pass"] / H["m_all"]
    sf = e_d / e_m
    rel_d_up, rel_d_dn = (hi - e_d) / e_d, (e_d - lo) / e_d
    rel_m = np.sqrt(var_m) / e_m
    clos = np.abs(e_m / e_m_all - 1)
    up = sf * np.sqrt(rel_d_up ** 2 + rel_m ** 2 + clos ** 2)
    dn = sf * np.sqrt(rel_d_dn ** 2 + rel_m ** 2 + clos ** 2)
    out = dict(doc=__doc__, pt_edges=PT, abseta_edges=ETA, sf=sf.tolist(), sf_up=(sf + up).tolist(), sf_down=(sf - dn).tolist(),
               eff_data=e_d.tolist(), eff_data_lo=lo.tolist(), eff_data_hi=hi.tolist(), eff_mc=e_m.tolist(),
               eff_mc_stat=np.sqrt(var_m).tolist(), eff_mc_all=e_m_all.tolist(), closure=clos.tolist(),
               data_ref=n.astype(int).tolist(), data_pass=k.astype(int).tolist(), eta_inclusive_pt_rows=merged,
               min_ref=MIN_REF)
    out["inclusive"] = incl
    json.dump(out, open(a.out, "w"), indent=1)
    print("inclusive: data %.4f (%d events), MC %.4f, SF %.4f; MC without the reference trigger %.4f" % (
        incl["eff_data"], incl["data_ref"], incl["eff_mc"], incl["sf"], incl["eff_mc_all"]))
    print("%-12s %-10s %6s %15s %8s %8s %8s %18s" % ("pT", "|eta_SC|", "data", "eff data", "eff MC", "closure", "SF", "SF down / up"))
    for i in range(len(PT) - 1):
        for j in range(len(ETA) - 1):
            print("%-12s %-10s %6d %6.3f [%.3f,%.3f] %8.3f %8.3f %8.3f %8.3f / %.3f" % (
                "%g-%g" % (PT[i], PT[i + 1]) if PT[i + 1] < 1e4 else "> %g" % PT[i], "%g-%g" % (ETA[j], ETA[j + 1]), n[i, j],
                e_d[i, j], lo[i, j], hi[i, j], e_m[i, j], clos[i, j], sf[i, j], sf[i, j] - dn[i, j], sf[i, j] + up[i, j]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
