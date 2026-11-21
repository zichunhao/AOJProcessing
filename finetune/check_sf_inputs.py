#!/usr/bin/env python3
"""Checks of the per-sample SF inputs made by make_sf_inputs.py, against the SF samples.

    bash ttbar_semilep/py3env.sh finetune/check_sf_inputs.py OUTDIR [--an-dir <store>/ttbar] [--rows 20]

Per sample:
- the stored rows are exactly the input rows in the window, channel and stage >= one_probe that pass an_anyak8 or
  skim_iso, with every selection flag recomputed here from the ttbar_tag columns (written out independently of
  make_eval_ttbar.cr_flags)
- label = truth category t (MC), -1 (data); data: weights all 1; MC: sum(w_norm) = xsec x L x sum(genWeight of the
  stored rows) / sum_genweight; weight = the product of the nominal columns
- --rows random rows traced to their input row (PFCands rebuilt, every other dataset equal), with their lepton,
  pileup, b-tag and L1 pre-firing weights recomputed here row by row (scalar correctionlib calls, the b-tag weight from btag_eff.json)
- flags nest: an => an_anyak8, iso30 => iso30_met20 => skim_iso, tau32 => btag => dphi
Data: SingleMuon rows are mu, SingleElectron rows e; no (run, lumi, event) twice in a sample or in both primary
datasets. MC, with --an-dir: for every sample the 'an'-preset production also has, the events flagged an & btag equal its
events at stage >= btag with the lepton path fired, in the window and channel (whole samples: its chunks are
mostly 2 files per job, these 3).
Prints ALL OK at the end, or the failures.
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
import make_finetune_set as mf  # noqa: E402
import make_sf_inputs as ms  # noqa: E402
import samples as smp  # noqa: E402


def my_flags(d, stages):
    """The selection flags, from the definitions in the docstring of make_sf_inputs.py."""
    ch = d["channel"]
    trig_an = np.where(ch == 1, d["trig_mu_an"] > 0, np.where(ch == 2, d["trig_el_an"] > 0, False))
    trig_sk = np.where(ch == 1, d["trig_mu"] > 0, np.where(ch == 2, d["trig_el"] > 0, False))
    lep = (d["lep_id_iso"] > 0) & (d["lep_pt"] > 30) & (d["n_veto_mu"] + d["n_veto_el"] == 1)
    met = d["met_pt"]
    an_anyak8 = (d["an_channel"] == ch) & trig_an & (met > 50)
    return dict(an=an_anyak8 & (d["n_ak8"] == 1), an_anyak8=an_anyak8, iso30=lep & trig_an & (met > 50),
                iso30_met20=lep & trig_an & (met > 20), skim_iso=lep & trig_sk & (met > 20),
                dphi=d["stage"] >= stages.index("dphi"), btag=d["stage"] >= stages.index("btag"),
                tau32=d["stage"] >= stages.index("tau32"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--an-dir", default="/ceph/cms/store/user/zichun/parcel/AOJ/ttbar",
                    help="the 'an'-preset production for the an-flag check ('none' to skip)")
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    S = json.load(open(os.path.join(a.outdir, "summary.json")))
    E = json.load(open(os.path.join(a.outdir, "btag_eff.json")))
    eff = np.array(E["eff"])
    et_path = os.path.join(a.outdir, "el_trig_sf.json")
    ET = json.load(open(et_path)) if os.path.exists(et_path) else None
    fails = []
    rng = np.random.default_rng(a.seed)
    C = ms.corr()

    def check(ok, msg):
        print(("ok    " if ok else "FAIL  ") + msg, flush=True)
        if not ok:
            fails.append(msg)

    sel = dict(pt_max=S["selection"]["pt_max"], msd=tuple(S["selection"]["msd"]))
    L = S["lumi_fb"]
    ev_pd = {}
    an_files = an_n = an_bad = 0
    for label in S["samples"]:
        s = smp.SAMPLES[label]
        mc = smp.is_mc(label)
        with h5py.File(os.path.join(a.outdir, label + ".h5"), "r") as f:
            files = json.loads(f.attrs["files"])
            src = str(f.attrs["src"])
            n = f["weight"].shape[0] if "weight" in f else 0
            D = dict((k, f[k][:]) for k in ("event_info", "ttbar_tag", "sel", "weights", "weight", "label", "source")
                     if k in f) if n else {}
            if n and mc:
                D["truth"] = f["truth"][:]
            sc = f["sel"].attrs["columns"].split(",") if n else ms.SEL_COLS
            wc = f["weights"].attrs["columns"].split(",") if n else ms.W_COLS
            tcols = f["ttbar_tag"].attrs["columns"].split(",") if n else None
            sumw_attr = float(f.attrs["sum_genweight"]) if mc else None
        # recount from the inputs, flags recomputed
        cnt, gw_sum, flag_bad, an_ok_here = 0, 0.0, 0, True
        want_src = []
        for i, fn in enumerate(files):
            with h5py.File(os.path.join(src, fn), "r") as g:
                if g["event_info"].shape[0] == 0:
                    continue
                c = g["ttbar_tag"].attrs["columns"].split(",")
                T = g["ttbar_tag"][:].astype(np.float64)
                d = dict((k, T[:, j]) for j, k in enumerate(c))
                stages = g["cutflow"].attrs["stages"].split(",")
                F = my_flags(d, stages)
                jk = g["jet_kinematics"][:]
                k = (mf.in_sel(jk, sel) & np.isin(d["channel"], s["channels"]) & (d["stage"] >= stages.index("one_probe"))
                     & (F["an_anyak8"] | F["skim_iso"]))
                rows = np.flatnonzero(k)
                cnt += len(rows)
                want_src.append(np.stack([np.full(len(rows), i), rows], axis=1))
                if mc:
                    gw_sum += float(g["gen_weight"][:][rows].astype(np.float64).sum())
                if n:
                    m = D["source"][:, 0] == i
                    got = D["sel"][m]
                    exp = np.stack([F[x][D["source"][m, 1]] for x in sc], axis=1).astype(np.int8)
                    flag_bad += int((got != exp).any(axis=1).sum())
        want_src = np.concatenate(want_src) if want_src else np.zeros((0, 2), np.int64)
        old = sorted(glob.glob(os.path.join(a.an_dir, "tt_%s_sub*.h5" % label))) if mc and a.an_dir != "none" else []
        if old and n:
            # the 'an' production used other chunk sizes (2 files per job for most samples): compare whole samples
            ev_an = set()
            for op in old:
                with h5py.File(op, "r") as h:
                    if h["event_info"].shape[0] == 0:
                        continue
                    ca = h["ttbar_tag"].attrs["columns"].split(",")
                    Ta = h["ttbar_tag"][:]
                    st_a = h["cutflow"].attrs["stages"].split(",")
                    cha = Ta[:, ca.index("channel")]
                    trg = np.where(cha == 1, Ta[:, ca.index("trig_mu")], np.where(cha == 2, Ta[:, ca.index("trig_el")], 0)) > 0
                    ka = (trg & (Ta[:, ca.index("stage")] >= st_a.index("btag")) & np.isin(cha, s["channels"])
                          & mf.in_sel(h["jet_kinematics"][:], sel))
                    ev_an |= set(map(tuple, h["event_info"][:][ka].tolist()))
            m = (D["sel"][:, sc.index("an")] > 0) & (D["sel"][:, sc.index("btag")] > 0)
            ev_me = set(map(tuple, D["event_info"][m].tolist()))
            an_files += 1
            an_n += len(ev_an)
            an_bad += len(ev_an ^ ev_me)
            check(ev_an == ev_me, "%-22s an & btag = the 'an' production (%d inputs): %d events, %d differ"
                  % (label, len(old), len(ev_an), len(ev_an ^ ev_me)))
        same_rows = n == cnt and (n == 0 or np.array_equal(D["source"], want_src))
        check(same_rows, "%-22s %7d rows = the %d recounted from %d inputs, same rows in order" % (label, n, cnt, len(files)))
        if not n:
            continue
        check(flag_bad == 0, "%-22s selection flags = recomputed (%d rows differ)" % (label, flag_bad))
        wn = D["weights"][:, wc.index("w_norm")].astype(np.float64)
        prod = np.prod([D["weights"][:, wc.index(x)].astype(np.float64) for x in ms.NOMINAL_PRODUCT], axis=0)
        check(np.allclose(D["weight"], prod, rtol=1e-5, atol=0), "%-22s weight = %s" % (label, " x ".join(ms.NOMINAL_PRODUCT)))
        if mc:
            want = s["xsec_pb"] * L * 1e3 * gw_sum / sumw_attr
            check(abs(wn.sum() - want) <= 1e-4 * abs(want), "%-22s sum w_norm %.6g = xsec x L x sum(genWeight) / sum_genweight %.6g" % (label, wn.sum(), want))
            check(np.array_equal(D["label"], (D["truth"][:, 0] == 3).astype(np.int8)), "%-22s label = truth category t (%d)" % (label, (D["label"] == 1).sum()))
        else:
            ch = D["ttbar_tag"][:, tcols.index("channel")]
            want_ch = 1 if label.startswith("SingleMuon") else 2
            check((D["label"] == -1).all() and (D["weights"] == 1).all() and (D["weight"] == 1).all() and (ch == want_ch).all(),
                  "%-22s label -1, weights 1, channel %d only" % (label, want_ch))
            ev_pd[label] = set(map(tuple, D["event_info"].tolist()))
        key = D["event_info"]
        check(len(np.unique(key, axis=0)) == n, "%-22s no (run, lumi, event) twice" % label)
        sc_i = dict((x, sc.index(x)) for x in sc)
        fl = D["sel"].astype(bool)
        nest = [("an", "an_anyak8"), ("iso30", "iso30_met20"), ("iso30_met20", "skim_iso"), ("tau32", "btag"), ("btag", "dphi")]
        check(all(not (fl[:, sc_i[x]] & ~fl[:, sc_i[y]]).any() for x, y in nest), "%-22s flags nest" % label)
        # random rows: datasets equal to the input, SFs recomputed
        pick = np.sort(rng.choice(n, min(a.rows, n), replace=False))
        good = sf_good = 0
        with h5py.File(os.path.join(a.outdir, label + ".h5"), "r") as fo:
            for r in pick:
                fi, row = D["source"][r]
                with h5py.File(os.path.join(src, files[fi]), "r") as g:
                    jk = g["jet_kinematics"][row:row + 1]
                    ok = np.array_equal(fo["PFCands"][r:r + 1], mf.skim_pf(g["PFCands"][row:row + 1], jk))
                    for k in ("jet_kinematics", "jet_tagging", "event_info", "ttbar_tag", "ak4") + (("truth",) if mc else ()):
                        ok &= np.array_equal(fo[k][r], g[k][row], equal_nan=True)
                    good += bool(ok)
                    if not mc:
                        sf_good += 1
                        continue
                    t = dict(zip(tcols, fo["ttbar_tag"][r].astype(np.float64)))
                    w = dict(zip(wc, fo["weights"][r].astype(np.float64)))
                    exp = dict(sf_pu=C["pu"].evaluate(float(t["pu_ntrueint"]), "nominal"))
                    if t["channel"] == 1:
                        exp.update(sf_lep_id=C["mu_id"].evaluate(t["lep_eta"], t["lep_pt"], "nominal"),
                                   sf_lep_iso=C["mu_iso"].evaluate(t["lep_eta"], t["lep_pt"], "nominal"), sf_lep_reco=1.0,
                                   sf_trig=C["mu_trig"].evaluate(t["lep_eta"], t["lep_pt"], "nominal"))
                    else:
                        st = 1.0
                        if ET is not None:
                            i = min(max(int(np.searchsorted(ET["pt_edges"], t["lep_pt"], side="right")) - 1, 0), len(ET["pt_edges"]) - 2)
                            j = min(max(int(np.searchsorted(ET["abseta_edges"], abs(t["lep_sc_eta"]), side="right")) - 1, 0), len(ET["abseta_edges"]) - 2)
                            st = ET["sf"][i][j]
                        exp.update(sf_lep_id=C["el"].evaluate("2016postVFP", "sf", "wp80iso", t["lep_sc_eta"], t["lep_pt"]),
                                   sf_lep_iso=1.0, sf_trig=st,
                                   sf_lep_reco=C["el"].evaluate("2016postVFP", "sf", "RecoAbove20", t["lep_sc_eta"], t["lep_pt"]))
                    bw = 1.0
                    for j in fo["ak4"][r]:
                        pt, eta, sc_b, hf, drp, drl = j[0], j[1], j[3], j[4], j[5], j[6]
                        if not (pt > 40 and abs(eta) < 2.4 and drp > 1.5 and drl > 0.5):
                            continue
                        hf = int(hf) if hf in (4, 5) else 0
                        cb = C["b_bc"] if hf else C["b_light"]
                        sfj = cb.evaluate("central", "M", hf, float(abs(eta)), float(pt))
                        ip = min(max(np.digitize(pt, E["pt_edges"]) - 1, 0), len(E["pt_edges"]) - 2)
                        ie = min(max(np.digitize(abs(eta), E["abseta_edges"]) - 1, 0), len(E["abseta_edges"]) - 2)
                        e = eff[[5, 4, 0].index(hf), ip, ie]
                        bw *= sfj if sc_b > ms.BTAG_WP else max(1 - sfj * e, 0.0) / (1 - e)
                    exp["sf_btag"] = bw
                    if "sf_l1pf" in w:
                        x = t.get("l1pf_nom", np.nan)
                        exp["sf_l1pf"] = x if np.isfinite(x) else 1.0
                    sf_good += all(abs(w[k] - v) <= 1e-5 * max(1.0, abs(v)) for k, v in exp.items())
        check(good == len(pick), "%-22s %d of %d random rows equal to their input row (PFCands rebuilt)" % (label, good, len(pick)))
        check(sf_good == len(pick), "%-22s %d of %d random rows: pileup, lepton, trigger, b-tag and pre-firing weights recomputed" % (label, sf_good, len(pick)))
    for era in ("2016G", "2016H"):
        mu, el = ev_pd.get("SingleMuon_" + era), ev_pd.get("SingleElectron_" + era)
        if mu is not None and el is not None:
            check(not (mu & el), "%s: no event stored from both SingleMuon and SingleElectron" % era)
    if a.an_dir != "none":
        check(an_files > 0 and an_bad == 0, "an & btag = the 'an'-preset production at stage >= btag (lepton path): %d events in "
              "%d samples, %d differ" % (an_n, an_files, an_bad))
    print("ALL OK" if not fails else "%d FAILED" % len(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
