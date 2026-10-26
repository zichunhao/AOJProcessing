#!/usr/bin/env python3
"""Checks of an evaluation set made by make_eval_ttbar.py (one part: mc or data), against its inputs.

    bash ttbar_semilep/py3env.sh finetune/check_eval_ttbar.py SETDIR --part mc [--an-dir <store>/ttbar] [--rows 20]

- per sample: the probes equal those of the inputs that pass the window, the channel and the stage, recounted here
- no (sample, run, lumi, event) twice
- label = (truth category == t) for MC, -1 for data; w_phys = genWeight x xsec x L / sum_genweight (MC), 1 (data)
- --rows random probes per sample traced to their input row (source): PFCands (rebuilt from the input's 11 columns),
  jet_kinematics, jet_tagging, event_info, ttbar_tag, truth, ak4
- MC, with --an-dir: in every input, the probes flagged an & btag are exactly the events that the 'an'-preset
  production (<an-dir>/tt_<chunk>.h5, the same chunk) has at stage >= btag with its lepton path fired (it also
  stores JetHT-only events), inside the window and channel
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
import make_eval_ttbar as me  # noqa: E402
import samples as smp  # noqa: E402


def load(shards, keys):
    out = {}
    for k in keys:
        out[k] = np.concatenate([h5py.File(p, "r")[k][:] for p in shards])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("setdir")
    ap.add_argument("--part", choices=["mc", "data"], required=True)
    ap.add_argument("--an-dir", default="/ceph/cms/store/user/zichun/parcel/AOJ/ttbar",
                    help="the 'an'-preset production, for the an-flag check (MC; 'none' to skip)")
    ap.add_argument("--rows", type=int, default=20, help="random probes per sample traced to their input")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    S = json.load(open(os.path.join(a.setdir, "summary_%s.json" % a.part)))
    shards = sorted(glob.glob(os.path.join(a.setdir, a.part, "%s_*.h5" % a.part)))
    fails = []

    def check(ok, msg):
        print(("ok    " if ok else "FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    with h5py.File(shards[0], "r") as f:
        names = f["sample"].attrs["names"].split(",")
        cols = {k: me.cols_of(f[k]) for k in ("ttbar_tag", "truth", "ak4", "cr")}
    D = load(shards, ["sample", "event_info", "label", "w_phys", "cr", "source", "truth", "jet_kinematics"])
    n = len(D["sample"])
    check(n == S["written"]["probes"], "%d probes in %d shards (summary %d)" % (n, len(shards), S["written"]["probes"]))
    key = np.concatenate([D["sample"][:, None].astype(np.int64), D["event_info"]], axis=1)
    check(len(np.unique(key, axis=0)) == n, "no (sample, run, lumi, event) twice")

    src_dir = S["src"]
    files = S["files"]
    sel = dict(pt_max=S["selection"]["pt_max"], msd=tuple(S["selection"]["msd"]))
    rng = np.random.default_rng(a.seed)
    an_bad = an_n = an_files = 0
    for label, e in S["samples"].items():
        code = names.index(label)
        m = D["sample"] == code
        s = smp.SAMPLES[label]
        used = sorted(set(D["source"][m, 0].tolist()))
        # recount from the inputs with the same file set (all of the label's files, or the first --max-files)
        lab_files = [i for i, p in enumerate(files) if me.label_of(p) == label]
        if S.get("max_files"):
            lab_files = lab_files[:S["max_files"]]
        cnt, gw_sum = 0, 0.0
        for i in lab_files:
            with h5py.File(os.path.join(src_dir, files[i]), "r") as f:
                if f["event_info"].shape[0] == 0:
                    continue
                c = me.cols_of(f["ttbar_tag"])
                T = f["ttbar_tag"][:]
                ch, stage = T[:, c.index("channel")], T[:, c.index("stage")]
                jk = f["jet_kinematics"][:]
                stages = f["cutflow"].attrs["stages"].split(",")
                k = ((jk[:, 0] < sel["pt_max"]) & (jk[:, 3] > sel["msd"][0]) & (jk[:, 3] < sel["msd"][1])
                     & np.isin(ch, s["channels"]) & (stage >= stages.index(S["min_stage"])))
                cnt += int(k.sum())
                if a.part == "mc":
                    gw_sum += float(f["gen_weight"][:][k].astype(np.float64).sum())
                if a.part == "mc" and a.an_dir != "none":
                    an_path = os.path.join(a.an_dir, files[i])
                    if os.path.exists(an_path):
                        with h5py.File(an_path, "r") as g:
                            ca = me.cols_of(g["ttbar_tag"])
                            Ta = g["ttbar_tag"][:]
                            jka = g["jet_kinematics"][:]
                            cha = Ta[:, ca.index("channel")]
                            # the 'an' production stores lepton OR JetHT-triggered events; the flag means the
                            # lepton route (the CR data are SingleMuon / SingleElectron): its lepton path must fire
                            trg = np.where(cha == 1, Ta[:, ca.index("trig_mu")], np.where(cha == 2, Ta[:, ca.index("trig_el")], 0)) > 0
                            ka = ((Ta[:, ca.index("stage")] >= stages.index("btag")) & np.isin(cha, s["channels"]) & trg
                                  & (jka[:, 0] < sel["pt_max"]) & (jka[:, 3] > sel["msd"][0]) & (jka[:, 3] < sel["msd"][1]))
                            ev_an = set(map(tuple, g["event_info"][:][ka].tolist()))
                        mi = m & (D["source"][:, 0] == i) & (D["cr"][:, 0] == 1) & (D["cr"][:, 6] == 1)
                        ev_ev = set(map(tuple, D["event_info"][mi].tolist()))
                        an_files += 1
                        an_n += len(ev_an)
                        an_bad += len(ev_an ^ ev_ev)
        check(cnt == int(m.sum()) == e["kept"], "%-20s %7d probes = %d recounted from %d inputs (summary %d)" % (
            label, m.sum(), cnt, len(lab_files), e["kept"]))
        if a.part == "mc":
            want = s["xsec_pb"] * S["lumi_fb"] * 1e3 * gw_sum / e["sum_genweight"]
            got = float(D["w_phys"][m].astype(np.float64).sum())
            check(abs(got - want) <= 1e-4 * max(abs(want), 1e-9), "%-20s sum w_phys %.4g = xsec x L x sum(genWeight) / sum_genweight %.4g" % (label, got, want))
            tc = cols["truth"].index("category")
            check(np.array_equal(D["label"][m], (D["truth"][m, tc] == me.TOP_CATEGORY).astype(np.int8)),
                  "%-20s label = truth category t (%d tops)" % (label, (D["label"][m] == 1).sum()))
        else:
            check((D["label"][m] == -1).all() and (D["w_phys"][m] == 1).all(), "%-20s label -1, w_phys 1" % label)
        # random rows against their input row
        idx = np.flatnonzero(m)
        pick = np.sort(rng.choice(idx, min(a.rows, len(idx)), replace=False)) if len(idx) else []
        good = 0
        for r in pick:
            sh, off = 0, r
            sizes = [h5py.File(p, "r")["sample"].shape[0] for p in shards]
            while off >= sizes[sh]:
                off -= sizes[sh]
                sh += 1
            fi, row = D["source"][r]
            with h5py.File(shards[sh], "r") as fo, h5py.File(os.path.join(src_dir, files[fi]), "r") as fs:
                jk = fs["jet_kinematics"][row:row + 1]
                ok = np.array_equal(fo["PFCands"][off:off + 1], mf.skim_pf(fs["PFCands"][row:row + 1], jk))
                ok &= np.array_equal(fo["jet_kinematics"][off], jk[0])
                ok &= np.array_equal(fo["jet_tagging"][off], fs["jet_tagging"][row])
                ok &= np.array_equal(fo["event_info"][off], fs["event_info"][row])
                t_src = fs["ttbar_tag"][row]
                ok &= np.array_equal(fo["ttbar_tag"][off][:len(t_src)], t_src, equal_nan=True)
                ok &= bool(np.isnan(fo["ttbar_tag"][off][len(t_src):]).all())
                if "truth" in fs:
                    tr = fs["truth"][row]
                    ok &= np.array_equal(fo["truth"][off][:len(tr)], tr, equal_nan=True)
                if "ak4" in fs:
                    ok &= np.array_equal(fo["ak4"][off], fs["ak4"][row], equal_nan=True)
                else:
                    ok &= bool(np.isnan(fo["ak4"][off]).all())
            good += bool(ok)
        check(good == len(pick), "%-20s %d of %d random probes equal to their input row (PFCands rebuilt)" % (label, good, len(pick)))
        if a.part == "data" and e.get("lumi_fraction") is not None:
            print("      %-20s %d certified lumi sections = %.4f of the era's" % (label, e["lumis_certified"], e["lumi_fraction"]))
    if a.part == "mc" and a.an_dir != "none":
        check(an_files > 0 and an_bad == 0, "an & btag flags = the 'an' production at stage >= btag: %d events in %d inputs, %d differ"
              % (an_n, an_files, an_bad))
    flags = D["cr"].astype(bool)
    c = cols["cr"]
    nest = [("tau32", "btag"), ("btag", "dphi"), ("iso30_met20", "skim")]
    check(all(not (flags[:, c.index(x)] & ~flags[:, c.index(y)]).any() for x, y in nest),
          "flags nest: tau32 => btag => dphi, iso30_met20 => skim")
    print("ALL OK" if not fails else "%d FAILED" % len(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
