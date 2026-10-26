#!/usr/bin/env python3
"""Tag-and-probe MC vs the fine-tuning signal: truth agreement, overlap, and an event veto list.

    python3 eval_overlap.py STORE [-o eval_events_TT.h5]

STORE holds merged_<label>.h5 (H5_maker --top_label, all AOJ jets), merged_<label>_topmatched.h5 (top_match == 2,
the fine-tuning signal) and ttbar/tt_<label>_sub*.h5 (ttbar_maker, made from the same NanoAOD in the same jobs).
For TTToSemiLeptonic and TTToHadronic:
  1. every stored probe is looked up in the all-jets file by (lumi, event, pT), and its ttbar_maker truth category
     is compared with H5_maker's top_match (expected: t <-> 2, W <-> 1, bq/other <-> 0);
  2. the fraction of probes, per stage, that are also fine-tuning signal jets;
  3. with -o: (run, lumi, event) of every stored event, and the fine-tuning jets that removing them costs per pT bin.
"""
import argparse
import glob
import os
import sys

import numpy as np
import h5py

LABELS = ["TTToSemiLeptonic", "TTToHadronic"]
PT_BINS = [300, 400, 500, 600, 800, 1000, np.inf]


def evkey(ev):
    """One int64 per (lumi, event); MC has a single run number (checked)."""
    assert (ev[:, 0] == ev[0, 0]).all(), "run number not constant"
    return ev[:, 1].astype(np.int64) * (1 << 32) + ev[:, 2].astype(np.int64)


def jetkey(ev, pt):
    return np.rec.fromarrays([evkey(ev), np.round(pt.astype(np.float64), 2)])


def probes(store, label):
    E, T, P, C = [], [], [], []
    for p in sorted(glob.glob(os.path.join(store, "ttbar", "tt_%s_sub*.h5" % label))):
        with h5py.File(p, "r") as f:
            cols, stages = f["ttbar_tag"].attrs["columns"].split(","), f["cutflow"].attrs["stages"].split(",")
            E.append(f["event_info"][:])
            t = f["ttbar_tag"][:]
            T.append(t[:, [cols.index("stage"), cols.index("channel")]])
            P.append(f["jet_kinematics"][:, 0])
            C.append(f["truth"][:, f["truth"].attrs["columns"].split(",").index("category")])
    T = np.concatenate(T)
    return np.concatenate(E), T[:, 0], T[:, 1], np.concatenate(P), np.concatenate(C), stages


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("store")
    p.add_argument("-o", "--output", default=None, help="write the veto list here (refuses to overwrite)")
    a = p.parse_args(argv)
    if a.output and os.path.exists(a.output):
        sys.exit("%s exists" % a.output)
    veto = {}
    for lab in LABELS:
        E, stage, chan, ppt, cat, stages = probes(a.store, lab)
        with h5py.File(os.path.join(a.store, "merged_%s.h5" % lab), "r") as f:
            gc = [c.decode() if isinstance(c, bytes) else c for c in f["gen_info"].attrs["columns"]]
            ka = jetkey(f["event_info"][:], f["jet_kinematics"][:, 0])
            tm = f["gen_info"][:, gc.index("top_match")].astype(int)
        with h5py.File(os.path.join(a.store, "merged_%s_topmatched.h5" % lab), "r") as f:
            ev_s, pt_s = f["event_info"][:], f["jet_kinematics"][:, 0]
        ia = dict((k, i) for i, k in enumerate(ka.tolist()))
        sig = set(jetkey(ev_s, pt_s).tolist())
        kp = jetkey(E, ppt)
        print("%s: %d stored probes" % (lab, len(kp)))
        for st in ("one_probe", "btag", "tau32"):
            m = (stage >= stages.index(st)) & (chan > 0)
            idx = np.array([ia.get(k, -1) for k in kp[m].tolist()]) if m.any() else np.zeros(0, int)
            found = idx >= 0
            M = np.zeros((4, 3), int)
            np.add.at(M, (cat[m][found].astype(int), tm[idx[found]]), 1)
            in_sig = np.array([k in sig for k in kp[m].tolist()], bool)
            print("  stage >= %-9s %8d probes, %8d found in merged_%s.h5; %7d (%.3f) are fine-tuning signal jets"
                  " (%.3f of the fully merged ones)" % (st, m.sum(), found.sum(), lab, in_sig.sum(), in_sig.mean(),
                                                         in_sig[cat[m] == 3].mean() if (cat[m] == 3).any() else np.nan))
            print("    truth category (other, bq, W, t) x top_match (0, 1, 2): %s" % M.tolist())
        drop = np.isin(evkey(ev_s), evkey(E))
        h_all = np.histogram(pt_s, PT_BINS)[0]
        h_keep = np.histogram(pt_s[~drop], PT_BINS)[0]
        print("  vetoing the %d stored events removes %d of %d fine-tuning jets (%.1f%%)" % (
            len(E), drop.sum(), len(pt_s), 100 * drop.mean()))
        print("    pT bin " + " ".join("%9s" % ("%d-%s" % (PT_BINS[i], "inf" if np.isinf(PT_BINS[i + 1]) else
                                                 int(PT_BINS[i + 1]))) for i in range(len(PT_BINS) - 1)))
        print("    all    " + " ".join("%9d" % v for v in h_all))
        print("    kept   " + " ".join("%9d" % v for v in h_keep))
        veto[lab] = (E, stage, stages)
    if a.output:
        with h5py.File(a.output, "w") as fo:
            fo.attrs["doc"] = ("(run, luminosityBlock, event) of every event stored by ttbar_semilep/ttbar_maker.py in "
                               "<store>/ttbar/tt_<label>_sub*.h5, per label. Remove these events from the fine-tuning "
                               "signal (merged_<label>_topmatched.h5) so that the tag-and-probe MC is independent of the "
                               "training set. <label>_stage = cuts passed (index in its 'stages' attribute).")
            for lab, (E, stage, stages) in veto.items():
                fo.create_dataset(lab, data=E, compression="gzip").attrs["columns"] = "run,luminosityBlock,event"
                fo.create_dataset(lab + "_stage", data=stage.astype(np.int8), compression="gzip").attrs["stages"] = \
                    ",".join(stages)
        print("wrote", a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
