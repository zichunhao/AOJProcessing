#!/usr/bin/env python3
# Read NanoAOD (AOJ PFNano Stage-1 output, or any Run-2 UL NanoAOD), apply the semileptonic-ttbar
# tag-and-probe selection (ttbar_selection.py) and write the probe AK8 jets as an HDF5 file in the
# AOJ jet-level format plus the tag-side information.
#
#   python3 ttbar_maker.py -i nano_data2016.root -o ttbar.h5 --sample_type data \
#       -j ../Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt
#   python3 ttbar_maker.py -i nano_mc2016post.root -o ttbar.h5 --sample_type MC
#   (through ttbar_semilep/py3env.sh: an isolated LCG_105a python3, never CMSSW's python2)
#
# Output (one row per stored event = one probe jet):
#   event_info (N,3) int64, jet_kinematics (N,4), jet_tagging (N,13)  -- exactly H5_maker.py's columns
#   PFCands (N,150,11)       -- only if the input has PFCands + FatJetPFCands (PFNano); H5_maker's
#                               convention: first 150 constituents in link order, then pt-sorted
#   ttbar_tag (N,K) float32  -- tag-side / event info, columns in attrs["columns"]
#   truth (N,K) float32 + gen_weight (N,) float64   -- MC only
#   cutflow/n, cutflow/w (len(CUTFLOW), 2) -- per channel (mu, el), every input event (w: sum genWeight)
# Root attrs: selection (JSON: era, trigger mode, cuts, store stage), n_entries, sum_genweight
# (MC: Runs/genEventSumw of the inputs), inputs, versions.
#
# What is stored: events with stage >= --store (default 'btag', i.e. everything except the tau32
# cut, whose decision is the ttbar_tag column pass_tau32); --store tau32 keeps the final selection.
# ttbar_tag/stage holds how far each stored event got, so looser stores can be cut offline.
#
# Portability (awkward 1.x / 2.x, uproot 4 / 5): awkward only at the uproot boundary.
import argparse
import json
import os
import platform
import sys

import numpy as np
import h5py
import uproot
import awkward as ak

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttbar_selection as ts  # noqa: E402

PRODUCER = "AOJProcessing/ttbar_semilep/ttbar_maker.py"
N_PF = 150
PF_COLS = ["pt", "eta", "phi", "mass", "d0", "d0Err", "dz", "dzErr", "charge", "pdgId", "puppiWeight"]
TAG_COLUMNS = ["channel", "stage", "lep_pdgId", "lep_pt", "lep_eta", "lep_phi", "lep_charge", "met_pt", "met_phi",
               "lepW_pt", "dphi_lep_probe", "dr_lep_probe", "n_aoj_jets", "n_ak8", "probe_idx", "tau32", "tau21",
               "pass_tau32", "n_b", "b_pt", "b_eta", "b_phi", "b_score", "b_dr_probe", "b_dr_lep",
               "trig_mu", "trig_el", "trig_jetht"]
TRUTH_COLUMNS = ["category", "had_top_pt", "had_top_dr", "n_had_W", "n_lep_W", "lep_flav"]


class Warner(object):
    def __init__(self):
        self.seen = set()

    def __call__(self, msg):
        if msg not in self.seen:
            self.seen.add(msg)
            print("WARNING: " + msg)
            sys.stdout.flush()


def runs_genweight(f):
    try:
        if "Runs" in f and "genEventSumw" in f["Runs"].keys():
            return float(np.sum(f["Runs"]["genEventSumw"].array(library="np").astype(np.float64)))
    except Exception as e:
        print("WARNING: cannot read Runs/genEventSumw (%s)" % e)
    return float("nan")


def to_np(x):
    return np.asarray(ak.to_numpy(x))


class Reader(object):
    """Which branches to read from a tree, and conversion of one block to the selection input."""

    def __init__(self, keys, era, cuts, is_mc, want_pf, warn):
        self.keys = set(keys)
        scalars, colls = ts.branches(era, cuts, is_mc)
        self.scalars = [b for b in scalars if b in self.keys]
        for b in ("run", "luminosityBlock", "event", "MET_pt", "MET_phi") + (("genWeight",) if is_mc else ()):
            if b not in self.keys:
                raise RuntimeError("required branch %s missing" % b)
        self.colls = {}
        for name, cols in colls.items():
            if "n" + name not in self.keys:
                if name == "GenPart":
                    warn("no GenPart in the input: MC truth not available")
                    continue
                raise RuntimeError("collection %s missing" % name)
            have = [c for c in cols if name + "_" + c in self.keys]
            miss = [c for c in cols if c not in have]
            if miss:
                if name == "FatJet" and all(c in ts.FATJET_TAGGING for c in miss) and not (set(miss) & {"tau2", "tau3"}):
                    warn("FatJet columns absent, stored as NaN: %s" % ", ".join(miss))
                else:
                    raise RuntimeError("%s columns missing: %s" % (name, ", ".join(miss)))
            self.colls[name] = have
        self.pf = want_pf and all(b in self.keys for b in ["nPFCands", "nFatJetPFCands", "FatJetPFCands_jetIdx",
                                                            "FatJetPFCands_pFCandsIdx"] + ["PFCands_" + c for c in PF_COLS])
        if want_pf and not self.pf:
            warn("PFCands / FatJetPFCands absent: no PFCands dataset (plain NanoAOD input)")
        if self.pf:
            self.colls["PFCands"] = list(PF_COLS)
            self.colls["FatJetPFCands"] = ["jetIdx", "pFCandsIdx"]

    def names(self):
        out = list(self.scalars)
        for name, cols in self.colls.items():
            out.append("n" + name)
            out += [name + "_" + c for c in cols]
        return out

    def convert(self, arr):
        ev = {}
        for b in self.scalars:
            ev[b] = to_np(arr[b])
        for name, cols in self.colls.items():
            counts = to_np(arr["n" + name]).astype(np.int64)
            d = dict((c, to_np(ak.flatten(arr[name + "_" + c]))) for c in cols)
            if name == "FatJet":
                for c in ts.FATJET_TAGGING:
                    if c not in d:
                        d[c] = np.full(int(counts.sum()), np.nan, np.float32)
            ev[name] = ts.Coll(counts, d)
        return ev


def pf_rows(ev, gidx_jet):
    """H5_maker PFCands of each selected probe: first 150 linked candidates, then pt-sorted (N,150,11)."""
    fj, link, pf = ev["FatJet"], ev["FatJetPFCands"], ev["PFCands"]
    out = np.zeros((len(gidx_jet), N_PF, 11), dtype=np.float32)
    for r, g in enumerate(gidx_jet):
        e = fj.evt[g]
        jl = g - fj.offsets[e]
        s0, s1 = link.offsets[e], link.offsets[e + 1]
        sel = link["jetIdx"][s0:s1] == jl
        idx = link["pFCandsIdx"][s0:s1][sel][:N_PF].astype(np.int64) + pf.offsets[e]
        if len(idx) == 0:
            continue
        pt, eta, phi, m = (pf[k][idx].astype(np.float64) for k in ("pt", "eta", "phi", "mass"))
        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
        E = np.sqrt(px * px + py * py + pz * pz + m * m)
        rows = np.stack([px, py, pz, E] + [pf[k][idx] for k in PF_COLS[4:]], axis=1).astype(np.float32)
        rows = rows[np.flip(np.argsort(np.sqrt(rows[:, 0] ** 2 + rows[:, 1] ** 2)))]      # == H5_maker.sort_pfcands
        out[r, :len(rows)] = rows
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="semileptonic-ttbar top-enriched AOJ jets from NanoAOD")
    p.add_argument("-i", "--input", nargs="+", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--sample_type", choices=["data", "MC"], required=True)
    p.add_argument("-j", "--json", default=None, help="golden JSON (data)")
    p.add_argument("--era", default=ts.DEFAULT_ERA, choices=sorted(ts.ERAS))
    p.add_argument("--trigger", default="lepton", choices=ts.TRIGGER_MODES)
    p.add_argument("--store", default="btag", choices=ts.CUTFLOW[1:], help="store events that passed up to this cut")
    p.add_argument("--preset", default="an", choices=sorted(ts.PRESETS),
                   help="cut set: an = HIG-24-010 implementation (default), paper = as worded in the paper")
    p.add_argument("--set", nargs="*", default=[], metavar="CUT=VALUE", help="override a cut of the preset")
    p.add_argument("--no-pf", action="store_true", help="do not read/write PF candidates")
    p.add_argument("--step", default="200 MB", help="uproot step_size (entries or size)")
    p.add_argument("--max-events", type=int, default=-1, help="per input file")
    p.add_argument("--timeout", type=float, default=600., help="XRootD request timeout [s] for root:// inputs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    is_mc = args.sample_type == "MC"
    cuts = ts.merged_cuts(ts.parse_overrides(args.set), args.preset)
    lumimask = ts.aoj.LumiMask(args.json) if (args.json and not is_mc) else None
    if not is_mc and lumimask is None:
        print("WARNING: data without a golden JSON (-j): no lumi selection")
    store_stage = ts.CUTFLOW.index(args.store)
    warn = Warner()
    step = int(args.step) if str(args.step).isdigit() else args.step
    print("ttbar_maker: python %s numpy %s awkward %s uproot %s h5py %s | era %s trigger %s store>=%s" % (
        platform.python_version(), np.__version__, ak.__version__, uproot.__version__, h5py.__version__,
        args.era, args.trigger, args.store))

    blocks = {k: [] for k in ("event_info", "jet_kinematics", "jet_tagging", "PFCands", "ttbar_tag", "truth", "gen_weight")}
    cf_n = np.zeros((len(ts.CUTFLOW), len(ts.CHANNELS)))
    cf_w = np.zeros_like(cf_n)
    n_entries, sumw, has_pf, has_truth = 0, 0.0, None, None
    for path in args.input:
        f = uproot.open(path, timeout=args.timeout) if path.startswith("root://") else uproot.open(path)
        t = f["Events"]
        if is_mc:
            sumw += runs_genweight(f)
        if t.num_entries == 0:
            print("%s: 0 entries" % path)
            continue
        rd = Reader(t.keys(), args.era, cuts, is_mc, not args.no_pf, warn)
        has_pf = rd.pf if has_pf is None else (has_pf and rd.pf)
        has_truth = ("GenPart" in rd.colls) if has_truth is None else (has_truth and "GenPart" in rd.colls)
        stop = None if args.max_events < 0 else min(args.max_events, t.num_entries)
        n_file = 0
        for arr in t.iterate(rd.names(), step_size=step, entry_stop=stop, library="ak"):
            ev = rd.convert(arr)
            n = len(ev["run"])
            n_file += n
            s = ts.select(ev, era=args.era, cuts=cuts, trigger=args.trigger, lumimask=lumimask)
            if s["filters_missing"]:
                warn("AOJ filters absent (skipped): %s" % ", ".join(s["filters_missing"]))
            w = ev["genWeight"].astype(np.float64) if is_mc else np.ones(n)
            cf_n += ts.cutflow(s)
            cf_w += ts.cutflow(s, w)
            keep = np.nonzero(s["stage"] >= store_stage)[0]
            if len(keep) == 0:
                continue
            fj = ev["FatJet"]
            gidx = fj.offsets[keep] + s["probe_idx"][keep]
            blocks["event_info"].append(np.stack([ev["run"][keep], ev["luminosityBlock"][keep],
                                                  ev["event"][keep]], axis=1).astype(np.int64))
            blocks["jet_kinematics"].append(np.stack([fj[k][gidx] for k in ("pt", "eta", "phi", "msoftdrop")],
                                                     axis=1).astype(np.float32))
            blocks["jet_tagging"].append(np.stack([fj[k][gidx] for k in ts.FATJET_TAGGING], axis=1).astype(np.float32))
            blocks["ttbar_tag"].append(np.stack([np.asarray(s[k])[keep]
                                                 .astype(np.float64) for k in TAG_COLUMNS], axis=1).astype(np.float32))
            if rd.pf:
                blocks["PFCands"].append(pf_rows(ev, gidx))
            if is_mc:
                blocks["gen_weight"].append(w[keep])
                if "GenPart" in ev:
                    g = ts.gen_truth(ev["GenPart"], s["probe_eta"], s["probe_phi"])
                    blocks["truth"].append(np.stack([np.asarray(g[k], np.float64)[keep] for k in TRUTH_COLUMNS],
                                                    axis=1).astype(np.float32))
        n_entries += n_file
        print("%s: %d entries, %d stored so far" % (path, n_file, sum(len(x) for x in blocks["event_info"])))
        sys.stdout.flush()

    def cat(name, shape, dtype):
        b = blocks[name]
        return np.concatenate(b).astype(dtype) if b else np.zeros((0,) + shape, dtype)

    with h5py.File(args.output, "w") as out:
        out.create_dataset("event_info", data=cat("event_info", (3,), np.int64), chunks=True, maxshape=(None, 3))
        for name, shape in (("jet_kinematics", (4,)), ("jet_tagging", (len(ts.FATJET_TAGGING),))):
            out.create_dataset(name, data=cat(name, shape, np.float32), chunks=True, maxshape=(None,) + shape,
                               compression="gzip")
        out["jet_kinematics"].attrs["columns"] = "pt,eta,phi,msoftdrop"
        out["jet_tagging"].attrs["columns"] = ",".join(ts.FATJET_TAGGING)
        out["event_info"].attrs["columns"] = "run,luminosityBlock,event"
        if has_pf:
            out.create_dataset("PFCands", data=cat("PFCands", (N_PF, 11), np.float32), chunks=True,
                               maxshape=(None, N_PF, 11), compression="gzip")
            out["PFCands"].attrs["columns"] = "Px,Py,Pz,E,d0,d0Err,dz,dzErr,charge,pdgId,puppiWeight"
        d = out.create_dataset("ttbar_tag", data=cat("ttbar_tag", (len(TAG_COLUMNS),), np.float32), compression="gzip")
        d.attrs["columns"] = ",".join(TAG_COLUMNS)
        d.attrs["doc"] = ("channel 1 = mu, 2 = el; stage = cuts passed in that channel (index in cutflow stages; "
                          "the last one = tau32); n_ak8 = AK8 jets counted for the one-probe requirement; lep_* = the tight lepton; lepW_pt = |pT(lep) + MET|; probe_idx = "
                          "FatJet index; n_b / b_* = b-tagged AK4 jets separated from probe and lepton (b_* = leading)")
        if is_mc:
            out.create_dataset("gen_weight", data=cat("gen_weight", (), np.float64))
            if has_truth:
                d = out.create_dataset("truth", data=cat("truth", (len(TRUTH_COLUMNS),), np.float32))
                d.attrs["columns"] = ",".join(TRUTH_COLUMNS)
                d.attrs["categories"] = ",".join(ts.TRUTH_CATEGORIES)
                d.attrs["doc"] = ("category: partons of a hadronic W (+ its top's b) within dR < %.1f of the probe: "
                                  "0 other, 1 b+q, 2 W (q,q'), 3 t (b,q,q')" % ts.TRUTH_DR)
        g = out.create_group("cutflow")
        g.create_dataset("n", data=cf_n.astype(np.int64))
        g.create_dataset("w", data=cf_w)
        g.attrs["stages"] = ",".join(ts.CUTFLOW)
        g.attrs["channels"] = ",".join(ts.CHANNELS)
        out.attrs["selection"] = json.dumps({"era": args.era, "trigger": args.trigger, "store": args.store, "preset": args.preset,
                                             "cuts": cuts, "json": os.path.basename(args.json) if args.json else None,
                                             "sample_type": args.sample_type})
        out.attrs["n_entries"] = n_entries
        out.attrs["sum_genweight"] = sumw if is_mc else float("nan")
        out.attrs["inputs"] = json.dumps(args.input)
        out.attrs["producer"] = PRODUCER
        out.attrs["versions"] = json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
                                                awkward=ak.__version__, uproot=uproot.__version__, h5py=h5py.__version__))
        n_stored = out["event_info"].shape[0]
    last = len(ts.CUTFLOW) - 1
    print("ttbar_maker: %d entries, stored %d (selected incl. tau32: mu %d, el %d) -> %s" % (
        n_entries, n_stored, cf_n[last, 0], cf_n[last, 1], args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
