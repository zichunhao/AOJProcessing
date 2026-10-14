#!/usr/bin/env python3
# Yields, MC truth composition and data/MC of ttbar_maker.py outputs.
#
#   python3 ttbar_summary.py --lumi 16.39 \
#       --mc TTToSemiLeptonic:365.34:tt_*.h5 --mc TTTo2L2Nu:88.29:tt2l_*.h5 \
#       --data SingleMuon:1.0:sm_*.h5 [--route lepton|jetht|jetht_not_lepton|any] [--json out.json]
#
# --mc NAME:XSEC_PB:GLOB[,GLOB...]   MC weight = gen_weight * xsec * lumi / sum(attrs sum_genweight)
# --data NAME:SCALE:GLOB[,GLOB...]   data rows weighted by SCALE (e.g. events in the full dataset /
#                                    events processed, to project a subset to the whole dataset)
# Offline levels (rows must have been stored with --store one_probe or looser), from ttbar_tag/stage:
#   probe = passed up to one_probe, dphi = + dphi(lep, probe), btag = + b tag, final = + tau32 < 0.5.
# --route selects rows by the stored trigger bits: lepton (mu channel & muon path, or el channel &
# electron path), jetht (AOJ JetHT OR), jetht_not_lepton, any (no extra requirement).  NOTE: in MC
# the AOJ JetHT OR includes paths that are prescaled in data, so the jetht route over-predicts there.
# Numpy + h5py only.
import argparse
import glob
import json
import sys

import numpy as np
import h5py

LEVELS = ["probe", "dphi", "btag", "final"]
LEVEL_STAGE = {"probe": "one_probe", "dphi": "dphi", "btag": "btag", "final": "tau32"}
PT_BINS = [300., 400., 500., 600., np.inf]
CATS = ["other", "bq", "W", "t"]
CHANNELS = {"mu": 1, "el": 2, "all": None}


def expand(spec):
    out = []
    for g in spec.split(","):
        m = sorted(glob.glob(g))
        if not m:
            raise SystemExit("no file matches %s" % g)
        out += m
    return out


def load(paths):
    cols = {}
    rows = {"pt": [], "msd": [], "tag": [], "truth": [], "w": [], "pn": []}
    sumw, n_entries, cf = 0.0, 0, None
    for p in paths:
        with h5py.File(p, "r") as f:
            tc = f["ttbar_tag"].attrs["columns"].split(",")
            cols["tag"] = tc
            rows["tag"].append(f["ttbar_tag"][:])
            rows["pt"].append(f["jet_kinematics"][:, 0])
            rows["msd"].append(f["jet_kinematics"][:, 3])
            jt = f["jet_tagging"].attrs.get("columns", "")
            jt = jt.split(",") if jt else []
            rows["pn"].append(f["jet_tagging"][:, jt.index("particleNet_TvsQCD")] if "particleNet_TvsQCD" in jt
                              else np.full(f["jet_tagging"].shape[0], np.nan, np.float32))
            if "truth" in f:
                cols["truth"] = f["truth"].attrs["columns"].split(",")
                rows["truth"].append(f["truth"][:])
            if "gen_weight" in f:
                rows["w"].append(f["gen_weight"][:])
            s = float(f.attrs.get("sum_genweight", np.nan))
            sumw += s if np.isfinite(s) else 0.0
            n_entries += int(f.attrs["n_entries"])
            c = f["cutflow/w"][:]
            cf = c if cf is None else cf + c
            stages = f["cutflow"].attrs["stages"].split(",")
            store = json.loads(f.attrs["selection"])["store"]
            if store not in ("one_probe", "trigger", "filters_json", "one_lepton", "met"):
                print("WARNING: %s stored with --store %s: the 'probe' level is incomplete" % (p, store))
    d = {"tag": np.concatenate(rows["tag"]), "pt": np.concatenate(rows["pt"]), "msd": np.concatenate(rows["msd"]),
         "pn": np.concatenate(rows["pn"])}
    d["c"] = dict((k, d["tag"][:, i]) for i, k in enumerate(cols["tag"]))
    if rows["truth"]:
        tr = np.concatenate(rows["truth"])
        d["truth"] = dict((k, tr[:, i]) for i, k in enumerate(cols["truth"]))
    if rows["w"]:
        d["gw"] = np.concatenate(rows["w"])
    d["sumw"], d["n_entries"], d["cutflow_w"], d["stages"] = sumw, n_entries, cf, stages
    return d


def masks(d, route):
    c = d["c"]
    lep = ((c["channel"] == 1) & (c["trig_mu"] > 0)) | ((c["channel"] == 2) & (c["trig_el"] > 0))
    jet = c["trig_jetht"] > 0
    r = {"lepton": lep, "jetht": jet, "jetht_not_lepton": jet & ~lep, "any": np.ones(len(jet), bool)}[route]
    if "stage" in c:
        out = {}
        for lev in LEVELS:
            st = LEVEL_STAGE[lev]
            out[lev] = r & (c["stage"] >= d["stages"].index(st)) if st in d["stages"] else r
        return out
    probe = r                                   # files written before the stage column existed
    btag = probe & (c["n_b"] >= 1)
    return {"probe": probe, "dphi": probe, "btag": btag, "final": btag & (c["pass_tau32"] > 0)}


def table(samples, route, lumi):
    res = {}
    for name, s in samples.items():
        d, kind = s["d"], s["kind"]
        if kind == "mc":
            norm = s["xsec"] * lumi * 1e3 / d["sumw"] if d["sumw"] > 0 else np.nan
            w = d["gw"] * norm
        else:
            norm = s["scale"]
            w = np.full(len(d["pt"]), norm)
        m = masks(d, route)
        r = {"kind": kind, "norm": norm, "n_entries": d["n_entries"], "levels": {}}
        for lev in LEVELS:
            out = {}
            for ch, code in CHANNELS.items():
                cm = m[lev] if code is None else (m[lev] & (d["c"]["channel"] == code))
                bins = {}
                for lo, hi in zip(PT_BINS[:-1] + [300., 400.], PT_BINS[1:] + [np.inf, np.inf]):
                    bm = cm & (d["pt"] > lo) & (d["pt"] <= hi)
                    e = {"yield": float(w[bm].sum()), "n": int(bm.sum()),
                         "err": float(np.sqrt((w[bm] ** 2).sum()))}
                    if "truth" in d and e["yield"] != 0:
                        cat = d["truth"]["category"]
                        e["frac"] = dict((CATS[k], float(w[bm & (cat == k)].sum() / e["yield"])) for k in range(4))
                    bins["%d-%s" % (lo, "inf" if not np.isfinite(hi) else int(hi))] = e
                out[ch] = bins
            r["levels"][lev] = out
        res[name] = r
    return res


def fmt(res, lumi, route):
    lines = ["route = %s, lumi = %.2f fb^-1 (MC = sigma x L; data = rows x SCALE)" % (route, lumi)]
    keys = ["300-inf", "400-inf", "300-400", "400-500", "500-600", "600-inf"]
    for lev in LEVELS:
        for ch in ("mu", "el", "all"):
            lines.append("")
            lines.append("[%s | %s]  probe pT bin:  %s" % (lev, ch, "  ".join("%14s" % k for k in keys)))
            tot_mc = dict((k, 0.0) for k in keys)
            for name, r in res.items():
                b = r["levels"][lev][ch]
                lines.append("  %-22s %s" % (name, "  ".join("%8.1f(%5d)" % (b[k]["yield"], b[k]["n"]) for k in keys)))
                if r["kind"] == "mc":
                    for k in keys:
                        tot_mc[k] += b[k]["yield"]
                if "frac" in b["300-inf"]:
                    lines.append("  %-22s %s" % ("   frac t/W/bq/other", "  ".join(
                        "%14s" % ("/".join("%.2f" % b[k]["frac"][c] for c in ("t", "W", "bq", "other"))
                                  if "frac" in b[k] else "-") for k in keys)))
            if sum(r["kind"] == "mc" for r in res.values()) > 1:
                lines.append("  %-22s %s" % ("sum MC", "  ".join("%14.1f" % tot_mc[k] for k in keys)))
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mc", action="append", default=[], metavar="NAME:XSEC_PB:GLOB")
    p.add_argument("--data", action="append", default=[], metavar="NAME:SCALE:GLOB")
    p.add_argument("--lumi", type=float, default=16.39, help="fb^-1 (2016G+H golden: 7.653 + 8.740)")
    p.add_argument("--route", default="lepton", choices=["lepton", "jetht", "jetht_not_lepton", "any"])
    p.add_argument("--json", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    samples = {}
    for spec in args.mc:
        name, xs, g = spec.split(":", 2)
        samples[name] = {"kind": "mc", "xsec": float(xs), "d": load(expand(g))}
    for spec in args.data:
        name, sc, g = spec.split(":", 2)
        samples[name] = {"kind": "data", "scale": float(sc), "d": load(expand(g))}
    res = table(samples, args.route, args.lumi)
    print(fmt(res, args.lumi, args.route))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"route": args.route, "lumi": args.lumi, "samples": res}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
