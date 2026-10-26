#!/usr/bin/env python3
"""Evaluation set in the semileptonic-ttbar control region: every tag-and-probe probe (one AK8 jet per event) of the
skim, in the input format of the fine-tuning sets, with what is needed to cut tighter offline and to measure the
tagger scale factor eff(data) / eff(MC) for fully merged tops.

    bash ttbar_semilep/py3env.sh finetune/make_eval_ttbar.py -o OUTDIR --part mc   [--src <store>/ttbar_skim_v3]
    bash ttbar_semilep/py3env.sh finetune/make_eval_ttbar.py -o OUTDIR --part data [--src <store>/ttbar_skim_v3]

Inputs: ttbar_maker.py outputs made with --preset skim --store one_probe (tt_<label>_subNNN.h5) of the CR samples of
ttbar_semilep/samples.py (cr = True; MC or data by --part). No gen matching and no resampling: a stored probe is kept
when it
  - lies in the fine-tuning window, as make_finetune_set.py: 20 < mSD < 500, pT < 2500 (--msd-range, --pt-max);
    pT > 300, |eta| < 2.5 and jetId == 6 are part of the probe definition
  - is in a channel its sample models (samples.py: MuEnriched QCD mu, inclusive pT-hat QCD e; SingleMuon mu,
    SingleElectron e, so an event in both primary datasets is counted once)
  - reached --min-stage (default one_probe, i.e. the whole skim; dphi / btag / tau32 are flags)
Per probe (OUTDIR/<part>/<part>_NNNN.h5, in source order):
  PFCands (N,150,16)  H5_skim_split.py features, exactly as in the fine-tuning sets: the models' input
  jet_kinematics, jet_tagging, event_info   as in the fine-tuning sets
  ttbar_tag (N,K)     every tag-side variable of ttbar_maker.py (union of the inputs' columns, NaN where a file lacks one)
  truth (N,K)         MC truth of the probe (NaN for data); ak4 (N,10,7) the event's AK4 jets (NaN if the input has none)
  label               1 = fully merged top (truth category t: b, q, q' within dR < 0.8 of the probe, the fine-tuning
                      signal definition), 0 = any other MC probe, -1 = data
  sample, process     sample code (names in the attribute) and process group (names in attrs process_names)
  w_phys              MC: genWeight x xsec x L / sum_genweight(sample) [events in --lumi]; data: 1
  cr (N,8) int8       selection flags, columns an, v1, v2, v3, skim, dphi, btag, tau32 (below)
  source (N,2) int32  input file (index into summary_<part>.json "files") and row
Selections (flags; e.g. v3 & btag = the scale-factor region, v3 & btag & tau32 = the top-enriched one):
  an     the HIG-24-010 analysis-note selection ('an' preset): the 'an' lepton (pT > 55, isolated, no other lepton),
         its trigger, MET > 50, exactly one AK8 jet above 300 GeV (equal to the 'an' production, check_eval_ttbar.py)
  v1     an without the extra-AK8 veto (the probe = the leading AOJ jet away from the lepton)
  v2     v1 with any isolated tight lepton above 30 GeV ('an' ID + isolation, exactly one veto lepton)
  v3     v2 with MET > 20 (the proposed evaluation selection)
  skim   the skim's lepton triggers (non-isolated leptons too, Mu50 / Ele115): v3 is inside it; skim & ~v2 = the
         QCD-enriched sideband
  dphi, btag, tau32   the cut-flow stage reached (cumulative: tau32 implies btag implies dphi)
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ttbar_semilep"))
import make_finetune_set as mf  # noqa: E402
import samples as smp  # noqa: E402

CR_COLS = ["an", "v1", "v2", "v3", "skim", "dphi", "btag", "tau32"]
LABELS = list(smp.SAMPLES)                       # sample code = index (every manifest label, CR or not)
AK4_SHAPE = (10, 7)
TOP_CATEGORY = 3                                 # truth category t (ttbar_selection.TRUTH_CATEGORIES)
CERT = "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
ERA_RUNS = {"2016G": (278820, 280385), "2016H": (280919, 284044)}


def label_of(path):
    return re.sub(r"_sub\d+\.h5$", "", os.path.basename(path))[len("tt_"):]


def cols_of(d):
    """Column names of a dataset: ttbar_maker writes one comma-joined string (H5_maker an array of names)."""
    c = mf.colnames(d)
    return c[0].split(",") if len(c) == 1 else c


def cr_flags(d, stages):
    """(N, len(CR_COLS)) int8 from the ttbar_tag columns d (dict of arrays) and the cut-flow stage names."""
    ch = d["channel"]
    trig_an = np.where(ch == 1, d["trig_mu_an"], np.where(ch == 2, d["trig_el_an"], 0)) > 0
    trig_sk = np.where(ch == 1, d["trig_mu"], np.where(ch == 2, d["trig_el"], 0)) > 0
    an_lep = d["an_channel"] == ch
    iso30 = (d["lep_id_iso"] > 0) & (d["lep_pt"] > 30) & ((d["n_veto_mu"] + d["n_veto_el"]) == 1)
    with np.errstate(invalid="ignore"):
        met50, met20 = d["met_pt"] > 50, d["met_pt"] > 20
    st = d["stage"]
    f = [an_lep & trig_an & met50 & (d["n_ak8"] == 1), an_lep & trig_an & met50, iso30 & trig_an & met50,
         iso30 & trig_an & met20, trig_sk,
         st >= stages.index("dphi"), st >= stages.index("btag"), st >= stages.index("tau32")]
    return np.stack(f, axis=1).astype(np.int8)


def file_info(path):
    """Pass 1: normalisation attributes, column sets, row count and processed lumi sections of one input."""
    with h5py.File(path, "r") as f:
        info = dict(path=path, n=int(f["event_info"].shape[0]), n_entries=int(f.attrs["n_entries"]),
                    sum_genweight=float(f.attrs["sum_genweight"]), format=int(f.attrs.get("format_version", 2)),
                    selection=json.loads(f.attrs["selection"]), tag=cols_of(f["ttbar_tag"]),
                    truth=cols_of(f["truth"]) if "truth" in f else None,
                    ak4=cols_of(f["ak4"]) if "ak4" in f else None, pf="PFCands" in f,
                    stages=f["cutflow"].attrs["stages"].split(","),
                    lumis=f["lumis"][:] if "lumis" in f else None)
    return info


def read_file(args):
    """Pass 2: the kept probes of one input, converted."""
    idx, path, label, part, S, min_stage, tagcols, truthcols, ak4cols, norm = args
    s = smp.SAMPLES[label]
    out, st = None, dict(rows=0, kept=0)
    with h5py.File(path, "r") as f:
        n = f["event_info"].shape[0]
        st["rows"] = n
        if n == 0:
            return idx, out, st
        stages = f["cutflow"].attrs["stages"].split(",")
        T = f["ttbar_tag"][:]
        tc = cols_of(f["ttbar_tag"])
        d = dict((k, T[:, i].astype(np.float64)) for i, k in enumerate(tc))
        jk_all = f["jet_kinematics"][:]
        keep = (mf.in_sel(jk_all, S) & np.isin(d["channel"], s["channels"])
                & (d["stage"] >= stages.index(min_stage)))
        rows = np.flatnonzero(keep)
        st["kept"] = len(rows)
        if not len(rows):
            return idx, out, st
        pf = []
        for b0 in range(0, n, mf.BLOCK):
            r = rows[(rows >= b0) & (rows < b0 + mf.BLOCK)]
            if len(r):
                pf.append(f["PFCands"][b0:min(b0 + mf.BLOCK, n)][r - b0])
        pf = np.concatenate(pf)
        jk = jk_all[rows]
        k = len(rows)
        tag = np.full((k, len(tagcols)), np.nan, np.float32)
        tag[:, :len(tc)] = T[rows]
        truth = np.full((k, len(truthcols)), np.nan, np.float32)
        if "truth" in f:
            tr = f["truth"][:][rows]
            truth[:, :tr.shape[1]] = tr
        ak4 = np.full((k,) + AK4_SHAPE, np.nan, np.float32)
        if "ak4" in f:
            ak4[:] = f["ak4"][:][rows]
        if part == "mc":
            label_arr = (truth[:, truthcols.index("category")] == TOP_CATEGORY).astype(np.int8)
            w = (f["gen_weight"][:][rows].astype(np.float64) * norm).astype(np.float32)
        else:
            label_arr = np.full(k, -1, np.int8)
            w = np.ones(k, np.float32)
        out = dict(PFCands=mf.skim_pf(pf, jk), jet_kinematics=jk, jet_tagging=f["jet_tagging"][:][rows],
                   event_info=f["event_info"][:][rows], ttbar_tag=tag, truth=truth, ak4=ak4, label=label_arr,
                   sample=np.full(k, LABELS.index(label), np.int8),
                   process=np.full(k, smp.GROUPS.index(s["group"]), np.int8), w_phys=w,
                   cr=cr_flags(d, stages)[rows], source=np.stack([np.full(k, idx), rows], axis=1).astype(np.int32))
    return idx, out, st


def union_columns(lists, what):
    """The longest column list; every other one must be a prefix of it (format versions append columns)."""
    lists = [l for l in lists if l is not None]
    if not lists:
        return None
    full = max(lists, key=len)
    for l in lists:
        if full[:len(l)] != l:
            sys.exit("%s columns differ between inputs (not a prefix): %s vs %s" % (what, l, full))
    return full


def cert_lumis(root):
    """Certified (run, lumi) pairs per era from the golden JSON in the repo."""
    js = json.load(open(os.path.join(root, CERT)))
    out = {}
    for era, (lo, hi) in ERA_RUNS.items():
        out[era] = set((int(r), l) for r, ranges in js.items() if lo <= int(r) <= hi
                       for a, b in ranges for l in range(a, b + 1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--part", choices=["mc", "data"], required=True)
    ap.add_argument("--src", default="/ceph/cms/store/user/zichun/parcel/AOJ/ttbar_skim_v3",
                    help="directory of the ttbar_maker outputs tt_<label>_subNNN.h5")
    ap.add_argument("--lumi", type=float, default=sum(smp.LUMI_FB.values()), help="fb^-1 for w_phys (default 2016G+H)")
    ap.add_argument("--msd-range", type=float, nargs=2, default=[20., 500.], help="mSD window (default 20 500 GeV)")
    ap.add_argument("--pt-max", type=float, default=2500., help="probe pT < this (default 2500 GeV)")
    ap.add_argument("--min-stage", default="one_probe", help="keep probes that reached this cut-flow stage")
    ap.add_argument("--labels", nargs="*", default=None, help="only these samples (tests)")
    ap.add_argument("--max-files", type=int, default=0, help="per sample, the first N inputs (tests; w_phys stays "
                                                                  "normalised to all inputs)")
    ap.add_argument("--jets-per-shard", type=int, default=1000000)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    S = dict(pt_max=a.pt_max, msd=tuple(a.msd_range))
    outdir = os.path.join(a.outdir, a.part)
    if glob.glob(os.path.join(outdir, "*.h5")):
        sys.exit("%s already has shards: use a new --outdir" % outdir)
    t0 = time.time()

    want = [l for l, s in smp.SAMPLES.items() if s["cr"] and (s["group"] == "data") == (a.part == "data")]
    if a.labels:
        want = [l for l in want if l in a.labels]
    files = sorted(p for p in glob.glob(os.path.join(a.src, "tt_*_sub*.h5")) if label_of(p) in want)
    skipped = sorted(set(label_of(p) for p in glob.glob(os.path.join(a.src, "tt_*_sub*.h5"))) - set(want))
    missing = [l for l in want if not any(label_of(p) == l for p in files)]
    print("%s: %d inputs of %d samples in %s; not used: %s; missing: %s" % (
        a.part, len(files), len(want) - len(missing), a.src, ", ".join(skipped) or "-", ", ".join(missing) or "-"))
    if not files:
        sys.exit("no inputs")

    with Pool(a.workers) as pool:
        info = pool.map(file_info, files, chunksize=4)
    sel0 = info[0]["selection"]
    for i in info:
        if i["selection"]["preset"] != "skim" or i["selection"]["store"] != "one_probe" or \
                i["selection"]["cuts"] != sel0["cuts"]:
            sys.exit("%s: not the skim preset stored at one_probe, or other cuts than %s" % (i["path"], files[0]))
        if not i["pf"] and i["n"]:
            sys.exit("%s: no PFCands" % i["path"])
    tagcols = union_columns([i["tag"] for i in info], "ttbar_tag")
    truthcols = union_columns([i["truth"] for i in info], "truth") or \
        ["category", "had_top_pt", "had_top_dr", "n_had_W", "n_lep_W", "lep_flav"]
    ak4cols = union_columns([i["ak4"] for i in info], "ak4") or \
        ["pt", "eta", "phi", "btagDeepFlavB", "hadronFlavour", "dr_probe", "dr_lep"]
    stages = info[0]["stages"]
    if a.min_stage not in stages:
        sys.exit("--min-stage %s: not a stage (%s)" % (a.min_stage, ", ".join(stages)))

    per = {}
    for i in info:
        l = label_of(i["path"])
        p = per.setdefault(l, dict(files=0, n_entries=0, sum_genweight=0.0, rows=0, formats=set(), lumis=set()))
        p["files"] += 1
        p["n_entries"] += i["n_entries"]
        p["sum_genweight"] += i["sum_genweight"] if a.part == "mc" else 0.0
        p["rows"] += i["n"]
        p["formats"].add(i["format"])
        if i["lumis"] is not None:
            p["lumis"] |= set(map(tuple, i["lumis"].tolist()))
    norm = {l: (smp.SAMPLES[l]["xsec_pb"] * a.lumi * 1e3 / p["sum_genweight"] if a.part == "mc" else 1.0)
            for l, p in per.items()}
    use = files
    if a.max_files:
        use = [p for p in files if sum(1 for q in files[:files.index(p) + 1] if label_of(q) == label_of(p)) <= a.max_files]
    tasks = [(files.index(p), p, label_of(p), a.part, S, a.min_stage, tagcols, truthcols, ak4cols, norm[label_of(p)])
             for p in use]

    attrs = dict(doc=__doc__, part=a.part, src=a.src, lumi_fb=a.lumi, selection=json.dumps(S), min_stage=a.min_stage,
                 skim_cuts=json.dumps(sel0["cuts"]), process_names=",".join(smp.GROUPS), cr_columns=",".join(CR_COLS))
    w = mf.Writer(outdir, a.part, a.jets_per_shard, attrs,
                  cols=dict(ttbar_tag=tagcols, truth=truthcols, ak4=ak4cols, cr=CR_COLS, source=["file", "row"]),
                  names=LABELS)
    kept = {}
    yields = {}
    with Pool(a.workers) as pool:
        for k, (idx, out, st) in enumerate(pool.imap(read_file, tasks, chunksize=1)):
            l = label_of(files[idx])
            kept[l] = kept.get(l, 0) + st["kept"]
            if out is not None:
                w.write(out)
                y = yields.setdefault(l, {})
                for name, m in (("all", np.ones(len(out["label"]), bool)), ("an_btag", out["cr"][:, 0] & out["cr"][:, 6]),
                                ("v3_btag", out["cr"][:, 3] & out["cr"][:, 6]),
                                ("v3_btag_tau32", out["cr"][:, 3] & out["cr"][:, 7])):
                    m = m.astype(bool)
                    c = y.setdefault(name, dict(n=0, w=0.0, n_top=0, w_top=0.0))
                    c["n"] += int(m.sum())
                    c["w"] += float(out["w_phys"][m].astype(np.float64).sum())
                    t = m & (out["label"] == 1)
                    c["n_top"] += int(t.sum())
                    c["w_top"] += float(out["w_phys"][t].astype(np.float64).sum())
            if k % 200 == 0:
                print("  %d/%d inputs, %d probes written (%.0f s)" % (k, len(tasks), w.total, time.time() - t0), flush=True)
    w.close()

    summary = dict(part=a.part, src=a.src, lumi_fb=a.lumi, selection=S, min_stage=a.min_stage, skim_cuts=sel0["cuts"],
                   files=[os.path.basename(p) for p in files], files_used=len(use), max_files=a.max_files,
                   columns=dict(ttbar_tag=tagcols, truth=truthcols, ak4=ak4cols, cr=CR_COLS),
                   written=dict(probes=w.total, shards=w.idx + 1), samples={}, seconds=None)
    cert = cert_lumis(os.path.dirname(HERE)) if a.part == "data" else {}
    for l, p in sorted(per.items(), key=lambda kv: LABELS.index(kv[0])):
        e = dict(group=smp.SAMPLES[l]["group"], files=p["files"], n_entries=p["n_entries"], rows=p["rows"],
                 kept=kept.get(l, 0), formats=sorted(p["formats"]), yields=yields.get(l, {}))
        if a.part == "mc":
            e.update(xsec_pb=smp.SAMPLES[l]["xsec_pb"], sum_genweight=p["sum_genweight"])
        else:
            era = l.split("_")[-1]
            got = len(p["lumis"] & cert[era]) if p["lumis"] else None
            e.update(lumis_certified=got, lumis_certified_era=len(cert[era]),
                     lumi_fraction=(got / len(cert[era]) if got is not None else None))
        summary["samples"][l] = e
    summary["seconds"] = time.time() - t0
    with open(os.path.join(a.outdir, "summary_%s.json" % a.part), "w") as fo:
        json.dump(summary, fo, indent=1)
    tot = dict((k, sum(y.get(k, {}).get("w", 0.0) for y in yields.values())) for k in ("all", "an_btag", "v3_btag", "v3_btag_tau32"))
    print("%d probes in %d shards; weighted: %s" % (w.total, w.idx + 1, ", ".join("%s %.0f" % kv for kv in tot.items())))
    print("done (%.0f s)" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
