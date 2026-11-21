#!/usr/bin/env python3
"""Per-sample inputs of the tagger scale-factor measurement in the semileptonic-ttbar control region: the selected
probes of the SF samples (one AK8 jet per event), in the input format of the fine-tuning sets, with selection flags
and per-row weights (normalisation and corrections, nominal and variations).

    bash ttbar_semilep/py3env.sh finetune/make_sf_inputs.py -o OUTDIR --lumi-csv pp_2016lumibyls.csv \
        --el-trig-sf el_trig_sf.json [--src <store>/sf_samples_20261026_skim]

Inputs: ttbar_maker.py outputs (--preset skim --store one_probe, format 3) of the CR samples of
ttbar_semilep/samples.py (cr = True; data and MC); --lumi-csv = brilcalc lumi by lumi section of the golden JSON
(CERN Open Data record 1059, pp_2016lumibyls.csv); --el-trig-sf = the electron trigger SF of etrig_sf.py (copied to
OUTDIR).
Stored probes: in the fine-tuning window (20 < mSD < 500, pT < 2500), in a channel the sample models (samples.py:
SingleMuon / MuEnriched QCD mu, SingleElectron / inclusive pT-hat QCD e), at stage >= one_probe, passing
an_anyak8 or skim_iso (so every selection flag below can be applied offline; dphi / btag / tau32 are flags).
Output OUTDIR/<label>.h5, one file per sample, in source order:
  PFCands (N,150,16)  H5_skim_split.py features, as in the fine-tuning sets: the models' input
  jet_kinematics, jet_tagging, event_info, ttbar_tag (every column of the input), ak4 (N,10,7), truth (MC)
  label               1 = fully merged top (truth category t), 0 = other MC probe, -1 = data
  sel (N,8) int8      an, an_anyak8, iso30, iso30_met20 (the nominal), skim_iso, dphi, btag, tau32
  weights (N,K) f32   w_norm and the corrections (columns below); weight (N,) f32 = the nominal product
  source (N,2) int32  input file (index into the `files` attribute) and row
  lumis (M,2) (data)  the lumi sections the label's inputs cover (union of the inputs' `lumis`)
Selections (ttbar_tag columns; trig_an = the channel's isolated paths IsoMu24|IsoTkMu24 / Ele27|Ele32 WPTight,
trig_skim = those OR Mu50|TkMu50 / Ele115; lepton = the tag lepton of the skim, the "an"-tight one when there is):
  an           an_channel == channel & trig_an & MET > 50 & n_ak8 == 1 (the analysis-note selection, exact)
  an_anyak8    an without the extra-AK8 veto
  iso30        "an" lepton ID + isolation (lep_id_iso) & lepton pT > 30 & exactly one veto lepton & trig_an & MET > 50
  iso30_met20  iso30 with MET > 20: the NOMINAL selection
  skim_iso     iso30_met20 with trig_skim instead of trig_an (adds the non-isolated paths)
  dphi, btag, tau32   the cut-flow stage reached (cumulative)
Weights (MC; data: all 1). weight = w_norm x sf_pu x sf_lep_id x sf_lep_iso x sf_lep_reco x sf_trig x sf_btag.
  w_norm       genWeight x xsec_pb x L x 1000 / sum(sum_genweight of every input of the label); L = the recorded
               luminosity of the lumi sections the data inputs cover (--lumi-csv), i.e. what the data saw
  sf_pu        LUM puWeights 2016postVFP_UL Collisions16_UltraLegacy_goldenJSON(pu_ntrueint); _up / _down
  sf_lep_*     mu: MUO muon_Z NUM_TightID_DEN_TrackerMuons, NUM_TightRelIso_DEN_TightIDandIPCut, reco 1;
               e: EGM UL-Electron-ID-SF wp80iso (id, isolation included: iso 1) and RecoAbove20, at the SC eta;
               _up / _down = nominal +- the total uncertainty
  sf_trig      mu: NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight (the isolated paths: exact for
               every selection but skim_iso); e: Ele27|Ele32 WPTight, measured here with the orthogonal muon
               trigger (etrig_sf.py: pT x abs(SC eta) bins, _up / _down its total uncertainty; no POG SF exists)
  sf_btag      fixed-WP method 1a on the jets the b-tag stage can use (AK4 pT > 40, |eta| < 2.4, tight ID,
               dR(probe) > 1.5, dR(lepton) > 0.5; DeepJet M 0.2489): prod_tagged SF x prod_untagged (1 - SF eff) /
               (1 - eff), SF = BTV deepJet_comb (b, c) / deepJet_incl (light), eff = MC efficiencies of these
               jets in ttbar + single top (btag_eff.json); _bc_up / _bc_down / _light_up / _light_down vary the SFs
  w_toppt      ttbar (TTToSemiLeptonic, TTTo2L2Nu) only, not in `weight`: sqrt(SF(t) SF(tbar)) with the TOP PAG
               data/POWHEG+Pythia8 SF(pT) = exp(0.0615 - 0.0005 pT), pT clamped at 500 GeV; w_toppt_nnlo the
               NNLO/POWHEG+Pythia8 one, 0.103 exp(-0.0118 pT) - 0.000134 pT + 0.973. Not normalised.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ttbar_semilep"))
import make_finetune_set as mf  # noqa: E402
import make_eval_ttbar as me  # noqa: E402
import samples as smp  # noqa: E402
import ttbar_selection as ts  # noqa: E402

SEL_COLS = ["an", "an_anyak8", "iso30", "iso30_met20", "skim_iso", "dphi", "btag", "tau32"]
NOMINAL = "iso30_met20"
W_COLS = ["w_norm", "sf_pu", "sf_pu_up", "sf_pu_down", "sf_lep_id", "sf_lep_id_up", "sf_lep_id_down",
          "sf_lep_iso", "sf_lep_iso_up", "sf_lep_iso_down", "sf_lep_reco", "sf_lep_reco_up", "sf_lep_reco_down",
          "sf_trig", "sf_trig_up", "sf_trig_down", "sf_btag", "sf_btag_bc_up", "sf_btag_bc_down",
          "sf_btag_light_up", "sf_btag_light_down", "w_toppt", "w_toppt_nnlo"]
NOMINAL_PRODUCT = ["w_norm", "sf_pu", "sf_lep_id", "sf_lep_iso", "sf_lep_reco", "sf_trig", "sf_btag"]
POG = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/"
ERA = "2016postVFP"
CUTS = ts.DEFAULT_CUTS                           # dphi / btag / tau32 are the same in the skim preset
BTAG_WP = ts.ERAS[ERA]["btag_wp"][CUTS["b_wp"]]
EFF_PT = [40., 60., 80., 100., 140., 200., 300., 600., 1e5]
EFF_ETA = [0., 0.8, 1.6, 2.4]
FLAVOURS = [5, 4, 0]
EFF_MIN_JETS = 30
TOPPT_SAMPLES = ("TTToSemiLeptonic", "TTTo2L2Nu")
AK4 = dict((c, i) for i, c in enumerate(ts.AK4_COLUMNS))

_CORR = {}


def corr():
    """correctionlib objects, loaded once per process."""
    if not _CORR:
        import correctionlib
        mu = correctionlib.CorrectionSet.from_file(POG + "MUO/2016postVFP_UL/muon_Z.json.gz")
        el = correctionlib.CorrectionSet.from_file(POG + "EGM/2016postVFP_UL/electron.json.gz")
        bt = correctionlib.CorrectionSet.from_file(POG + "BTV/2016postVFP_UL/btagging.json.gz")
        pu = correctionlib.CorrectionSet.from_file(POG + "LUM/2016postVFP_UL/puWeights.json.gz")
        _CORR.update(mu_id=mu["NUM_TightID_DEN_TrackerMuons"], mu_iso=mu["NUM_TightRelIso_DEN_TightIDandIPCut"],
                     mu_trig=mu["NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight"],
                     el=el["UL-Electron-ID-SF"], b_bc=bt["deepJet_comb"], b_light=bt["deepJet_incl"],
                     pu=pu["Collisions16_UltraLegacy_goldenJSON"])
    return _CORR


def tag_dict(f):
    T = f["ttbar_tag"][:]
    return dict((k, T[:, i].astype(np.float64)) for i, k in enumerate(me.cols_of(f["ttbar_tag"]))), T


def flags(d, stages):
    """(N, len(SEL_COLS)) int8; me.cr_flags gives an, an_anyak8, iso30, iso30_met20, skim, dphi, btag, tau32."""
    cr = me.cr_flags(d, stages).astype(bool)
    ch = d["channel"]
    trig_sk = np.where(ch == 1, d["trig_mu"], np.where(ch == 2, d["trig_el"], 0)) > 0
    lep = (d["lep_id_iso"] > 0) & (d["lep_pt"] > 30) & ((d["n_veto_mu"] + d["n_veto_el"]) == 1)
    skim_iso = lep & trig_sk
    return np.stack([cr[:, 0], cr[:, 1], cr[:, 2], cr[:, 3], skim_iso, cr[:, 5], cr[:, 6], cr[:, 7]], axis=1).astype(np.int8)


def kept_rows(f, label, S):
    """Rows to store, the tag columns and the flags of one input."""
    s = smp.SAMPLES[label]
    stages = f["cutflow"].attrs["stages"].split(",")
    d, T = tag_dict(f)
    fl = flags(d, stages)
    jk = f["jet_kinematics"][:]
    keep = (mf.in_sel(jk, S) & np.isin(d["channel"], s["channels"]) & (d["stage"] >= stages.index("one_probe"))
            & ((fl[:, SEL_COLS.index("an_anyak8")] > 0) | (fl[:, SEL_COLS.index("skim_iso")] > 0)))
    return np.flatnonzero(keep), d, T, fl, jk


def btag_jets(ak4):
    """Candidate jets of the b-tag requirement: (valid mask, pt, abseta, flavour, tagged), each (N, 10)."""
    pt, eta = ak4[..., AK4["pt"]], ak4[..., AK4["eta"]]
    with np.errstate(invalid="ignore"):
        ok = ((pt > CUTS["b_min_pt"]) & (np.abs(eta) < CUTS["b_max_abs_eta"])
              & (ak4[..., AK4["dr_probe"]] > CUTS["b_min_dr_fj"]) & (ak4[..., AK4["dr_lep"]] > CUTS["b_min_dr_lep"]))
        tagged = ok & (ak4[..., AK4["btagDeepFlavB"]] > BTAG_WP)
    fl = np.nan_to_num(ak4[..., AK4["hadronFlavour"]], nan=0).astype(np.int64)
    fl = np.where(fl == 5, 5, np.where(fl == 4, 4, 0))
    return ok, pt, np.abs(eta), fl, tagged


def eff_hist(args):
    """b-tag efficiency histograms (numerator, denominator per flavour) of one ttbar / single-top input."""
    path, label, S, norm = args
    num = np.zeros((len(FLAVOURS), len(EFF_PT) - 1, len(EFF_ETA) - 1))
    den, cnt = np.zeros_like(num), np.zeros_like(num)
    with h5py.File(path, "r") as f:
        if f["event_info"].shape[0] == 0:
            return num, den, cnt
        rows, d, T, fl, jk = kept_rows(f, label, S)
        rows = rows[fl[rows, SEL_COLS.index(NOMINAL)] > 0]
        if not len(rows):
            return num, den, cnt
        ak4 = f["ak4"][:][rows]
        w = f["gen_weight"][:][rows].astype(np.float64) * norm
    ok, pt, aeta, flav, tagged = btag_jets(ak4)
    ww = np.broadcast_to(w[:, None], ok.shape)
    for i, fv in enumerate(FLAVOURS):
        m = ok & (flav == fv)
        den[i] += np.histogram2d(pt[m], aeta[m], bins=[EFF_PT, EFF_ETA], weights=ww[m])[0]
        num[i] += np.histogram2d(pt[m & tagged], aeta[m & tagged], bins=[EFF_PT, EFF_ETA], weights=ww[m & tagged])[0]
        cnt[i] += np.histogram2d(pt[m], aeta[m], bins=[EFF_PT, EFF_ETA])[0]
    return num, den, cnt


def btag_weights(ak4, eff):
    """Method-1a event weights: nominal, bc up/down, light up/down; and the number of tagged candidate jets."""
    ok, pt, aeta, flav, tagged = btag_jets(ak4)
    out = dict((k, np.ones(len(ak4))) for k in ("sf_btag", "sf_btag_bc_up", "sf_btag_bc_down", "sf_btag_light_up",
                                                 "sf_btag_light_down"))
    if not ok.any():
        return out, (tagged.sum(axis=1))
    C = corr()
    ip = np.clip(np.digitize(pt, EFF_PT) - 1, 0, len(EFF_PT) - 2)
    ie = np.clip(np.digitize(aeta, EFF_ETA) - 1, 0, len(EFF_ETA) - 2)
    fi = np.where(flav == 5, 0, np.where(flav == 4, 1, 2))
    e = eff[fi, ip, ie]
    sf = {}
    sel_pt, sel_eta, sel_fl = pt[ok].astype(np.float64), aeta[ok].astype(np.float64), flav[ok]
    hf = sel_fl > 0
    for syst in ("central", "up", "down"):
        v = np.ones(len(sel_pt))
        if hf.any():
            v[hf] = C["b_bc"].evaluate(syst, CUTS["b_wp"], sel_fl[hf], sel_eta[hf], sel_pt[hf])
        if (~hf).any():
            v[~hf] = C["b_light"].evaluate(syst, CUTS["b_wp"], sel_fl[~hf], sel_eta[~hf], sel_pt[~hf])
        sf[syst] = v

    def weight(sfj):
        full = np.ones(ok.shape)
        full[ok] = sfj
        per = np.where(tagged, full, np.maximum(1 - full * e, 0.0) / (1 - e))
        return np.where(ok, per, 1.0).prod(axis=1)

    out["sf_btag"] = weight(sf["central"])
    for var in ("up", "down"):
        out["sf_btag_bc_" + var] = weight(np.where(hf, sf[var], sf["central"]))
        out["sf_btag_light_" + var] = weight(np.where(hf, sf["central"], sf[var]))
    return out, tagged.sum(axis=1)


def el_trig_lookup(T, pt, sc_eta):
    """(sf, up, down) of the electron trigger SF table T (etrig_sf.py) at the electrons' pT and abs(SC eta)."""
    ip = np.clip(np.digitize(pt, T["pt_edges"]) - 1, 0, len(T["pt_edges"]) - 2)
    ie = np.clip(np.digitize(np.abs(sc_eta), T["abseta_edges"]) - 1, 0, len(T["abseta_edges"]) - 2)
    return tuple(np.asarray(T[k])[ip, ie] for k in ("sf", "sf_up", "sf_down"))


def lepton_sfs(d, el_trig=None):
    """Lepton ID / isolation / reco / trigger SFs with up / down, per row (1 where not applicable)."""
    C = corr()
    n = len(d["channel"])
    out = dict((k, np.ones(n)) for k in W_COLS if k.startswith("sf_lep_") or k.startswith("sf_trig"))
    mu, el = d["channel"] == 1, d["channel"] == 2
    if mu.any():
        eta, pt = d["lep_eta"][mu], d["lep_pt"][mu]
        for key, name in (("sf_lep_id", "mu_id"), ("sf_lep_iso", "mu_iso"), ("sf_trig", "mu_trig")):
            for suf, v in (("", "nominal"), ("_up", "systup"), ("_down", "systdown")):
                out[key + suf][mu] = C[name].evaluate(eta, pt, v)
    if el.any():
        eta, pt = d["lep_sc_eta"][el], d["lep_pt"][el]
        for key, wp in (("sf_lep_id", "wp80iso"), ("sf_lep_reco", "RecoAbove20")):
            for suf, v in (("", "sf"), ("_up", "sfup"), ("_down", "sfdown")):
                out[key + suf][el] = C["el"].evaluate(ERA, v, wp, eta, pt)
        if el_trig is not None:
            out["sf_trig"][el], out["sf_trig_up"][el], out["sf_trig_down"][el] = el_trig_lookup(el_trig, pt, eta)
    return out


def toppt(truth, cols):
    t, tb = truth[:, cols.index("gen_top_pt")].astype(np.float64), truth[:, cols.index("gen_antitop_pt")].astype(np.float64)
    data = lambda x: np.exp(0.0615 - 0.0005 * np.minimum(x, 500.))
    nnlo = lambda x: 0.103 * np.exp(-0.0118 * x) - 0.000134 * x + 0.973
    ok = np.isfinite(t) & np.isfinite(tb)
    w1 = np.where(ok, np.sqrt(data(np.nan_to_num(t)) * data(np.nan_to_num(tb))), 1.0)
    w2 = np.where(ok, np.sqrt(nnlo(np.nan_to_num(t)) * nnlo(np.nan_to_num(tb))), 1.0)
    return w1, w2


def read_file(args):
    """The stored probes of one input, converted, with flags and weights."""
    idx, path, label, S, norm, eff, el_trig = args
    mc = smp.is_mc(label)
    st = dict(rows=0, kept=0, nb_mismatch=0, nb_checked=0)
    with h5py.File(path, "r") as f:
        n = f["event_info"].shape[0]
        st["rows"] = n
        empty = n == 0
        rows, d, T, fl, jk_all = (np.zeros(0, np.int64), None, None, None, None) if empty else kept_rows(f, label, S)
        k = len(rows)
        st["kept"] = k
        if k:
            pf = []
            for b0 in range(0, n, mf.BLOCK):
                r = rows[(rows >= b0) & (rows < b0 + mf.BLOCK)]
                if len(r):
                    pf.append(f["PFCands"][b0:min(b0 + mf.BLOCK, n)][r - b0])
            pf = np.concatenate(pf)
            jk = jk_all[rows]
            ak4 = f["ak4"][:][rows]
            truth = f["truth"][:][rows] if mc else None
            gw = f["gen_weight"][:][rows].astype(np.float64) if mc else None
            out = dict(PFCands=mf.skim_pf(pf, jk), jet_kinematics=jk, jet_tagging=f["jet_tagging"][:][rows],
                       event_info=f["event_info"][:][rows], ttbar_tag=T[rows], ak4=ak4)
            if mc:
                out["truth"] = truth
            tcols = me.cols_of(f["truth"]) if mc else None
        else:
            out = None
    if not k:
        return idx, None, st
    dk = dict((c, v[rows]) for c, v in d.items())
    W = dict((c, np.ones(k)) for c in W_COLS)
    bw, n_tag = btag_weights(ak4, eff)
    st["nb_checked"] = k
    st["nb_mismatch"] = int((n_tag != dk["n_b"]).sum())
    if mc:
        W["w_norm"] = gw * norm
        C = corr()
        npu = dk["pu_ntrueint"]
        for suf, v in (("", "nominal"), ("_up", "up"), ("_down", "down")):
            W["sf_pu" + suf] = C["pu"].evaluate(npu, v)
        W.update(lepton_sfs(dk, el_trig))
        W.update(bw)
        if label in TOPPT_SAMPLES:
            W["w_toppt"], W["w_toppt_nnlo"] = toppt(truth, tcols)
        out["label"] = (truth[:, tcols.index("category")] == me.TOP_CATEGORY).astype(np.int8)
    else:
        out["label"] = np.full(k, -1, np.int8)
    nominal = np.prod([W[c] for c in NOMINAL_PRODUCT], axis=0)
    out["sel"] = fl[rows]
    out["weights"] = np.stack([W[c] for c in W_COLS], axis=1).astype(np.float32)
    out["weight"] = nominal.astype(np.float32)
    out["source"] = np.stack([np.full(k, idx), rows], axis=1).astype(np.int32)
    return idx, out, st


class LabelWriter(mf.Writer):
    """One file per sample, OUTDIR/<label>.h5 (the fine-tuning sets' chunking and compression)."""

    def _roll(self):
        self.idx += 1
        self.n = 0
        self.f = h5py.File(os.path.join(self.dir, "%s.h5" % self.prefix), "w")
        for k, v in self.attrs.items():
            self.f.attrs[k] = v


def file_info(path):
    with h5py.File(path, "r") as f:
        sel = json.loads(f.attrs["selection"])
        return dict(path=path, n=int(f["event_info"].shape[0]), sum_genweight=float(f.attrs["sum_genweight"]),
                    format=int(f.attrs["format_version"]), preset=sel["preset"], store=sel["store"],
                    cuts=sel["cuts"], lumis=f["lumis"][:] if "lumis" in f else None,
                    tag=me.cols_of(f["ttbar_tag"]), truth=me.cols_of(f["truth"]) if "truth" in f else None)


def covered_lumi(lumi_csv, lumis_by_era):
    """Recorded luminosity [fb^-1] of the golden-JSON lumi sections, per era: all, and those covered."""
    per_ls = {}
    for row in csv.reader(l for l in open(lumi_csv) if not l.startswith("#")):
        per_ls[(int(row[0].split(":")[0]), int(row[1].split(":")[0]))] = float(row[6])
    out = {}
    for era, (lo, hi) in me.ERA_RUNS.items():
        gold = sum(v for (r, _), v in per_ls.items() if lo <= r <= hi)
        cov = sum(per_ls.get(x, 0.0) for x in lumis_by_era[era])
        out[era] = dict(golden_fb=gold, covered_fb=cov, golden_ls=sum(1 for (r, _) in per_ls if lo <= r <= hi),
                        covered_ls=len(lumis_by_era[era] & set(per_ls)), not_golden_ls=len(lumis_by_era[era] - set(per_ls)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--src", default="/ceph/cms/store/user/zichun/parcel/AOJ/sf_samples_20261026_skim")
    ap.add_argument("--lumi-csv", required=True, help="pp_2016lumibyls.csv (Open Data record 1059)")
    ap.add_argument("--el-trig-sf", default=None, help="electron trigger SF table of etrig_sf.py (none: 1)")
    ap.add_argument("--msd-range", type=float, nargs=2, default=[20., 500.])
    ap.add_argument("--pt-max", type=float, default=2500.)
    ap.add_argument("--labels", nargs="*", default=None, help="only these samples (tests; data needed for L)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    S = dict(pt_max=a.pt_max, msd=tuple(a.msd_range))
    if glob.glob(os.path.join(a.outdir, "*.h5")):
        sys.exit("%s already has outputs: use a new --outdir" % a.outdir)
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    want = [l for l, s in smp.SAMPLES.items() if s["cr"] and (not a.labels or l in a.labels)]
    files = dict((l, sorted(glob.glob(os.path.join(a.src, "tt_%s_sub*.h5" % l)))) for l in want)
    missing = [l for l in want if not files[l]]
    if missing:
        sys.exit("no inputs for %s" % ", ".join(missing))
    allf = [p for l in want for p in files[l]]
    print("%d inputs of %d samples in %s" % (len(allf), len(want), a.src), flush=True)

    with Pool(a.workers) as pool:
        info = dict((i["path"], i) for i in pool.map(file_info, allf, chunksize=8))
    cuts0 = info[allf[0]]["cuts"]
    for i in info.values():
        if i["preset"] != "skim" or i["store"] != "one_probe" or i["cuts"] != cuts0 or i["format"] != 3:
            sys.exit("%s: not format 3, skim preset, one_probe, or other cuts" % i["path"])
    lumis = dict((e, set()) for e in me.ERA_RUNS)
    per_pd = {}
    for l in want:
        if not smp.is_mc(l):
            ls = set()
            for p in files[l]:
                ls |= set(map(tuple, info[p]["lumis"].tolist()))
            per_pd[l] = ls
            lumis[l.split("_")[-1]] |= ls
    for era in me.ERA_RUNS:
        pds = [per_pd[l] for l in per_pd if l.endswith(era)]
        if len(pds) == 2 and pds[0] != pds[1]:
            print("WARNING %s: SingleMuon and SingleElectron cover different lumi sections" % era)
    L = covered_lumi(a.lumi_csv, lumis)
    lumi_fb = sum(v["covered_fb"] for v in L.values())
    print("luminosity: %s -> L = %.4f fb^-1" % (json.dumps(L), lumi_fb), flush=True)
    sumw = dict((l, sum(info[p]["sum_genweight"] for p in files[l])) for l in want if smp.is_mc(l))
    norm = dict((l, smp.SAMPLES[l]["xsec_pb"] * lumi_fb * 1e3 / sumw[l] if smp.is_mc(l) else 1.0) for l in want)

    # b-tag efficiencies from ttbar + single top (nominal selection, all stored stages)
    eff_labels = [l for l in want if smp.SAMPLES[l]["group"] in ("ttbar", "single_top")]
    with Pool(a.workers) as pool:
        H = pool.map(eff_hist, [(p, l, S, norm[l]) for l in eff_labels for p in files[l]], chunksize=4)
    num, den, cnt = (sum(h[i] for h in H) for i in range(3))
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = num / den
        eta_incl = num.sum(axis=2, keepdims=True) / den.sum(axis=2, keepdims=True)
        fl_incl = num.sum(axis=(1, 2), keepdims=True) / den.sum(axis=(1, 2), keepdims=True)
    # bins with < EFF_MIN_JETS jets: the |eta|-inclusive value of the pT bin, else the flavour's inclusive one
    eff = np.where(cnt >= EFF_MIN_JETS, eff, np.where(cnt.sum(axis=2, keepdims=True) >= EFF_MIN_JETS,
                                                        np.broadcast_to(eta_incl, eff.shape), np.broadcast_to(fl_incl, eff.shape)))
    eff = np.clip(np.nan_to_num(eff, nan=0.5), 0.001, 0.999)
    json.dump(dict(doc="DeepJet M (%.4f) efficiency of the b-tag candidate jets in %s, %s selection, weighted by "
                       "w_norm; eff[flavour][pt bin][abs-eta bin]" % (BTAG_WP, "+".join(eff_labels), NOMINAL),
                   flavours=FLAVOURS, pt_edges=EFF_PT, abseta_edges=EFF_ETA, eff=eff.tolist(),
                   jets=cnt.tolist(), min_jets=EFF_MIN_JETS), open(os.path.join(a.outdir, "btag_eff.json"), "w"), indent=1)
    print("b-tag efficiencies (|eta|-averaged, per pT bin): %s" % dict(
        (fv, np.round(num[i].sum(1) / np.maximum(den[i].sum(1), 1e-9), 3).tolist()) for i, fv in enumerate(FLAVOURS)), flush=True)

    el_trig = None
    if a.el_trig_sf:
        el_trig = json.load(open(a.el_trig_sf))
        json.dump(el_trig, open(os.path.join(a.outdir, "el_trig_sf.json"), "w"), indent=1)
    summary = dict(src=a.src, selection=S, nominal=NOMINAL, el_trig_sf=bool(el_trig), sel_columns=SEL_COLS, weight_columns=W_COLS,
                   nominal_weight=NOMINAL_PRODUCT, lumi=L, lumi_fb=lumi_fb, skim_cuts=cuts0, samples={})
    for l in want:
        s = smp.SAMPLES[l]
        attrs = dict(doc=__doc__, label=l, group=s["group"], channels=json.dumps(list(s["channels"])),
                     src=a.src, files=json.dumps([os.path.basename(p) for p in files[l]]), selection=json.dumps(S),
                     nominal=NOMINAL, nominal_weight=",".join(NOMINAL_PRODUCT), lumi_fb=lumi_fb,
                     skim_cuts=json.dumps(cuts0), btag_wp=BTAG_WP)
        if smp.is_mc(l):
            attrs.update(xsec_pb=s["xsec_pb"], xsec_ref=s["xsec_ref"], sum_genweight=sumw[l])
        cols = dict(ttbar_tag=info[files[l][0]]["tag"], ak4=ts.AK4_COLUMNS, sel=SEL_COLS, weights=W_COLS,
                    source=["file", "row"])
        if smp.is_mc(l):
            cols["truth"] = info[files[l][0]]["truth"]
        w = LabelWriter(a.outdir, l, 1 << 62, attrs, cols=cols)
        tasks = [(i, p, l, S, norm[l], eff, el_trig) for i, p in enumerate(files[l])]
        tot = dict(rows=0, kept=0, nb_mismatch=0, nb_checked=0)
        with Pool(a.workers) as pool:
            for idx, out, st in pool.imap(read_file, tasks, chunksize=1):
                for kk in tot:
                    tot[kk] += st[kk]
                if out is not None:
                    w.write(out)
        if w.f is None:                                   # no probe: an empty file with the attributes
            w._roll()
        if not smp.is_mc(l):
            w.f.create_dataset("lumis", data=np.array(sorted(per_pd[l]), np.int64).reshape(-1, 2))
            w.f["lumis"].attrs["columns"] = "run,luminosityBlock"
        w.close()
        summary["samples"][l] = dict(group=s["group"], files=len(files[l]), **tot)
        if smp.is_mc(l):
            summary["samples"][l].update(xsec_pb=s["xsec_pb"], sum_genweight=sumw[l])
        print("  %-22s %4d inputs, %8d rows -> %7d stored; b-tag jets vs n_b: %d of %d differ (%.0f s)" % (
            l, len(files[l]), tot["rows"], tot["kept"], tot["nb_mismatch"], tot["nb_checked"], time.time() - t0), flush=True)
    summary["seconds"] = time.time() - t0
    json.dump(summary, open(os.path.join(a.outdir, "summary.json"), "w"), indent=1)
    print("done (%.0f s)" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
