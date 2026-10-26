#!/usr/bin/env python3
"""The samples of the semileptonic-ttbar control region (2016G+H data, UL16 post-VFP MC): the one list that the
production (file lists, Condor jobs), the summaries and the evaluation set read.

    python3 ttbar_semilep/samples.py                          # the table
    python3 ttbar_semilep/samples.py --regen [GROUP|LABEL...] # condor/regen_lists.txt lines (default: every CR sample)
    python3 ttbar_semilep/samples.py --make-lists [LABEL...]  # the missing file_lists/<label>.txt, from the Open Data
                                                              # file index (existing lists are kept)
    python3 ttbar_semilep/samples.py --status [GROUP|LABEL...] # online / on-demand files per record
Only public CMS Open Data is used: every file list holds eospublic paths (root://eospublic.cern.ch//eos/opendata/cms/).
Some records are partly or wholly "on demand" (on tape at CERN: most V+jets, single-top tW, MuEnriched QCD, DY and ttZ
records): request their files on the record page (https://opendata.cern.ch/record/<record>) and run once --status
shows them online; a job whose files are not online fails at staging and is held.

Per label:
  group     ttbar, single_top, wjets_lnu, wjets_qq, zjets_qq, dy_ll, diboson, ttv, qcd_mu, qcd, data
  channels  the CR channels the sample models (1 = mu, 2 = e): QCD MuEnriched only mu, the inclusive pT-hat QCD
            only e (so they never double count); data: mu from SingleMuon, e from SingleElectron (an event in both
            primary datasets is counted once)
  record    CERN Open Data record (recid); dataset = its title
  xsec_pb   the cross section used [pb]; xsec_ref says where it comes from; xsec_record_pb = the Open Data record
            value (GenXSecAnalyzer, i.e. the generator's order, filter efficiency included), for comparison
  cr        True = part of the CR (fit / evaluation MC); False = listed for completeness, not used (note says why)
Normalisation: w = genWeight * xsec_pb * L / sum(genWeight) over all events of the label (Runs/genEventSumw).
"""
import argparse
import json
import os
import sys
import urllib.request

LUMI_FB = {"2016G": 7.653, "2016H": 8.740}         # golden JSON (Open Data record 1059); total 16.39 fb^-1
CHANNELS = {1: "mu", 2: "e"}
BOTH = (1, 2)
_UL16 = "RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17"

# k-factors applied to LO V+jets (madgraphMLM HT-binned) cross sections; the same as the AOJ W+jets so far
K_W, K_Z = 1.21, 1.23
TT_REF = "831.76 pb (TOP++ NNLO+NNLL, mt = 172.5) x BR"


def _mc(group, record, dataset, xsec_pb, xsec_ref, xsec_record_pb, channels=BOTH, cr=True, note=""):
    return dict(group=group, record=record, dataset="/%s/%s-%s/MINIAODSIM" % (dataset[0], _UL16, dataset[1]),
                xsec_pb=xsec_pb, xsec_ref=xsec_ref, xsec_record_pb=xsec_record_pb, channels=channels, cr=cr, note=note)


SAMPLES = {
    # ---- ttbar (POWHEG; record values are the inclusive NLO generator cross section, without the BR)
    "TTToSemiLeptonic": _mc("ttbar", 67992, ("TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8", "v1"), 365.34, TT_REF + " 0.4392", 687.1),
    "TTTo2L2Nu": _mc("ttbar", 67800, ("TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8", "v1"), 88.29, TT_REF + " 0.1061", 687.1),
    "TTToHadronic": _mc("ttbar", 67840, ("TTToHadronic_TuneCP5_13TeV-powheg-pythia8", "v1"), 377.96, TT_REF + " 0.4544", 687.1,
                        cr=False, note="the fine-tuning signal: kept out of the CR MC (0.03 % of the region)"),
    # ---- single top (standard CMS values: Hathor NLO t-channel, approx. NNLO tW x BR, NLO s-channel x BR)
    "ST_tW_top": _mc("single_top", 64894, ("ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8", "v1"),
                     19.47, "35.85 pb (approx. NNLO) x (1 - BR(W->had)^2)", 32.45),
    "ST_tW_antitop": _mc("single_top", 64838, ("ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8", "v1"),
                         19.47, "35.85 pb (approx. NNLO) x (1 - BR(W->had)^2)", 32.51),
    "ST_tch_top": _mc("single_top", 64758, ("ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8", "v3"),
                      136.02, "Hathor NLO", 113.4),
    "ST_tch_antitop": _mc("single_top", 64658, ("ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8", "v3"),
                          80.95, "Hathor NLO", 67.93),
    "ST_sch": _mc("single_top", 64634, ("ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8", "v1"),
                  3.36, "10.32 pb (NLO) x BR(W->lnu)", 3.549),
}
# ---- W(->lnu)+jets, HT-binned madgraphMLM (LO x K_W; LO values from the HH4b xsecs.json)
for _b, _hi, _rec, _lo, _r in [(70, "100", 69732, 1275., 1269.), (100, "200", 69720, 1270., 1254.), (200, "400", 69724, 336.8, 336.4),
                               (400, "600", 69728, 44.64, 45.21), (600, "800", 69730, 11.6, 10.98), (800, "1200", 69734, 4.898, 4.927),
                               (1200, "2500", 69722, 1.142, 1.157), (2500, "Inf", 69726, 0.02678, 0.02624)]:
    SAMPLES["WJets_HT%d" % _b] = _mc("wjets_lnu", _rec, ("WJetsToLNu_HT-%dTo%s_TuneCP5_13TeV-madgraphMLM-pythia8" % (_b, _hi),
                                                          "v2" if _b == 2500 else "v1"),
                                     round(_lo * K_W, 6), "LO x %.2f" % K_W, _r)
# ---- W(->qq)+jets and Z(->qq)+jets, HT-binned madgraphMLM (record LO x K)
for _b, _hi, _rw, _xw, _rz, _xz in [(200, "400", 69750, 2564., 74911, 1012.), (400, "600", 69752, 277.2, 74913, 114.5),
                                    (600, "800", 69754, 59.11, 74915, 25.37), (800, "Inf", 69756, 28.76, 74917, 12.92)]:
    SAMPLES["WJetsToQQ_HT%d" % _b] = _mc("wjets_qq", _rw, ("WJetsToQQ_HT-%dto%s_TuneCP5_13TeV-madgraphMLM-pythia8" % (_b, _hi), "v2"),
                                         round(_xw * K_W, 6), "record LO x %.2f" % K_W, _xw)
    SAMPLES["ZJetsToQQ_HT%d" % _b] = _mc("zjets_qq", _rz, ("ZJetsToQQ_HT-%dto%s_TuneCP5_13TeV-madgraphMLM-pythia8" % (_b, _hi), "v2"),
                                         round(_xz * K_Z, 6), "record LO x %.2f" % K_Z, _xz)
# ---- Drell-Yan (Z/gamma* -> ll, m_ll > 50), HT-binned madgraphMLM (record LO x K_Z)
for _b, _hi, _rec, _x in [(70, "100", 35662, 140.0), (100, "200", 35650, 140.2), (200, "400", 35654, 38.38), (400, "600", 35658, 5.212),
                          (600, "800", 35660, 1.266), (800, "1200", 35664, 0.5686), (1200, "2500", 35652, 0.1332), (2500, "Inf", 35656, 0.002977)]:
    SAMPLES["DYJetsToLL_HT%d" % _b] = _mc("dy_ll", _rec, ("DYJetsToLL_M-50_HT-%dto%s_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8" % (_b, _hi), "v2"),
                                          round(_x * K_Z, 6), "record LO x %.2f" % K_Z, _x)
# ---- dibosons (inclusive pythia8; standard CMS higher-order values) and ttV (no UL16 ttW / ttH on the portal)
SAMPLES.update({
    "WW": _mc("diboson", 72695, ("WW_TuneCP5_13TeV-pythia8", "v1"), 118.7, "NNLO", 75.88),
    "WZ": _mc("diboson", 72753, ("WZ_TuneCP5_13TeV-pythia8", "v1"), 47.13, "NLO (MCFM)", 27.56),
    "ZZ": _mc("diboson", 75592, ("ZZ_TuneCP5_13TeV-pythia8", "v1"), 16.523, "NLO (MCFM)", 12.08),
    "TTZToLLNuNu": _mc("ttv", 68186, ("TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8", "v1"), 0.2439, "record (NLO)", 0.2439),
})
# ---- QCD: MuEnriched (pT-hat bins, mu channel; record values include the muon filter efficiency) and the inclusive
# pT-hat bins (e channel; the AOJ / fine-tuning QCD samples, plus 120-300 GeV: ~half of the CR's QCD is below
# pT-hat 300). The HT-binned QCD of the v2 skim is retired.
for _b, _hi, _rec, _x in [(120, "170", 63153, 21180.), (170, "300", 63173, 7011.), (300, "470", 63191, 620.3), (470, "600", 63215, 59.08),
                          (600, "800", 63227, 18.20), (800, "1000", 63231, 3.275), (1000, None, 63145, 1.079)]:
    _ds = "QCD_Pt-%d%s_MuEnrichedPt5_TuneCP5_13TeV-pythia8" % (_b, "To" + _hi if _hi else "")
    SAMPLES["QCD_MuEnr_Pt%d" % _b] = _mc("qcd_mu", _rec, (_ds, "v1"), _x, "record (filter efficiency included)", _x, channels=(1,))
for _b, _hi, _rec, _x in [(120, "170", 63155, 407400.), (170, "300", 63175, 103700.), (300, "470", 63193, 6832.), (470, "600", 63217, 551.9),
                          (600, "800", 63229, 156.5), (800, "1000", 63233, 26.25), (1000, "1400", 63147, 7.475), (1400, "1800", 63157, 0.6482),
                          (1800, "2400", 63177, 0.08746), (2400, "3200", 63187, 0.005235), (3200, "Inf", 63211, 0.0001352)]:
    SAMPLES["QCD_Pt%d" % _b] = _mc("qcd", _rec, ("QCD_Pt_%dto%s_TuneCP5_13TeV_pythia8" % (_b, _hi), "v1"), _x, "record (LO)", _x,
                                   channels=(2,), note="QCD_Pt2400: 1 of 76 files on disk" if _b == 2400 else "")
for _b, _hi, _rec, _x in [(300, "500", 63118, 324700.), (500, "700", 63126, 30350.), (700, "1000", 63139, 6425.),
                          (1000, "1500", 63078, 1117.), (1500, "2000", 63094, 108.3), (2000, "Inf", 63102, 21.94)]:
    SAMPLES["QCD_HT%d" % _b] = _mc("qcd_ht", _rec, ("QCD_HT%dto%s_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8" % (_b, _hi), "v1"),
                                   _x, "record (LO)", _x, channels=(2,), cr=False,
                                   note="retired: the e-channel QCD is the inclusive pT-hat QCD (v2 ttbar_skim/ has it)")
# ---- data
for _pd, _ch, _recs in [("SingleMuon", (1,), {"2016G": 30513, "2016H": 30546}), ("SingleElectron", (2,), {"2016G": 30512, "2016H": 30545})]:
    for _era, _rec in _recs.items():
        SAMPLES["%s_%s" % (_pd, _era)] = dict(group="data", record=_rec, dataset="/%s/Run%s-UL2016_MiniAODv2-v2/MINIAOD" % (_pd, _era),
                                              xsec_pb=None, xsec_ref="", xsec_record_pb=None, channels=_ch, cr=True,
                                              note="lumi %.3f fb^-1 (golden JSON)" % LUMI_FB[_era])

GROUPS = ["ttbar", "single_top", "wjets_lnu", "wjets_qq", "zjets_qq", "dy_ll", "diboson", "ttv", "qcd_mu", "qcd", "qcd_ht", "data"]


def is_mc(label):
    return SAMPLES[label]["group"] != "data"


def file_list(label, root="."):
    return os.path.join(root, "file_lists", label + ".txt")


def select(names):
    """Labels for a list of groups / labels (default: every CR sample)."""
    if not names:
        return [l for l, s in SAMPLES.items() if s["cr"]]
    out = []
    for n in names:
        hit = [l for l, s in SAMPLES.items() if s["group"] == n] if n in GROUPS else [n]
        for l in hit:
            if l not in SAMPLES:
                sys.exit("unknown sample or group '%s' (groups: %s)" % (l, ", ".join(GROUPS)))
            if l not in out:
                out.append(l)
    return out


EOSPUBLIC = "root://eospublic.cern.ch//eos/opendata/cms/"


def record_files(record):
    """(title, eospublic URIs, number of on-demand files) of a record, from its Open Data file index."""
    with urllib.request.urlopen("https://opendata.cern.ch/api/records/%d" % record, timeout=120) as r:
        m = json.load(r)["metadata"]
    files = [f for fi in m.get("_file_indices", []) for f in fi["files"]]
    uris = sorted(f["uri"] for f in files)
    if not all(u.startswith(EOSPUBLIC) for u in uris):
        raise RuntimeError("record %d: a file is not on eospublic, e.g. %s" % (record, uris[0]))
    return m["title"], uris, sum(1 for f in files if f.get("availability", "online") != "online")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--regen", nargs="*", metavar="GROUP|LABEL", help="print condor/regen_lists.txt lines")
    g.add_argument("--make-lists", nargs="*", metavar="LABEL", help="write the missing file_lists/<label>.txt")
    g.add_argument("--status", nargs="*", metavar="GROUP|LABEL", help="online / on-demand files of each record")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), help="AOJProcessing/")
    a = ap.parse_args(argv)
    if a.regen is not None:
        for l in select(a.regen):
            print("%s file_lists/%s.txt %s" % (l, l, l))
        return 0
    if a.status is not None:
        for l in select(a.status):
            _, uris, n_tape = record_files(SAMPLES[l]["record"])
            print("%-22s %6d  %5d files  %5d on demand%s" % (l, SAMPLES[l]["record"], len(uris), n_tape,
                                                             "  -> request at https://opendata.cern.ch/record/%d" % SAMPLES[l]["record"] if n_tape else ""))
        return 0
    if a.make_lists is not None:
        for l in select(a.make_lists):
            path = file_list(l, a.root)
            if os.path.exists(path):
                print("keep  %s (exists)" % path)
                continue
            title, uris, n_tape = record_files(SAMPLES[l]["record"])
            if title != SAMPLES[l]["dataset"]:
                sys.exit("%s: record %d is %s, expected %s" % (l, SAMPLES[l]["record"], title, SAMPLES[l]["dataset"]))
            with open(path, "w") as f:
                f.write("".join(u + "\n" for u in uris))
            print("wrote %s (%d files%s)" % (path, len(uris), ", %d on demand: request them first" % n_tape if n_tape else ""))
        return 0
    print("%-22s %-11s %-3s %6s %12s %12s  %s" % ("label", "group", "ch", "record", "xsec [pb]", "record [pb]", "note"))
    for l, s in SAMPLES.items():
        ch = "+".join(CHANNELS[c] for c in s["channels"])
        x = "%12.6g" % s["xsec_pb"] if s["xsec_pb"] is not None else "%12s" % "-"
        xr = "%12.6g" % s["xsec_record_pb"] if s["xsec_record_pb"] is not None else "%12s" % "-"
        print("%-22s %-11s %-3s %6d %s %s  %s%s" % (l, s["group"], ch, s["record"], x, xr, "" if s["cr"] else "[not in CR] ", s["note"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
