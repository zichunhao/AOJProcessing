#!/usr/bin/env python3
# Tests of the semileptonic-ttbar selection / maker.
#
#   bash ttbar_semilep/py3env.sh ttbar_semilep/tests/test_ttbar.py         # plain runner (re-runs in el9 on el8)
#   python3 -m pytest ttbar_semilep/tests/test_ttbar.py -k toy              # if pytest + deps are there
#
# Toy tests need nothing.  Tests on real NanoAOD are SKIPPED when the file is missing:
#   AOJ_TT_TEST_MC     NanoAOD MC with GenPart (default: the 2018 TTToSemiLeptonic PFNano file on the UAF)
#   AOJ_TT_TEST_MC_ERA era of that file (default 2018)
#   AOJ_TT_TEST_PFNANO AOJ PFNano data file (default ../../run_2016H/nano_data2016.root, 64 events)
#   AOJ_TT_TEST_OUT    output directory (default: a fresh temporary directory)
# The real-file test compares the vectorised selection with an independent per-event loop.
import ast
import json
import math
import os
import re
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
AOJ = os.path.dirname(PKG)
sys.path.insert(0, PKG)
import ttbar_selection as ts  # noqa: E402

MC = os.environ.get("AOJ_TT_TEST_MC", "/home/users/zichun/parcel/kt-studies/TTbar/TTToSemiLeptonic_TuneCP5_13TeV-"
                    "powheg-pythia8/TTToSemiLeptonic/nano_mc2018_1-265.root")
MC_ERA = os.environ.get("AOJ_TT_TEST_MC_ERA", "2018")
PFNANO = os.environ.get("AOJ_TT_TEST_PFNANO", os.path.join(AOJ, "run_2016H", "nano_data2016.root"))
OUT = os.environ.get("AOJ_TT_TEST_OUT") or tempfile.mkdtemp(prefix="aoj_tt_test_")
os.makedirs(OUT, exist_ok=True)


class Skip(Exception):
    pass


def need(path):
    if not os.path.exists(path):
        raise Skip("missing input %s" % path)


def heavy():
    try:
        import uproot  # noqa: F401
        import awkward  # noqa: F401
        import h5py  # noqa: F401
    except ImportError as e:
        raise Skip("needs uproot/awkward/h5py (%s)" % e)


# =============================================================================== toy events
MU_COLS = {"pt": 0., "eta": 0., "phi": 0., "charge": -1, "tightId": True, "looseId": True, "pfRelIso04_all": 0.}
EL_COLS = {"pt": 0., "eta": 0., "phi": 0., "charge": -1, "deltaEtaSC": 0., "cutBased": 4,
           "mvaFall17V2Iso_WP80": True, "mvaFall17V2noIso_WP80": True, "pfRelIso03_all": 0.}
JET_COLS = {"pt": 0., "eta": 0., "phi": 0., "jetId": 6, "btagDeepFlavB": 0.}
FJ_COLS = dict({"pt": 0., "eta": 0., "phi": 0., "msoftdrop": 0., "jetId": 6},
               **dict((k, 0.) for k in ts.FATJET_TAGGING))


def coll(events, name, defaults):
    counts = [len(e.get(name, [])) for e in events]
    cols = {}
    for k, d in defaults.items():
        vals = [o.get(k, d) for e in events for o in e.get(name, [])]
        cols[k] = np.asarray(vals, dtype=np.asarray([d]).dtype)
    return ts.Coll(counts, cols)


def toy(events):
    """events: list of dicts with 'hlt' (list of fired paths), 'flags' (dict), 'met', 'met_phi',
    'lumi', and object lists 'Muon', 'Electron', 'Jet', 'FatJet' (dicts overriding the defaults)."""
    n = len(events)
    ev = {"run": np.ones(n, np.int64), "luminosityBlock": np.asarray([e.get("lumi", 1) for e in events], np.int64),
          "event": np.arange(n), "MET_pt": np.asarray([e.get("met", 80.) for e in events]),
          "MET_phi": np.asarray([e.get("met_phi", 0.5) for e in events])}
    paths = sorted(set(p for e in events for p in e.get("hlt", [])))
    for p in paths:
        ev[p] = np.asarray([p in e.get("hlt", []) for e in events])
    for fl in ts.aoj.AOJ_FILTERS:
        ev[fl] = np.asarray([e.get("flags", {}).get(fl, True) for e in events])
    ev["Muon"] = coll(events, "Muon", MU_COLS)
    ev["Electron"] = coll(events, "Electron", EL_COLS)
    ev["Jet"] = coll(events, "Jet", JET_COLS)
    ev["FatJet"] = coll(events, "FatJet", FJ_COLS)
    return ev


MU_TRIG = ts.ERAS["2016postVFP"]["triggers"]["an"]["mu"][0]
EL_TRIG = ts.ERAS["2016postVFP"]["triggers"]["an"]["el"][0]


def base_mu(**kw):
    """One event passing everything in the muon channel (2016postVFP, default "an" cuts)."""
    e = {"hlt": [MU_TRIG], "met": 80., "met_phi": 0.5,
         "Muon": [dict(pt=60., eta=0.5, phi=0., charge=-1, pfRelIso04_all=0.05)],
         "FatJet": [dict(pt=400., eta=0., phi=3.0, msoftdrop=170., tau1=0.3, tau2=0.2, tau3=0.08)],
         "Jet": [dict(pt=200., eta=0., phi=3.0, btagDeepFlavB=0.9),          # inside the probe: not a tag b
                 dict(pt=50., eta=0.5, phi=0.8, btagDeepFlavB=0.9)]}         # dR(mu) 0.8, dR(probe) 2.26
    e.update(kw)
    return e


def mod(e, name, i, **kw):
    e = json.loads(json.dumps(e))
    e[name][i].update(kw)
    return e


def stages(events, trigger="lepton", lumimask=None, era="2016postVFP", cuts=None):
    s = ts.select(toy(events), era=era, trigger=trigger, lumimask=lumimask, cuts=cuts)
    return s, s["stage_mu"].tolist(), s["stage_el"].tolist()


ALL = len(ts.CUTFLOW) - 1          # 8: passed everything
S = dict((k, i) for i, k in enumerate(ts.CUTFLOW))


def h5maker_lists():
    """triggers / filters exactly as written in ../H5_maker.py (the drift-proof reference)."""
    tree = ast.parse(open(os.path.join(AOJ, "H5_maker.py")).read())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("filters", "triggers"):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return found["triggers"], found["filters"]


def test_aoj_lists_identical_to_h5maker():
    """aoj_common.py (our copy) == H5_maker.py's lists, and == event_level/aoj_selection.py if present."""
    trig, fil = h5maker_lists()
    assert trig == ts.aoj.AOJ_TRIGGERS, "aoj_common.AOJ_TRIGGERS drifted from H5_maker.py"
    assert fil == ts.aoj.AOJ_FILTERS, "aoj_common.AOJ_FILTERS drifted from H5_maker.py"
    assert len(trig) == 85 and len(fil) == 8
    ev = os.path.join(AOJ, "event_level", "aoj_selection.py")
    if os.path.exists(ev):
        tree = ast.parse(open(ev).read())
        names = ("AOJ_TRIGGERS", "AOJ_FILTERS", "AOJ_JET_MIN_PT", "AOJ_JET_MAX_ABS_ETA", "AOJ_JET_ID")
        theirs = dict((n.targets[0].id, ast.literal_eval(n.value)) for n in ast.walk(tree)
                      if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                      and n.targets[0].id in names)
        ours = dict((k, getattr(ts.aoj, k)) for k in names)
        assert theirs == ours, "event_level/aoj_selection.py differs from aoj_common.py"


def test_shared_lists_and_h5maker_columns():
    """jet_tagging columns == H5_maker.py."""
    src = open(os.path.join(AOJ, "H5_maker.py")).read()
    m = re.search(r"jet_tagging\s*=\s*\[(.*?)\]", src, re.S)
    names = [x.strip().replace("jet.", "") for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    assert names == ts.FATJET_TAGGING, names
    assert len(ts.aoj.AOJ_TRIGGERS) == 85 and len(ts.aoj.AOJ_FILTERS) == 8
    assert ts.DEFAULT_CUTS["fj_min_pt"] == 300. and ts.DEFAULT_CUTS["fj_jet_id"] == 6
    assert ts.PRESETS["an"] == ts.DEFAULT_CUTS and ts.DEFAULT_CUTS["fj_count_min_pt"] == 300.
    assert all(set(v) == set(ts.DEFAULT_CUTS) for v in ts.PRESETS.values())


def test_toy_each_cut():
    """Default ("an") cuts, one toy event per cut / boundary."""
    b = base_mu()
    two_jets = lambda **kw: dict(b, FatJet=b["FatJet"] + [dict(dict(eta=1., phi=-1., msoftdrop=60.), **kw)])  # noqa: E731
    cases = [
        ("pass", b, ALL),
        ("no trigger", dict(b, hlt=[]), 0),
        ("electron trigger only", dict(b, hlt=[EL_TRIG]), 0),
        ("other muon path only (Mu50)", dict(b, hlt=["HLT_Mu50"]), 0),
        ("filter fails", dict(b, flags={"Flag_goodVertices": False}), S["trigger"]),
        ("muon not isolated", mod(b, "Muon", 0, pfRelIso04_all=0.2), S["filters_json"]),
        ("muon pt 50 (veto only)", mod(b, "Muon", 0, pt=50.), S["filters_json"]),
        ("muon pt == 55", mod(b, "Muon", 0, pt=55.), S["filters_json"]),
        ("muon |eta| 2.45", mod(b, "Muon", 0, eta=2.45), S["filters_json"]),
        ("muon not tightId", mod(b, "Muon", 0, tightId=False), S["filters_json"]),
        ("extra iso muon pt 35", dict(b, Muon=b["Muon"] + [dict(pt=35., eta=1., phi=2., tightId=False, pfRelIso04_all=0.1)]),
         S["filters_json"]),
        ("extra iso muon pt 25 (below veto)", dict(b, Muon=b["Muon"] + [dict(pt=25., eta=1., phi=2., pfRelIso04_all=0.1)]), ALL),
        ("extra non-iso muon pt 35", dict(b, Muon=b["Muon"] + [dict(pt=35., eta=1., phi=2., pfRelIso04_all=0.5)]), ALL),
        ("extra veto electron pt 35", dict(b, Electron=[dict(pt=35., eta=1., phi=2., cutBased=1, mvaFall17V2Iso_WP80=False)]),
         S["filters_json"]),
        ("extra electron pt 25", dict(b, Electron=[dict(pt=25., eta=1., phi=2., cutBased=1)]), ALL),
        ("extra electron failing veto ID", dict(b, Electron=[dict(pt=35., eta=1., phi=2., cutBased=0,
                                                                   mvaFall17V2Iso_WP80=False)]), ALL),
        ("MET 40", dict(b, met=40.), S["one_lepton"]),
        ("MET == 50", dict(b, met=50.), S["one_lepton"]),
        ("only AK8 has pt 250 (not AOJ)", mod(b, "FatJet", 0, pt=250.), S["met"]),
        ("probe pt == 300", mod(b, "FatJet", 0, pt=300.), S["met"]),
        ("probe |eta| 2.5", mod(b, "FatJet", 0, eta=2.5), S["met"]),
        ("probe jetId 2", mod(b, "FatJet", 0, jetId=2), S["met"]),
        ("second AK8 pt 170 (veto at 300)", two_jets(pt=170.), ALL),
        ("second AK8 pt 310", two_jets(pt=310.), S["met"]),
        ("second AK8 pt 350, mSD 30", two_jets(pt=350., msoftdrop=30.), S["met"]),
        ("second AK8 pt 150 (not counted)", two_jets(pt=150.), ALL),
        ("second AK8 |eta| 2.7 (not counted)", two_jets(pt=400., eta=2.7), ALL),
        ("probe mSD 40 (no mSD cut)", mod(b, "FatJet", 0, msoftdrop=40.), ALL),
        ("dphi(mu, probe) 1.9", mod(b, "FatJet", 0, phi=1.9), S["one_probe"]),
        ("dphi(mu, probe) == 2", mod(b, "FatJet", 0, phi=2.0), S["one_probe"]),
        ("b fails WP", mod(b, "Jet", 1, btagDeepFlavB=0.2), S["dphi"]),
        ("b at dR 0.45 of the muon", mod(b, "Jet", 1, eta=0.5, phi=0.45), S["dphi"]),
        ("b at dR 0.6 of the muon", mod(b, "Jet", 1, eta=0.5, phi=0.6), ALL),
        ("b at dR 1.2 of the probe", mod(b, "Jet", 1, eta=0.0, phi=1.8), S["dphi"]),
        ("b inside the probe only", dict(b, Jet=b["Jet"][:1]), S["dphi"]),
        ("b |eta| 2.45", mod(b, "Jet", 1, eta=2.45, phi=0.), S["dphi"]),
        ("b pt 35", mod(b, "Jet", 1, pt=35.), S["dphi"]),
        ("b jetId 0", mod(b, "Jet", 1, jetId=0), S["dphi"]),
        ("tau32 0.6", mod(b, "FatJet", 0, tau3=0.12), S["btag"]),
        ("tau32 == 0.5", mod(b, "FatJet", 0, tau3=0.1), S["btag"]),
    ]
    s, smu, sel = stages([c[1] for c in cases])
    bad = [(c[0], got, c[2]) for c, got in zip(cases, smu) if got != c[2]]
    assert not bad, bad
    assert all(x == 0 or x <= S["filters_json"] for x in sel)       # never an electron event
    assert s["selected"].tolist() == [st == ALL for st in smu]
    # tag-side quantities of the base event
    assert s["channel"][0] == 1 and s["lep_pdgId"][0] == 13 and s["probe_idx"][0] == 0 and s["n_ak8"][0] == 1
    assert s["n_b"][0] == 1 and s["b_idx"][0] == 1 and abs(s["tau32"][0] - 0.4) < 1e-6
    wx, wy = 60. + 80. * math.cos(0.5), 80. * math.sin(0.5)
    assert abs(s["lepW_pt"][0] - math.hypot(wx, wy)) < 1e-6
    assert abs(s["dphi_lep_probe"][0] - 3.0) < 1e-9


def test_toy_paper_preset():
    """The "paper" preset: other triggers, lepton > 50, AK8 count above 300, mSD > 50, no dphi, looser b cuts."""
    b = dict(base_mu(), hlt=["HLT_IsoMu24"])
    paper = ts.merged_cuts(None, "paper")
    cases = [
        ("pass", b, ALL, ALL),
        ("muon pt 52", mod(b, "Muon", 0, pt=52.), ALL, S["filters_json"]),
        ("second AK8 pt 170", dict(b, FatJet=b["FatJet"] + [dict(pt=170., eta=1., phi=-1.)]), ALL, ALL),
        ("probe mSD 40", mod(b, "FatJet", 0, msoftdrop=40.), S["met"], ALL),
        ("dphi 1.9", mod(b, "FatJet", 0, phi=1.9), ALL, S["one_probe"]),
        ("b pt 35", mod(b, "Jet", 1, pt=35.), ALL, S["dphi"]),
        ("extra iso muon pt 15", dict(b, Muon=b["Muon"] + [dict(pt=15., eta=1., phi=2., pfRelIso04_all=0.1)]),
         S["filters_json"], ALL),
    ]
    evs = [c[1] for c in cases]
    _, s_paper, _ = stages(evs, cuts=paper)
    an = ts.merged_cuts(None, "an")
    _, s_an, _ = stages([dict(e, hlt=[MU_TRIG]) for e in evs], cuts=an)
    bad = [(c[0], p, a, c[2], c[3]) for c, p, a in zip(cases, s_paper, s_an) if (p, a) != (c[2], c[3])]
    assert not bad, bad
    _, s_wrongtrig, _ = stages([dict(b, hlt=["HLT_IsoMu27"])])     # not an "an" 2016 path
    assert s_wrongtrig == [0]
    _, s_tk, _ = stages([dict(b, hlt=["HLT_IsoTkMu24"])])          # the other half of the 2016 muon OR
    assert s_tk == [ALL]
    # "an160": the analysis note's veto of any other AK8 jet above 160 GeV
    a160 = ts.merged_cuts(None, "an160")
    extra = [dict(pt=170., eta=1., phi=-1.)]
    _, s160, _ = stages([dict(b, hlt=[MU_TRIG]), dict(b, hlt=[MU_TRIG], FatJet=b["FatJet"] + extra)], cuts=a160)
    _, s300, _ = stages([dict(b, hlt=[MU_TRIG], FatJet=b["FatJet"] + extra)])
    assert s160 == [ALL, S["met"]] and s300 == [ALL], (s160, s300)


def test_toy_probe_choice_and_electron_channel():
    b = base_mu()
    fwd_first = dict(b, FatJet=[dict(pt=450., eta=2.7, phi=2.5, msoftdrop=90.), b["FatJet"][0]])
    el = dict(b, hlt=[EL_TRIG], Muon=[], Electron=[dict(pt=60., eta=0.5, phi=0., charge=1, deltaEtaSC=0.)])
    el_ele32 = dict(el, hlt=["HLT_Ele32_WPTight_Gsf"])
    el_gap = mod(el, "Electron", 0, eta=1.45, deltaEtaSC=0.05)
    el_eta = mod(el, "Electron", 0, eta=2.45)
    el_ele115 = dict(el, hlt=["HLT_Ele115_CaloIdVT_GsfTrkIdT"])
    el_mu_trig = dict(el, hlt=[MU_TRIG])
    el_not_wp80 = mod(el, "Electron", 0, mvaFall17V2Iso_WP80=False)
    s, smu, sel = stages([fwd_first, el, el_ele32, el_gap, el_eta, el_ele115, el_mu_trig, el_not_wp80])
    assert smu[0] == ALL and s["probe_idx"][0] == 1 and s["n_aoj_jets"][0] == 1 and s["n_ak8"][0] == 1
    assert sel == [0, ALL, ALL, S["filters_json"], S["filters_json"], 0, 0, S["filters_json"]], sel
    assert s["channel"].tolist() == [1, 2, 2, 0, 0, 2, 2, 0] and s["lep_pdgId"][1] == -11


def test_toy_skim_preset():
    """"skim": each loosened cut vs "an", the tag lepton / probe choice, and the "an" decision kept in an_channel."""
    b = base_mu()
    far = dict(pt=310., eta=1., phi=-1., msoftdrop=60.)
    noniso_el = dict(pt=60., eta=0.5, phi=0., charge=1, mvaFall17V2Iso_WP80=False, pfRelIso03_all=0.4)
    cases = [  # (name, event, "an" stage, "skim" stage, an_channel)
        ("pass", b, ALL, ALL, 1),
        ("muon relIso 0.5", mod(b, "Muon", 0, pfRelIso04_all=0.5), S["filters_json"], ALL, 0),
        ("muon relIso 1.1", mod(b, "Muon", 0, pfRelIso04_all=1.1), S["filters_json"], S["filters_json"], 0),
        ("muon pt 35", mod(b, "Muon", 0, pt=35.), S["filters_json"], ALL, 0),
        ("muon pt 25", mod(b, "Muon", 0, pt=25.), S["filters_json"], S["filters_json"], 0),
        ("MET 30", dict(b, met=30.), S["one_lepton"], ALL, 1),
        ("MET 15", dict(b, met=15.), S["one_lepton"], S["one_lepton"], 1),
        ("second AK8 pt 310", dict(b, FatJet=b["FatJet"] + [far]), S["met"], ALL, 1),
        ("extra non-iso tight muon pt 80", dict(b, Muon=b["Muon"] + [dict(pt=80., eta=1., phi=-2., pfRelIso04_all=0.5)]),
         ALL, ALL, 1),
        ("leading AK8 around the muon", dict(b, FatJet=[dict(pt=500., eta=0.5, phi=0.1, msoftdrop=40.)] + b["FatJet"]),
         S["met"], ALL, 1),
        ("Mu50 only, non-iso muon", dict(mod(b, "Muon", 0, pfRelIso04_all=0.5), hlt=["HLT_Mu50"]), 0, ALL, 0),
        ("muon + non-WP80 electron pt 90", dict(b, Electron=[dict(noniso_el, pt=90., eta=1., phi=-2.)]),
         S["filters_json"], ALL, 0),
    ]
    sk = ts.merged_cuts(None, "skim")
    s_sk, sk_mu, _ = stages([c[1] for c in cases], cuts=sk)
    s_an, an_mu, _ = stages([c[1] for c in cases])
    bad = [(c[0], a, k, c[2], c[3]) for c, a, k in zip(cases, an_mu, sk_mu) if (a, k) != (c[2], c[3])]
    assert not bad, bad
    assert s_sk["an_channel"].tolist() == [c[4] for c in cases], s_sk["an_channel"].tolist()
    assert s_sk["an_channel"].tolist() == s_an["an_channel"].tolist()          # the same decision in both presets
    assert s_an["an_channel"].tolist() == [1 if st >= S["one_lepton"] else 0 for st in an_mu]
    i = dict((c[0], k) for k, c in enumerate(cases))
    assert abs(s_sk["lep_reliso"][i["muon relIso 0.5"]] - 0.5) < 1e-9 and not s_sk["lep_id_iso"][i["muon relIso 0.5"]]
    assert s_sk["lep_id_iso"][i["pass"]] and s_sk["n_lep_tight"][i["pass"]] == 1
    k = i["extra non-iso tight muon pt 80"]                    # the "an"-tight muon stays the tag, not the harder one
    assert s_sk["lep_pt"][k] == 60. and s_sk["n_lep_tight"][k] == 2 and s_an["n_lep_tight"][k] == 1
    k = i["second AK8 pt 310"]
    assert s_sk["n_ak8"][k] == 2 and s_sk["probe_idx"][k] == 0 and abs(s_sk["ak8_other_pt"][k] - 310.) < 1e-9
    assert np.isnan(s_sk["ak8_other_pt"][i["pass"]])
    assert s_sk["probe_idx"][i["leading AK8 around the muon"]] == 1        # dR(mu) 0.1 < 0.8: not the probe
    k = i["Mu50 only, non-iso muon"]
    assert s_sk["trig_mu"][k] and not s_sk["trig_mu_an"][k] and s_an["trig_mu_an"][i["pass"]]
    # electron channel: no-iso WP80 + relIso03 < 1 in the skim, the iso WP80 in "an"
    el = dict(b, hlt=[EL_TRIG], Muon=[], Electron=[noniso_el])
    el_iso = mod(el, "Electron", 0, mvaFall17V2Iso_WP80=True, pfRelIso03_all=0.05)
    el_loose = mod(el, "Electron", 0, pfRelIso03_all=1.2)
    s2, _, sk_el = stages([el, el_iso, el_loose], cuts=sk)
    _, _, an_el = stages([el, el_iso, el_loose])
    assert sk_el == [ALL, ALL, S["filters_json"]] and an_el == [S["filters_json"], ALL, S["filters_json"]], (sk_el, an_el)
    assert s2["channel"].tolist() == [2, 2, 0] and s2["an_channel"].tolist() == [0, 2, 0]
    assert abs(s2["lep_reliso"][0] - 0.4) < 1e-9 and s2["lep_id_iso"].tolist() == [False, True, False]
    # the skim trigger set = the "an" paths + the high-pt ones
    T = ts.ERAS["2016postVFP"]["triggers"]
    assert T["skim"]["mu"] == ["HLT_IsoMu24", "HLT_IsoTkMu24", "HLT_Mu50", "HLT_TkMu50"]
    assert T["skim"]["el"] == ["HLT_Ele27_WPTight_Gsf", "HLT_Ele32_WPTight_Gsf", "HLT_Ele115_CaloIdVT_GsfTrkIdT"]


def test_samples_manifest():
    """samples.py: every CR sample has a public (eospublic) file list and an MC cross section, each channel's QCD
    comes from one sample set, data channels follow the primary dataset, and condor/job.sh's data-vs-MC decision
    (by label) agrees with the manifest."""
    import re
    import subprocess
    import samples as smp
    seen = {}
    for label, s in smp.SAMPLES.items():
        assert s["group"] in smp.GROUPS and set(s["channels"]) <= {1, 2} and s["channels"], label
        path = smp.file_list(label, AOJ)
        if s["cr"]:
            lines = [x.strip() for x in open(path) if x.strip()]
            assert lines and all(x.startswith(smp.EOSPUBLIC) for x in lines), label
            assert len(set(lines)) == len(lines), label
            for c in s["channels"]:
                seen.setdefault(c, set()).add(s["group"])
        if s["group"] == "data":
            assert s["channels"] == ((1,) if label.startswith("SingleMuon") else (2,)), label
        else:
            assert s["xsec_pb"] > 0 and s["dataset"].endswith("/MINIAODSIM"), label
    assert "qcd_mu" in seen[1] and "qcd" not in seen[1] and "qcd" in seen[2] and "qcd_mu" not in seen[2]
    assert not any(g == "qcd_ht" for c in seen for g in seen[c])
    case = re.search(r'case "\$DATASET" in\n(.*?)\nesac', open(os.path.join(AOJ, "condor", "job.sh")).read(), re.S).group(1)
    for label, s in smp.SAMPLES.items():
        got = subprocess.run(["bash", "-c", 'DATASET="$1"; case "$DATASET" in\n%s\nesac; echo $STYPE' % case, "_", label],
                             capture_output=True, text=True).stdout.strip()
        assert got == ("data" if s["group"] == "data" else "MC"), (label, got)


def test_toy_2d_isolation_and_ak4():
    """v3 bookkeeping: the 2D-isolation jet is found after taking the lepton out of its own jet (Muon_jetIdx), and
    the per-event AK4 array (tight ID, pT > 20, |eta| < 2.5, pT order, NaN padded)."""
    b = base_mu()
    own = dict(pt=65., eta=0.5, phi=0.0, btagDeepFlavB=0.1)            # the muon (pt 60) + 5 GeV: its own jet
    events = [dict(b, Jet=b["Jet"] + [own]),                            # isolated muon: nearest other jet at dR 0.8
              dict(b, Jet=b["Jet"] + [dict(own, pt=110.)]),             # muon inside a 50 GeV jet: dR ~ 0
              dict(b, Jet=b["Jet"] + [own]),                            # no jetIdx: the uncleaned jet, dR 0
              dict(b, Jet=b["Jet"] + [dict(pt=30., eta=2.45, phi=1., jetId=6), dict(pt=25., eta=0., phi=2., jetId=0),
                                      dict(pt=15., eta=0., phi=2.)])]
    ev = toy(events)
    ev["Muon"].cols["jetIdx"] = np.array([2, 2, -1, -1])
    ev["Jet"].cols["rawFactor"] = np.zeros(len(ev["Jet"].evt))
    ev["Jet"].cols["hadronFlavour"] = np.where(ev["Jet"]["btagDeepFlavB"] > 0.5, 5, 0)
    s = ts.select(ev, era="2016postVFP", cuts=ts.merged_cuts(None, "skim"))
    dr = s["lep_jet_dr"]
    assert abs(dr[0] - 0.8) < 1e-6 and dr[1] < 1e-6 and dr[2] < 1e-6 and abs(dr[3] - 0.8) < 1e-6, dr
    lv = 60. * np.array([1., 0., math.sinh(0.5)])                     # lepton vs the 50 GeV jet at dphi 0.8
    jv = 50. * np.array([math.cos(0.8), math.sin(0.8), math.sinh(0.5)])
    assert abs(s["lep_jet_ptrel"][0] - np.linalg.norm(np.cross(lv, jv)) / np.linalg.norm(jv)) < 1e-6
    a = s["ak4"]
    assert a.shape == (4, ts.N_AK4, len(ts.AK4_COLUMNS))
    c = dict((k, i) for i, k in enumerate(ts.AK4_COLUMNS))
    assert a[0, :3, c["pt"]].tolist() == [200., 50., 65.] and np.isnan(a[0, 3:]).all()
    assert a[3, :3, c["pt"]].tolist() == [200., 50., 30.] and np.isnan(a[3, 3:]).all()   # jetId 0 and pT 15 left out
    assert a[0, :2, c["hadronFlavour"]].tolist() == [5., 5.] and a[0, 2, c["hadronFlavour"]] == 0.
    assert abs(a[0, 1, c["dr_lep"]] - 0.8) < 1e-6 and abs(a[0, 0, c["dr_probe"]]) < 1e-6
    del ev["Muon"].cols["jetIdx"], ev["Jet"].cols["hadronFlavour"]       # inputs without them: no cleaning, NaN flavour
    s = ts.select(ev, era="2016postVFP", cuts=ts.merged_cuts(None, "skim"))
    assert s["lep_jet_dr"][0] < 1e-6 and np.isnan(s["ak4"][0, :3, c["hadronFlavour"]]).all()


def test_toy_trigger_modes_json_and_eras():
    b = base_mu()
    jet_only = dict(b, hlt=["HLT_PFJet500"])
    missing_branch = dict(b, hlt=["HLT_SomethingElse"])
    evs = [b, jet_only, missing_branch, dict(b, lumi=7)]
    lm = ts.aoj.LumiMask(ranges={"1": [[1, 5]]})
    _, smu, _ = stages(evs, trigger="lepton", lumimask=lm)
    assert smu == [ALL, 0, 0, S["trigger"]], smu
    _, smu, _ = stages(evs, trigger="jetht")
    assert smu == [0, ALL, 0, 0], smu
    _, smu, _ = stages(evs, trigger="lepton_or_jetht")
    assert smu == [ALL, ALL, 0, ALL], smu
    _, smu, _ = stages(evs, trigger="none")
    assert smu == [ALL] * 4
    # DeepJet medium WP differs per era; the event fires the 2016 and the 2018 "an" muon paths
    wp = dict(mod(b, "Jet", 1, btagDeepFlavB=0.26), hlt=[MU_TRIG] + ts.ERAS["2018"]["triggers"]["an"]["mu"])
    # (0.26 > 0.2489 = 2016post M, < 0.2783 = 2018 M)
    _, s16, _ = stages([wp], era="2016postVFP")
    _, s18, _ = stages([wp], era="2018")
    assert s16 == [ALL] and s18 == [S["dphi"]], (s16, s18)
    # cut overrides
    _, smu, _ = stages([dict(b, met=40.)], cuts=ts.parse_overrides(["met_min=30"]))
    assert smu == [ALL]
    for bad in ({"no_such_cut": 1},):
        try:
            ts.merged_cuts(bad)
            raise AssertionError("unknown cut accepted")
        except KeyError:
            pass
    try:
        ts.merged_cuts(None, "no_such_preset")
        raise AssertionError("unknown preset accepted")
    except KeyError:
        pass


def test_toy_cutflow_weights():
    b = base_mu()
    s, smu, _ = stages([b, dict(b, met=40.), dict(b, hlt=[])])
    cf = ts.cutflow(s)
    assert cf[:, 0].tolist() == [3, 2, 2, 2, 1, 1, 1, 1, 1], cf[:, 0]
    assert cf[:, 1].tolist() == [3, 0, 0, 0, 0, 0, 0, 0, 0]
    cfw = ts.cutflow(s, weights=[2., 3., 5.])
    assert cfw[:, 0].tolist() == [10, 5, 5, 5, 2, 2, 2, 2, 2]


# =============================================================================== toy GenPart
def gen_event(b, u, d, extra=None, top_pt=400.):
    """t -> W+ b, W+ -> W+ (copy) -> u dbar; tbar -> W- bbar, W- -> mu- nubar.  b/u/d: (eta, phi)."""
    P = [  # pdg, mother, pt, eta, phi
        (21, -1, 0., 0., 0.),          # 0 g
        (6, 0, top_pt, 0., 0.),        # 1 t
        (-6, 0, 300., 1.5, 2.6),       # 2 tbar
        (6, 1, top_pt, 0., 0.),        # 3 t (last copy)
        (24, 3, 200., 0., 0.),         # 4 W+
        (5, 3, 100., b[0], b[1]),      # 5 b
        (24, 4, 200., 0., 0.),         # 6 W+ (last copy)
        (2, 6, 100., u[0], u[1]),      # 7 u
        (-1, 6, 100., d[0], d[1]),     # 8 dbar
        (-24, 2, 150., 1.4, 2.5),      # 9 W-
        (-5, 2, 100., 1.7, 2.9),       # 10 bbar
        (13, 9, 80., 1.2, 2.4),        # 11 mu-
        (-14, 9, 70., 1.5, 2.7),       # 12 nubar
    ]
    return P + (extra or [])


def gen_coll(events):
    counts = [len(e) for e in events]
    flat = [p for e in events for p in e]
    cols = {"pdgId": np.asarray([p[0] for p in flat], np.int32),
            "genPartIdxMother": np.asarray([p[1] for p in flat], np.int32),
            "pt": np.asarray([p[2] for p in flat], np.float32),
            "eta": np.asarray([p[3] for p in flat], np.float32),
            "phi": np.asarray([p[4] for p in flat], np.float32)}
    return ts.Coll(counts, cols)


def test_toy_gen_truth():
    far = (2.0, -2.0)
    evs = [
        gen_event((0.1, 0.1), (0.3, 0.0), (-0.3, 0.2)),              # t
        gen_event((0.9, 0.0), (0.3, 0.0), (-0.3, 0.2)),              # W (b just outside)
        gen_event((0.1, 0.0), (0.3, 0.0), (1.5, 0.0)),               # bq
        gen_event(far, (0.0, 2.0), (1.5, 1.0)),                      # other
        gen_event((0.1, 0.1), (0.3, 0.0), (-0.3, 0.2)),              # no probe -> other
    ]
    # all-hadronic: the second W decays to quarks merged with the probe at (1.5, 2.6); first top far away
    ah = gen_event(far, far, far)
    ah = [p for p in ah if p[0] not in (13, -14)] + [(1, 9, 80., 1.4, 2.5), (-2, 9, 70., 1.6, 2.7)]
    evs.append(ah)
    # W+jets-like: a W from a quark (no top) decaying to quarks merged with the probe
    evs.append([(2, -1, 0., 0., 0.), (24, 0, 300., 0., 0.), (2, 1, 150., 0.2, 0.1), (-1, 1, 150., -0.2, -0.1)])
    gen = gen_coll(evs)
    pe = np.array([0., 0., 0., 0., np.nan, 1.5, 0.])
    pp = np.array([0., 0., 0., 0., np.nan, 2.6, 0.])
    g = ts.gen_truth(gen, pe, pp)
    assert g["category"].tolist() == [3, 2, 1, 0, 0, 3, 2], g["category"]
    assert g["n_had_W"].tolist() == [1, 1, 1, 1, 1, 2, 1]
    assert g["n_lep_W"].tolist() == [1, 1, 1, 1, 1, 0, 0]
    assert g["lep_flav"].tolist() == [13, 13, 13, 13, 13, 0, 0]
    assert abs(g["had_top_pt"][0] - 400.) < 1e-3 and abs(g["had_top_dr"][0]) < 1e-6
    assert abs(g["had_top_pt"][5] - 300.) < 1e-3          # nearest hadronic top to the probe = the tbar
    # no probe: the hadronic top is still reported (dR undefined); no top at all (W+jets): NaN
    assert abs(g["had_top_pt"][4] - 400.) < 1e-3 and np.isnan(g["had_top_dr"][4])
    assert np.isnan(g["had_top_pt"][6]) and np.isnan(g["had_top_dr"][6])


# =============================================================================== toy NanoAOD end to end
def write_toy_nano(path, is_mc):
    import uproot
    import awkward as ak

    def jag(x, dt):
        return ak.values_astype(ak.Array(x), dt)
    b = base_mu()
    evs = [b, dict(b, met=40.), mod(b, "FatJet", 0, tau3=0.12), dict(b, lumi=9),
           dict(b, hlt=[EL_TRIG], Muon=[], Electron=[dict(pt=70., eta=-0.4, phi=0.1, charge=1)])]
    n = len(evs)
    ev = toy(evs)
    d = {"run": np.ones(n, np.uint32), "luminosityBlock": ev["luminosityBlock"].astype(np.uint32),
         "event": np.arange(100, 100 + n, dtype=np.uint64),
         "MET_pt": ev["MET_pt"].astype(np.float32), "MET_phi": ev["MET_phi"].astype(np.float32)}
    for k, v in ev.items():
        if k.startswith("HLT_") or k.startswith("Flag_"):
            d[k] = v.astype(bool)
    dts = {"charge": np.int32, "tightId": bool, "looseId": bool, "cutBased": np.int32, "mvaFall17V2Iso_WP80": bool,
           "jetId": np.int32, "nConstituents": np.uint8}
    for name in ("Muon", "Electron", "Jet", "FatJet"):
        c = ev[name]
        d[name] = ak.zip(dict((k, jag(ak.unflatten(c[k], c.counts).tolist(), dts.get(k, np.float32))) for k in c.cols))
    # PF candidates: event 0's probe links 160 candidates with pt 1..160 in link order -> H5_maker keeps the
    # FIRST 150 (pt 1..150) and sorts them; other events: 3 candidates
    pf_pt = [list(np.arange(1., 161.))] + [[5., 9., 7.]] * (n - 1)
    links_j = [[0] * 160] + [[0, 0, 0]] * (n - 1)
    links_p = [list(range(160))] + [[2, 0, 1]] * (n - 1)

    def per(x, val):
        return [[val] * len(r) for r in x]
    d["PFCands"] = ak.zip({"pt": jag(pf_pt, np.float32), "eta": jag(per(pf_pt, 0.1), np.float32),
                           "phi": jag(per(pf_pt, 3.0), np.float32), "mass": jag(per(pf_pt, 0.14), np.float32),
                           "d0": jag(per(pf_pt, 0.01), np.float32), "d0Err": jag(per(pf_pt, 0.02), np.float32),
                           "dz": jag(per(pf_pt, 0.03), np.float32), "dzErr": jag(per(pf_pt, 0.04), np.float32),
                           "charge": jag(per(pf_pt, 1), np.int32), "pdgId": jag(per(pf_pt, 211), np.int32),
                           "puppiWeight": jag(per(pf_pt, 1.), np.float32)})
    d["FatJetPFCands"] = ak.zip({"jetIdx": jag(links_j, np.int32), "pFCandsIdx": jag(links_p, np.int32)})
    if is_mc:
        d["genWeight"] = np.full(n, 2.0, np.float32)
        gp = [gen_event((0.1, 3.1), (0.3, 3.0), (-0.3, 2.9))] * n          # merged around the probe (0, 3.0)
        g = gen_coll(gp)
        d["GenPart"] = ak.zip(dict((k, jag(ak.unflatten(g[k], g.counts).tolist(),
                                            np.int32 if k in ("pdgId", "genPartIdxMother") else np.float32))
                                   for k in g.cols))
    with uproot.recreate(path) as f:
        f["Events"] = d
        if is_mc:
            f["Runs"] = {"run": np.array([1], np.uint32), "genEventSumw": np.array([123.0])}
        # lumi 3 has no event (e.g. all filtered out before the output): it is still a processed lumi section
        f["LuminosityBlocks"] = {"run": np.array([1, 1, 1], np.uint32), "luminosityBlock": np.array([1, 3, 9], np.uint32)}
    return n


def test_toy_nanoaod_end_to_end():
    heavy()
    import h5py
    import ttbar_maker
    for is_mc in (False, True):
        tf = os.path.join(OUT, "toy_tt_%s.root" % ("mc" if is_mc else "data"))
        write_toy_nano(tf, is_mc)
        jf = os.path.join(OUT, "toy_tt_golden.json")
        json.dump({"1": [[1, 5]]}, open(jf, "w"))                 # lumi 9 (event 3) uncertified
        out = os.path.join(OUT, "toy_tt_%d.h5" % is_mc)
        args = ["-i", tf, "-o", out, "--sample_type", "MC" if is_mc else "data"] + ([] if is_mc else ["-j", jf])
        ttbar_maker.main(args)
        with h5py.File(out, "r") as f:
            ei = f["event_info"][:]
            # stored (stage >= btag): 0 (passes), 2 (fails tau32 only), 4 (electron); 1 fails MET; 3 JSON (data)
            exp = [100, 102, 103, 104] if is_mc else [100, 102, 104]
            assert ei[:, 2].tolist() == exp, ei
            tag = f["ttbar_tag"][:]
            col = f["ttbar_tag"].attrs["columns"].split(",")
            tc = dict((k, tag[:, i]) for i, k in enumerate(col))
            assert tc["pass_tau32"].tolist() == ([1, 0, 1, 1] if is_mc else [1, 0, 1])
            assert tc["channel"].tolist() == ([1, 1, 1, 2] if is_mc else [1, 1, 2])
            assert tc["stage"].tolist() == ([ALL, S["btag"], ALL, ALL] if is_mc else [ALL, S["btag"], ALL])
            assert f["jet_kinematics"].shape == (len(exp), 4) and f["jet_tagging"].shape == (len(exp), 13)
            assert abs(f["jet_kinematics"][0, 3] - 170.) < 1e-3
            assert f["jet_tagging"].attrs["columns"].split(",") == ts.FATJET_TAGGING
            pf = f["PFCands"][:]
            assert pf.shape == (len(exp), 150, 11)
            pt0 = np.hypot(pf[0, :, 0], pf[0, :, 1])
            # first 150 in link order (pt 1..150) then pt-descending; the pt 151..160 candidates are dropped
            assert abs(pt0[0] - 150.) < 1e-3 and abs(pt0[149] - 1.) < 1e-3 and np.all(np.diff(pt0) <= 1e-4)
            e0 = math.sqrt((150. * math.cosh(0.1)) ** 2 + 0.14 ** 2)
            assert abs(pf[0, 0, 3] - e0) < 1e-2 and pf[0, 0, 9] == 211 and abs(pf[0, 0, 10] - 1.) < 1e-6
            pt1 = np.hypot(pf[1, :3, 0], pf[1, :3, 1])
            assert np.allclose(pt1, [9., 7., 5.], atol=1e-4) and not pf[1, 3:].any()
            # processed lumi sections, certified ones only in data (lumi 9 is not)
            assert f["lumis"][:].tolist() == ([[1, 1], [1, 3], [1, 9]] if is_mc else [[1, 1], [1, 3]]), f["lumis"][:]
            ak4 = f["ak4"][:]
            assert ak4.shape == (len(exp), ts.N_AK4, len(ts.AK4_COLUMNS))
            assert ak4[0, :2, 0].tolist() == [200., 50.] and np.isnan(ak4[0, 2:]).all()
            cf = f["cutflow/n"][:]
            # all, trigger (event 4 is an electron), filters_json (lumi 9 not certified, data), one_lepton,
            # met (event 1), one_probe, dphi, btag, tau32 (event 2)
            n_mu = [5, 4, 4, 4, 3, 3, 3, 3, 2] if is_mc else [5, 4, 3, 3, 2, 2, 2, 2, 1]
            assert cf[:, 0].tolist() == n_mu, cf[:, 0]
            assert cf[-1, 1] == 1
            if is_mc:
                assert f.attrs["sum_genweight"] == 123.0
                assert f["gen_weight"][:].tolist() == [2.0] * 4
                assert f["cutflow/w"][0, 0] == 10.0
                tr = f["truth"][:]
                tcol = f["truth"].attrs["columns"].split(",")
                assert tr[:, tcol.index("category")].tolist() == [3, 3, 3, 3]
            else:
                assert "truth" not in f and "gen_weight" not in f
        # --store tau32 keeps the final selection only
        out2 = os.path.join(OUT, "toy_tt_final_%d.h5" % is_mc)
        ttbar_maker.main(args[:3] + [out2] + args[4:] + ["--store", "tau32", "--no-pf"])
        with h5py.File(out2, "r") as f:
            assert f["event_info"][:, 2].tolist() == ([100, 103, 104] if is_mc else [100, 104])
            assert "PFCands" not in f


def test_toy_summary():
    """ttbar_summary.py on a toy MC output: normalisation (xsec * lumi / sum_genweight) and levels."""
    heavy()
    import ttbar_maker
    import ttbar_summary
    tf = os.path.join(OUT, "toy_tt_sum.root")
    write_toy_nano(tf, True)
    out = os.path.join(OUT, "toy_tt_sum.h5")
    ttbar_maker.main(["-i", tf, "-o", out, "--sample_type", "MC", "--store", "one_probe", "--no-pf"])
    js = os.path.join(OUT, "toy_tt_sum.json")
    ttbar_summary.main(["--lumi", "1.0", "--mc", "toy:123:" + out, "--json", js])
    r = json.load(open(js))["samples"]["toy"]
    norm = 123 * 1.0 * 1e3 / 123.0                       # xsec * lumi[pb^-1] / sum_genweight
    assert abs(r["norm"] - norm) < 1e-9
    # stored: events 0, 2, 3 (mu) and 4 (el), genWeight 2 each; event 2 fails tau32 only
    assert abs(r["levels"]["btag"]["all"]["300-inf"]["yield"] - 4 * 2 * norm) < 1e-6
    assert abs(r["levels"]["final"]["mu"]["300-inf"]["yield"] - 2 * 2 * norm) < 1e-6
    assert r["levels"]["final"]["all"]["300-inf"]["frac"]["t"] == 1.0


# =============================================================================== real NanoAOD
def reference_loop(path, era, n_max=None, cuts=None):
    """Independent per-event implementation of the selection (plain python loops) -> list of
    (event, stage_mu, stage_el, probe_idx, n_b)."""
    import uproot
    c = cuts or ts.DEFAULT_CUTS
    T = ts.ERAS[era]["triggers"][c["trigger_set"]]
    t = uproot.open(path)["Events"]
    keys = set(t.keys())
    trig = [x for x in set(T["mu"] + T["el"]) if x in keys]
    filt = [x for x in ts.aoj.AOJ_FILTERS if x in keys]
    names = (["event", "MET_pt"] + trig + filt
             + ["Muon_" + k for k in ("pt", "eta", "phi", "tightId", "looseId", "pfRelIso04_all")]
             + ["Electron_" + k for k in ("pt", "eta", "phi", "deltaEtaSC", "cutBased", c["el_id"])]
             + ["Jet_" + k for k in ("pt", "eta", "phi", "jetId", "btagDeepFlavB")]
             + ["FatJet_" + k for k in ("pt", "eta", "phi", "msoftdrop", "jetId", "tau2", "tau3")])
    a = t.arrays(names, entry_stop=n_max, library="np")
    wp = ts.ERAS[era]["btag_wp"][c["b_wp"]]

    def dphi(p1, p2):
        return abs((p1 - p2 + math.pi) % (2 * math.pi) - math.pi)

    def dr(e1, p1, e2, p2):
        return math.sqrt((e1 - e2) ** 2 + dphi(p1, p2) ** 2)
    res = []
    for i in range(len(a["event"])):
        tmu = any(a[x][i] for x in T["mu"] if x in a)
        tel = any(a[x][i] for x in T["el"] if x in a)
        fok = all(a[x][i] for x in filt)
        mus, vmu, els, vel = [], 0, [], 0
        for j in range(len(a["Muon_pt"][i])):
            pt, eta, iso = a["Muon_pt"][i][j], a["Muon_eta"][i][j], a["Muon_pfRelIso04_all"][i][j]
            if abs(eta) >= c["mu_max_abs_eta"]:
                continue
            tight = pt > c["lep_min_pt"] and a["Muon_tightId"][i][j] and iso < c["mu_max_reliso"]
            if tight:
                mus.append(j)
            if tight or (pt > c["veto_min_pt"] and a["Muon_looseId"][i][j] and iso < c["veto_mu_max_reliso"]):
                vmu += 1
        for j in range(len(a["Electron_pt"][i])):
            pt = a["Electron_pt"][i][j]
            sc = abs(a["Electron_eta"][i][j] + a["Electron_deltaEtaSC"][i][j])
            if sc >= c["el_max_abs_eta"]:
                continue
            tight = pt > c["lep_min_pt"] and not (1.4442 < sc < 1.5660) and a["Electron_" + c["el_id"]][i][j]
            if tight:
                els.append(j)
            if tight or (pt > c["veto_min_pt"] and a["Electron_cutBased"][i][j] >= c["veto_el_min_cutbased"]):
                vel += 1
        is_mu = len(mus) == 1 and vmu == 1 and vel == 0
        is_el = len(els) == 1 and vel == 1 and vmu == 0
        le = lp = None
        if is_mu:
            le, lp = a["Muon_eta"][i][mus[0]], a["Muon_phi"][i][mus[0]]
        elif is_el:
            le, lp = a["Electron_eta"][i][els[0]], a["Electron_phi"][i][els[0]]
        counted = [j for j in range(len(a["FatJet_pt"][i]))
                   if a["FatJet_pt"][i][j] > c["fj_count_min_pt"] and abs(a["FatJet_eta"][i][j]) < c["fj_max_abs_eta"]]
        pidx = counted[0] if counted else -1
        probe_ok = (len(counted) == 1 and a["FatJet_pt"][i][pidx] > c["fj_min_pt"] and a["FatJet_jetId"][i][pidx] == c["fj_jet_id"]
                    and (c["fj_min_msd"] <= 0 or a["FatJet_msoftdrop"][i][pidx] > c["fj_min_msd"]))
        nb, dphi_ok = 0, False
        if probe_ok and le is not None:
            fe, fp = a["FatJet_eta"][i][pidx], a["FatJet_phi"][i][pidx]
            dphi_ok = c["min_dphi_lep_probe"] <= 0 or dphi(lp, fp) > c["min_dphi_lep_probe"]
            for j in range(len(a["Jet_pt"][i])):
                je, jp = a["Jet_eta"][i][j], a["Jet_phi"][i][j]
                if (a["Jet_pt"][i][j] > c["b_min_pt"] and abs(je) < c["b_max_abs_eta"] and (int(a["Jet_jetId"][i][j]) & 2)
                        and a["Jet_btagDeepFlavB"][i][j] > wp and dr(je, jp, fe, fp) > c["b_min_dr_fj"]
                        and dr(je, jp, le, lp) > c["b_min_dr_lep"]):
                    nb += 1
        tau = (a["FatJet_tau3"][i][pidx] / a["FatJet_tau2"][i][pidx]) if (pidx >= 0 and a["FatJet_tau2"][i][pidx] > 0) else 9.
        st = []
        for t_ok, is_ch in ((tmu, is_mu), (tel, is_el)):
            steps = [t_ok, fok, is_ch, a["MET_pt"][i] > c["met_min"], probe_ok, dphi_ok, nb >= 1, tau < c["tau32_max"]]
            k = 0
            while k < len(steps) and steps[k]:
                k += 1
            st.append(k)
        res.append((int(a["event"][i]), st[0], st[1], pidx, nb))
    return res


def read_selection(path, era, n_max=None, cuts=None):
    import uproot
    import ttbar_maker
    t = uproot.open(path)["Events"]
    is_mc = "genWeight" in t.keys()
    warn = ttbar_maker.Warner()
    rd = ttbar_maker.Reader(t.keys(), era, cuts, is_mc, False, warn)
    ev = rd.convert(t.arrays(rd.names(), entry_stop=n_max, library="ak"))
    return ev, ts.select(ev, era=era, cuts=cuts)


def test_real_mc_matches_reference_loop():
    """Both presets: vectorised selection == independent per-event loop on real NanoAOD."""
    heavy()
    need(MC)
    n_max = 20000
    for preset in ("an", "an160", "paper"):
        cuts = ts.merged_cuts(None, preset)
        ev, s = read_selection(MC, MC_ERA, n_max, cuts)
        ref = reference_loop(MC, MC_ERA, n_max, cuts)
        assert len(ref) == len(s["stage_mu"])
        bad = []
        for i, (evn, smu, sel, pidx, nb) in enumerate(ref):
            got = (int(ev["event"][i]), int(s["stage_mu"][i]), int(s["stage_el"][i]))
            if got != (evn, smu, sel):
                bad.append((i, got, (evn, smu, sel)))
            elif max(smu, sel) >= S["dphi"] and (int(s["probe_idx"][i]) != pidx or int(s["n_b"][i]) != nb):
                bad.append((i, "probe/b", (int(s["probe_idx"][i]), int(s["n_b"][i])), (pidx, nb)))
        assert not bad, (preset, bad[:10])
        st = np.maximum(s["stage_mu"], s["stage_el"])
        print("    %s: %d events: %d reach one_probe, %d dphi, %d btag, %d selected" % (
            preset, len(st), (st >= S["one_probe"]).sum(), (st >= S["dphi"]).sum(), (st >= S["btag"]).sum(), (st >= ALL).sum()))
        assert (st >= S["btag"]).sum() > 0


def test_real_mc_truth_semileptonic():
    """TTToSemiLeptonic: every event has exactly one hadronic and one leptonic W decay; merged tops at high pt."""
    heavy()
    need(MC)
    import uproot
    t = uproot.open(MC)["Events"]
    if "nGenPart" not in t.keys():
        raise Skip("no GenPart")
    ev, s = read_selection(MC, MC_ERA, 20000)
    g = ts.gen_truth(ev["GenPart"], s["probe_eta"], s["probe_phi"])
    frac_ok = np.mean((g["n_had_W"] == 1) & (g["n_lep_W"] == 1))
    assert frac_ok > 0.99, frac_ok
    has = np.isfinite(s["probe_pt"])
    hi = has & (s["probe_pt"] > 500)
    lo = has & (s["probe_pt"] < 400)
    f_hi, f_lo = np.mean(g["category"][hi] == 3), np.mean(g["category"][lo] == 3)
    print("    fully-merged fraction of probe candidates: pt<400 %.2f (%d)  pt>500 %.2f (%d)" % (f_lo, lo.sum(), f_hi, hi.sum()))
    assert f_hi > f_lo


def test_real_pfnano_pfcands_like_h5maker():
    """pf_rows on the AOJ PFNano data file == H5_maker's construction (independent loop) for every AOJ jet."""
    heavy()
    need(PFNANO)
    import uproot
    import ttbar_maker
    t = uproot.open(PFNANO)["Events"]
    rd = ttbar_maker.Reader(t.keys(), "2016postVFP", None, False, True, ttbar_maker.Warner())
    assert rd.pf
    ev = rd.convert(t.arrays(rd.names(), library="ak"))
    fj = ev["FatJet"]
    gidx = np.nonzero(ts.aoj.aoj_jet_mask(fj["pt"], fj["eta"], fj["jetId"]))[0]
    got = ttbar_maker.pf_rows(ev, gidx)
    a = t.arrays(["FatJetPFCands_jetIdx", "FatJetPFCands_pFCandsIdx"] + ["PFCands_" + k for k in ttbar_maker.PF_COLS],
                 library="np")
    for r, g in enumerate(gidx):
        i, j = int(fj.evt[g]), int(g - fj.offsets[fj.evt[g]])
        idx = a["FatJetPFCands_pFCandsIdx"][i][a["FatJetPFCands_jetIdx"][i] == j][:150]
        rows = []
        for k in idx:
            pt, eta, phi, m = (float(a["PFCands_" + x][i][k]) for x in ("pt", "eta", "phi", "mass"))
            px, py, pz = pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta)
            rows.append([px, py, pz, math.sqrt(px * px + py * py + pz * pz + m * m)]
                        + [float(a["PFCands_" + x][i][k]) for x in ttbar_maker.PF_COLS[4:]])
        rows = np.asarray(rows, np.float32).reshape(-1, 11)
        rows = rows[np.flip(np.argsort(np.sqrt(rows[:, 0] ** 2 + rows[:, 1] ** 2)))]
        exp = np.zeros((150, 11), np.float32)
        exp[:len(rows)] = rows
        assert np.allclose(got[r], exp, rtol=1e-5, atol=1e-4), (r, i, j)
    print("    %d AOJ jets checked" % len(gidx))
    assert len(gidx) > 0


# =============================================================================== runner
def main():
    tests = [(k, v) for k, v in sorted(globals().items(), key=lambda kv: kv[1].__code__.co_firstlineno
                                       if callable(kv[1]) and hasattr(kv[1], "__code__") else 0)
             if k.startswith("test_") and callable(v)]
    print("AOJ ttbar_semilep tests | outputs in %s" % OUT)
    res = {}
    for name, fn in tests:
        try:
            fn()
            res[name] = "PASS"
        except Skip as e:
            res[name] = "SKIP (%s)" % e
        except Exception:
            traceback.print_exc()
            res[name] = "FAIL"
        print("%-48s %s" % (name, res[name]))
        sys.stdout.flush()
    nf = sum(1 for v in res.values() if v == "FAIL")
    print("\n%d passed, %d skipped, %d failed" % (sum(1 for v in res.values() if v == "PASS"),
                                                 sum(1 for v in res.values() if v.startswith("SKIP")), nf))
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
