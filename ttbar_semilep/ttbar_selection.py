# Semileptonic-ttbar tag-and-probe selection: top-enriched AOJ AK8 jets from collision data.
#
# Default ("an" preset) = the ttbar tag-and-probe implementation of the CMS HIG-24-010 analysis
# (except the extra-AK8 veto: other AK8 jets above 300 GeV, the AOJ probe threshold; "an160" = 160 as there):
#   single-lepton trigger (HLT_Ele32_WPTight_Gsf or HLT_IsoMu27); >= 1 isolated e (mu) with pT > 55,
#   |eta| < 2.4 (tag); >= 1 AK8 jet with pT > 160, |eta| < 2.5 (probe); tag and probe back to back,
#   dphi > 2.0; no additional AK8 jet with pT > 160; no additional isolated lepton with pT > 30;
#   MET > 50; >= 1 medium DeepJet AK4 jet with pT > 40, |eta| < 2.4, dR(b, probe) > 1.5, dR(b, tag) > 0.5
# (+ tau32 < 0.5 on the probe, from the paper's description, as the last, optional stage), adapted to
# Run-2 UL NanoAOD (2016 Open Data) and to the AOJ jet definition: the probe must also be an AOJ jet
# (H5_maker.py: pt > 300, |eta| < 2.5, jetId == 6).  2016 substitutes: muons HLT_IsoMu24 || HLT_IsoTkMu24 (the
# OR the Muon POG measured 2016 UL trigger SFs for, w.r.t. tight ID + tight PF iso = this tag muon; there are none
# for IsoMu27); electrons HLT_Ele27_WPTight_Gsf || HLT_Ele32_WPTight_Gsf (Ele32_WPTight does not exist in Run2016G).
# The "paper" preset approximates the region as described in the HIG-24-010 paper (lepton pT > 50, veto
# leptons > 10, an AK8 jet with mSD > 50, no dphi cut, b-jet dR > 0.8 / 0.4, also the high-pT paths).
# The "skim" preset is a loose superset for offline cuts: isolated OR high-pT lepton paths, a tight-ID
# lepton with pT > 30 and relIso < 1 (electrons: the no-iso MVA WP80), MET > 20, any number of AK8 jets
# above 300 (probe = the leading AOJ jet with dR(lepton) > 0.8).  The tag lepton is the "an"-tight one
# whenever the event has one, and the outputs an_channel / trig_*_an / n_ak8 / met_pt / lep_* let the
# "an" selection be re-applied offline.
#
# Cut flow (cumulative, per channel mu / el):
#   all -> trigger -> filters_json -> one_lepton -> met -> one_probe -> dphi -> btag -> tau32
# The selection is evaluated for every event; nothing here decides what is stored.
#
# Pure numpy, Python >= 3.6.  A NanoAOD collection is a Coll = per-event counts + flat columns,
# so the code is identical for awkward 1.x / 2.x (only the maker touches uproot/awkward).
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aoj_common as aoj  # noqa: E402  (AOJ trigger/filter lists, LumiMask, AOJ jet definition; copy of event_level's)

# ---------------------------------------------------------------------------
# Era-dependent inputs.  Triggers per trigger set: "an" = the isolated single-lepton paths of the
# HIG-24-010 implementation (with the 2016 substitute above); "paper" = the lowest unprescaled isolated
# path(s) OR the non-isolated high-pT path(s).  2016/2017 electron paths are those of the SingleElectron
# PD (HLT_Photon175/200 live in SinglePhoton; 2018 has one EGamma PD).  DeepJet WPs: BTV UL.
# ---------------------------------------------------------------------------
ERAS = {
    "2016postVFP": {
        "triggers": {"an": {"mu": ["HLT_IsoMu24", "HLT_IsoTkMu24"], "el": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele32_WPTight_Gsf"]},
                     "paper": {"mu": ["HLT_IsoMu24", "HLT_IsoTkMu24", "HLT_Mu50", "HLT_TkMu50"],
                               "el": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele115_CaloIdVT_GsfTrkIdT"]}},
        "btag_wp": {"L": 0.0480, "M": 0.2489, "T": 0.6377},
    },
    "2016preVFP": {
        "triggers": {"an": {"mu": ["HLT_IsoMu24", "HLT_IsoTkMu24"], "el": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele32_WPTight_Gsf"]},
                     "paper": {"mu": ["HLT_IsoMu24", "HLT_IsoTkMu24", "HLT_Mu50", "HLT_TkMu50"],
                               "el": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele115_CaloIdVT_GsfTrkIdT"]}},
        "btag_wp": {"L": 0.0508, "M": 0.2598, "T": 0.6502},
    },
    "2017": {
        "triggers": {"an": {"mu": ["HLT_IsoMu27"], "el": ["HLT_Ele32_WPTight_Gsf"]},
                     "paper": {"mu": ["HLT_IsoMu27", "HLT_Mu50", "HLT_OldMu100", "HLT_TkMu100"],
                               "el": ["HLT_Ele35_WPTight_Gsf", "HLT_Ele115_CaloIdVT_GsfTrkIdT"]}},
        "btag_wp": {"L": 0.0532, "M": 0.3040, "T": 0.7476},
    },
    "2018": {
        "triggers": {"an": {"mu": ["HLT_IsoMu27"], "el": ["HLT_Ele32_WPTight_Gsf"]},
                     "paper": {"mu": ["HLT_IsoMu24", "HLT_Mu50", "HLT_OldMu100", "HLT_TkMu100"],
                               "el": ["HLT_Ele32_WPTight_Gsf", "HLT_Ele115_CaloIdVT_GsfTrkIdT", "HLT_Photon200"]}},
        "btag_wp": {"L": 0.0490, "M": 0.2783, "T": 0.7100},
    },
}
for _e in ERAS.values():         # "skim" = the union of the "an" and the "paper" paths, per channel
    _t = _e["triggers"]
    _t["skim"] = dict((ch, _t["an"][ch] + [p for p in _t["paper"][ch] if p not in _t["an"][ch]]) for ch in ("mu", "el"))
DEFAULT_ERA = "2016postVFP"      # Run2016G + Run2016H (the AOJ eras)
# Single-lepton paths stored one bit each in trig_bits (bit k = TRIG_BITS[k]; absent paths = 0), for
# offline trigger choices / trigger-efficiency studies.  2016 names; the other eras' paths are not bits.
TRIG_BITS = ["HLT_IsoMu24", "HLT_IsoTkMu24", "HLT_Mu50", "HLT_TkMu50", "HLT_Ele27_WPTight_Gsf",
             "HLT_Ele32_WPTight_Gsf", "HLT_Ele115_CaloIdVT_GsfTrkIdT", "HLT_Photon175"]
# AK4 jets stored per event (output "ak4", shape (n, N_AK4, len(AK4_COLUMNS))) for b-tag event weights; the b-tag
# SFs cover pT > 20, |eta| < 2.5 (UL16 DeepJet)
AK4_COLUMNS = ["pt", "eta", "phi", "btagDeepFlavB", "hadronFlavour", "dr_probe", "dr_lep"]
N_AK4, AK4_MIN_PT, AK4_MAX_ABS_ETA = 10, 20.0, 2.5

DEFAULT_CUTS = {                 # == PRESETS["an"]
    "trigger_set": "an",
    # tag lepton: >= 1 isolated lepton pT > 55, |eta| < 2.4; no additional isolated lepton with pT > 30
    "lep_min_pt": 55.0,
    "mu_max_abs_eta": 2.4,
    "mu_max_reliso": 0.15,        # Muon_tightId and pfRelIso04_all < 0.15
    "el_max_abs_eta": 2.4,        # |eta_SC|, ECAL gap 1.4442-1.566 excluded for the tag
    "el_id": "mvaFall17V2Iso_WP80",
    "veto_min_pt": 30.0,
    "veto_mu_max_reliso": 0.25,   # Muon_looseId and pfRelIso04_all < 0.25
    "veto_el_min_cutbased": 1,    # Electron_cutBased >= 1 (veto ID, includes isolation)
    "met_min": 50.0,              # PF MET (MET_pt)
    # probe: exactly one AK8 jet with pT > fj_count_min_pt and |eta| < fj_max_abs_eta, which must be an
    # AOJ jet (pT > fj_min_pt, jetId == fj_jet_id) with mSD > fj_min_msd (0 = no mSD cut).  The analysis note
    # vetoes other AK8 jets above its 160 GeV probe threshold; with the 300 GeV AOJ probe we veto above 300
    # (~3x the events at the same top purity in the R&D; preset "an160" restores 160)
    "fj_count_min_pt": 300.0,
    "fj_min_pt": aoj.AOJ_JET_MIN_PT,
    "fj_max_abs_eta": aoj.AOJ_JET_MAX_ABS_ETA,
    "fj_jet_id": aoj.AOJ_JET_ID,
    "fj_min_msd": 0.0,
    "min_dphi_lep_probe": 2.0,    # tag and probe back to back (0 = no cut)
    # b-tagged AK4 jet away from the probe and the tag
    "b_min_pt": 40.0,
    "b_max_abs_eta": 2.4,
    "b_wp": "M",
    "b_min_dr_fj": 1.5,
    "b_min_dr_lep": 0.5,
    "tau32_max": 0.5,
    # lepton logic: "an" = exactly one tight lepton and no other veto lepton; "skim" = >= 1 tight lepton,
    # the "an"-tight one preferred (no veto; an_channel keeps the "an" decision)
    "lep_mode": "an",
    "el_max_reliso": 0.0,         # Electron_pfRelIso03_all upper cut on the tight electron (0 = none)
    "max_n_ak8": 1,               # counted AK8 jets allowed (1 = "exactly one")
    "probe_min_dr_lep": 0.0,      # dR(probe, tag lepton) lower cut on the probe choice (0 = none)
}
PRESETS = {
    "an": dict(DEFAULT_CUTS),
    "an160": dict(DEFAULT_CUTS, fj_count_min_pt=160.0),     # the analysis note's AK8 veto threshold, verbatim
    # the region as worded in the HIG-24-010 paper ("well separated" -> dR > 0.8 / 0.4 is our reading;
    # "additional AK8 jets" -> AK8 jets above the probe threshold, i.e. pT > 300 here)
    "paper": dict(DEFAULT_CUTS, trigger_set="paper", lep_min_pt=50.0, el_max_abs_eta=2.5, veto_min_pt=10.0,
                  fj_count_min_pt=300.0, fj_min_msd=50.0, min_dphi_lep_probe=0.0, b_min_pt=30.0, b_min_dr_fj=0.8,
                  b_min_dr_lep=0.4),
    # loose skim (see the header); dphi / btag / tau32 stay as in "an" (they are applied offline anyway)
    "skim": dict(DEFAULT_CUTS, trigger_set="skim", lep_mode="skim", lep_min_pt=30.0, mu_max_reliso=1.0,
                 el_id="mvaFall17V2noIso_WP80", el_max_reliso=1.0, met_min=20.0, max_n_ak8=99, probe_min_dr_lep=0.8),
}
AN_EL_ID = DEFAULT_CUTS["el_id"]          # the "an" electron ID (lep_id_iso, an_channel)
LEP_MODES = ("an", "skim")

CUTFLOW = ["all", "trigger", "filters_json", "one_lepton", "met", "one_probe", "dphi", "btag", "tau32"]
CHANNELS = ["mu", "el"]          # channel code = index + 1 (0 = no channel)
TRIGGER_MODES = ["lepton", "jetht", "lepton_or_jetht", "none"]

# truth category of the probe (MC): partons of a hadronic W (and of its top) inside dR < 0.8
TRUTH_CATEGORIES = ["other", "bq", "W", "t"]      # code = index; t = b, q, q' all inside
TRUTH_DR = 0.8

FATJET_TAGGING = ["nConstituents", "tau1", "tau2", "tau3", "tau4",
                  "particleNet_H4qvsQCD", "particleNet_HbbvsQCD", "particleNet_HccvsQCD", "particleNet_QCD",
                  "particleNet_TvsQCD", "particleNet_WvsQCD", "particleNet_ZvsQCD", "particleNet_mass"]  # == H5_maker jet_tagging


def merged_cuts(overrides=None, preset="an"):
    if preset not in PRESETS:
        raise KeyError("unknown preset '%s' (known: %s)" % (preset, ", ".join(PRESETS)))
    c = dict(PRESETS[preset])
    for k, v in (overrides or {}).items():
        if k not in c:
            raise KeyError("unknown cut '%s' (known: %s)" % (k, ", ".join(sorted(c))))
        c[k] = type(c[k])(v) if not isinstance(c[k], str) else str(v)
    return c


def parse_overrides(items):
    """['met_min=40', 'b_wp=T'] -> dict (values converted by merged_cuts)."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError("cut override must be name=value, got '%s'" % it)
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------
def offsets_from_counts(counts):
    o = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=o[1:])
    return o


class Coll(object):
    """One NanoAOD collection: per-event counts + flat numpy columns (no prefix: 'pt', 'eta', ...)."""

    def __init__(self, counts, cols):
        self.counts = np.asarray(counts, dtype=np.int64)
        self.n = len(self.counts)
        self.offsets = offsets_from_counts(self.counts)
        self.evt = np.repeat(np.arange(self.n, dtype=np.int64), self.counts)
        self.cols = dict((k, np.asarray(v)) for k, v in cols.items())
        for k, v in self.cols.items():
            if len(v) != len(self.evt):
                raise ValueError("column %s has %d rows, counts sum to %d" % (k, len(v), len(self.evt)))

    def __getitem__(self, k):
        return self.cols[k]

    def __contains__(self, k):
        return k in self.cols

    def count(self, mask):
        return np.bincount(self.evt[mask], minlength=self.n).astype(np.int64)

    def first(self, mask):
        """Global row of the first True object of each event (NanoAOD order, i.e. leading pt), -1 if none."""
        out = np.full(self.n, -1, dtype=np.int64)
        idx = np.nonzero(mask)[0]
        if len(idx):
            ev, pos = np.unique(self.evt[idx], return_index=True)
            out[ev] = idx[pos]
        return out

    def take(self, col, gidx, fill=np.nan):
        """Per-event value of column `col` at global rows gidx (fill where gidx < 0)."""
        v = self.cols[col]
        out = np.full(len(gidx), fill, dtype=np.float64)
        ok = gidx >= 0
        out[ok] = v[gidx[ok]]
        return out

    def take_bool(self, mask, gidx):
        """Per-event value of a per-object boolean array at global rows gidx (False where gidx < 0)."""
        out = np.zeros(len(gidx), dtype=bool)
        ok = gidx >= 0
        out[ok] = np.asarray(mask, bool)[gidx[ok]]
        return out

    def local(self, gidx):
        out = np.full(len(gidx), -1, dtype=np.int64)
        ok = gidx >= 0
        out[ok] = gidx[ok] - self.offsets[self.evt[gidx[ok]]]
        return out


def wrap_phi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    return np.hypot(eta1 - eta2, wrap_phi(phi1 - phi2))


def any_fired(ev, names, n):
    """OR of the paths whose branch is present (an absent path cannot have fired)."""
    if not names:
        return np.zeros(n, dtype=bool)
    return aoj.trigger_matrix(ev, names, n).any(axis=1)


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------
def muon_masks(mu, c):
    base = np.abs(mu["eta"]) < c["mu_max_abs_eta"]
    iso = mu["pfRelIso04_all"]
    tight = base & (mu["pt"] > c["lep_min_pt"]) & mu["tightId"].astype(bool) & (iso < c["mu_max_reliso"])
    veto = base & (mu["pt"] > c["veto_min_pt"]) & mu["looseId"].astype(bool) & (iso < c["veto_mu_max_reliso"])
    return tight, veto | tight


def electron_iso(el):
    """Electron_pfRelIso03_all, NaN where the input does not have it."""
    return el["pfRelIso03_all"] if "pfRelIso03_all" in el else np.full(len(el.evt), np.nan)


def electron_masks(el, c):
    sc_eta = np.abs(el["eta"] + el["deltaEtaSC"])
    acc = sc_eta < c["el_max_abs_eta"]
    gap = (sc_eta > 1.4442) & (sc_eta < 1.5660)
    tight = acc & ~gap & (el["pt"] > c["lep_min_pt"]) & el[c["el_id"]].astype(bool)
    if c["el_max_reliso"] > 0:
        with np.errstate(invalid="ignore"):
            tight = tight & (electron_iso(el) < c["el_max_reliso"])
    veto = acc & (el["pt"] > c["veto_min_pt"]) & (el["cutBased"] >= c["veto_el_min_cutbased"])
    return tight, veto | tight


def probe_candidates(fj, c):
    """(AOJ jets, AK8 jets counted for 'exactly one AK8 jet', jets that qualify as the probe)."""
    aoj_jet = aoj.aoj_jet_mask(fj["pt"], fj["eta"], fj["jetId"], min_pt=c["fj_min_pt"],
                               max_abs_eta=c["fj_max_abs_eta"], jet_id_value=c["fj_jet_id"])
    counted = (fj["pt"] > c["fj_count_min_pt"]) & (np.abs(fj["eta"]) < c["fj_max_abs_eta"])
    probe = aoj_jet & counted
    if c["fj_min_msd"] > 0:
        probe = probe & (fj["msoftdrop"] > c["fj_min_msd"])
    return aoj_jet, counted, probe


# ---------------------------------------------------------------------------
# Event selection
# ---------------------------------------------------------------------------
def select(ev, era=DEFAULT_ERA, cuts=None, trigger="lepton", lumimask=None):
    """Evaluate the selection on every event.

    ev: dict with 1D numpy arrays (run, luminosityBlock, MET_pt, MET_phi, HLT_*, Flag_*; absent HLT
        paths count as not fired, absent Flag_* are skipped as in H5_maker) and Coll objects
        'Muon', 'Electron', 'Jet', 'FatJet'.
    trigger: 'lepton' (mu channel needs a muon path, el channel an electron path), 'jetht' (OR of the
        85 AOJ JetHT paths), 'lepton_or_jetht', 'none'.
    lumimask: aoj.LumiMask (data) or None.
    Returns a dict of per-event numpy arrays; 'stage_mu' / 'stage_el' = number of consecutive
    CUTFLOW cuts passed after 'all' (so stage == len(CUTFLOW) - 1 means selected incl. tau32).
    """
    if era not in ERAS:
        raise KeyError("unknown era '%s' (known: %s)" % (era, ", ".join(ERAS)))
    if trigger not in TRIGGER_MODES:
        raise KeyError("unknown trigger mode '%s' (known: %s)" % (trigger, ", ".join(TRIGGER_MODES)))
    E = ERAS[era]
    c = merged_cuts(cuts)
    if c["lep_mode"] not in LEP_MODES:
        raise KeyError("unknown lep_mode '%s' (known: %s)" % (c["lep_mode"], ", ".join(LEP_MODES)))
    mu, el, jet, fj = ev["Muon"], ev["Electron"], ev["Jet"], ev["FatJet"]
    n = mu.n
    out = {}

    # trigger
    T = E["triggers"][c["trigger_set"]]
    trig_mu = any_fired(ev, T["mu"], n)
    trig_el = any_fired(ev, T["el"], n)
    trig_jet = any_fired(ev, aoj.AOJ_TRIGGERS, n)
    out.update(trig_mu=trig_mu, trig_el=trig_el, trig_jetht=trig_jet)
    T_an = E["triggers"]["an"]                  # the "an" paths, for re-applying "an" offline
    out["trig_mu_an"] = trig_mu if T_an is T else any_fired(ev, T_an["mu"], n)
    out["trig_el_an"] = trig_el if T_an is T else any_fired(ev, T_an["el"], n)
    if trigger == "lepton":
        t_ch = {"mu": trig_mu, "el": trig_el}
    elif trigger == "jetht":
        t_ch = {"mu": trig_jet, "el": trig_jet}
    elif trigger == "lepton_or_jetht":
        t_ch = {"mu": trig_mu | trig_jet, "el": trig_el | trig_jet}
    else:
        t_ch = {"mu": np.ones(n, bool), "el": np.ones(n, bool)}

    # MET filters (AOJ list) + golden JSON
    filt, missing = aoj.filter_pass(ev, aoj.AOJ_FILTERS, n)
    out["filters_missing"] = missing
    if lumimask is not None:
        filt = filt & lumimask(ev["run"], ev["luminosityBlock"])

    # leptons
    mu_t, mu_v = muon_masks(mu, c)
    el_t, el_v = electron_masks(el, c)
    n_mu_t, n_mu_v, n_el_t, n_el_v = mu.count(mu_t), mu.count(mu_v), el.count(el_t), el.count(el_v)
    ca = c if c["lep_mode"] == "an" else PRESETS["an"]      # "an" lepton decision (an_channel)
    if ca is c:
        mu_ta, el_ta, n_mu_va, n_el_va = mu_t, el_t, n_mu_v, n_el_v
    else:
        mu_ta, mu_va = muon_masks(mu, ca)
        el_ta, el_va = electron_masks(el, ca)
        n_mu_va, n_el_va = mu.count(mu_va), el.count(el_va)
    an_mu = (mu.count(mu_ta) == 1) & (n_mu_va == 1) & (n_el_va == 0)
    an_el = (el.count(el_ta) == 1) & (n_el_va == 1) & (n_mu_va == 0)
    out["an_channel"] = np.where(an_mu, 1, np.where(an_el, 2, 0)).astype(np.int8)
    if c["lep_mode"] == "an":
        is_mu = (n_mu_t == 1) & (n_mu_v == 1) & (n_el_v == 0)
        is_el = (n_el_t == 1) & (n_el_v == 1) & (n_mu_v == 0)
        imu, iel = mu.first(mu_t), el.first(el_t)
    else:
        # >= 1 tight lepton; per flavour the "an"-tight one if any (so an "an" event keeps its lepton),
        # then the flavour with an "an"-tight lepton, else the higher-pt one
        fa_mu, fa_el = mu.first(mu_ta), el.first(el_ta)
        imu = np.where(fa_mu >= 0, fa_mu, mu.first(mu_t))
        iel = np.where(fa_el >= 0, fa_el, el.first(el_t))
        has_mu, has_el = imu >= 0, iel >= 0
        a_mu, a_el = fa_mu >= 0, fa_el >= 0
        pt_mu, pt_el = mu.take("pt", imu), el.take("pt", iel)
        one_an = a_mu ^ a_el
        with np.errstate(invalid="ignore"):
            is_mu = has_mu & np.where(one_an, a_mu, ~has_el | (pt_mu >= pt_el))
            is_el = has_el & np.where(one_an, a_el, ~has_mu | (pt_el > pt_mu))
    out["n_lep_tight"] = n_mu_t + n_el_t
    lep = {}
    for k in ("pt", "eta", "phi", "charge"):
        lep[k] = np.where(is_mu, mu.take(k, imu), np.where(is_el, el.take(k, iel), np.nan))
    out["channel"] = np.where(is_mu, 1, np.where(is_el, 2, 0)).astype(np.int8)
    out["lep_idx"] = np.where(is_mu, mu.local(imu), np.where(is_el, el.local(iel), -1))
    out["lep_pdgId"] = np.where(is_mu, -13 * lep["charge"], np.where(is_el, -11 * lep["charge"], 0))
    for k, v in lep.items():
        out["lep_" + k] = v
    out["n_veto_mu"], out["n_veto_el"] = n_mu_v, n_el_v
    # isolation and the "an" ID+iso decision of the chosen lepton (mu: tightId and relIso04 < "an" cut;
    # el: the "an" MVA WP), for offline iso cuts / sidebands
    out["lep_reliso"] = np.where(is_mu, mu.take("pfRelIso04_all", imu),
                                 np.where(is_el, el.take("pfRelIso03_all", iel) if "pfRelIso03_all" in el
                                          else np.full(n, np.nan), np.nan))
    with np.errstate(invalid="ignore"):
        mu_isoid = mu["tightId"].astype(bool) & (mu["pfRelIso04_all"] < ca["mu_max_reliso"])
    el_isoid = el[AN_EL_ID].astype(bool) if AN_EL_ID in el else np.zeros(len(el.evt), bool)
    out["lep_id_iso"] = np.where(is_mu, mu.take_bool(mu_isoid, imu), np.where(is_el, el.take_bool(el_isoid, iel), False))

    # MET
    met, met_phi = np.asarray(ev["MET_pt"], np.float64), np.asarray(ev["MET_phi"], np.float64)
    out["met_pt"], out["met_phi"] = met, met_phi
    wx = lep["pt"] * np.cos(lep["phi"]) + met * np.cos(met_phi)
    wy = lep["pt"] * np.sin(lep["phi"]) + met * np.sin(met_phi)
    out["lepW_pt"] = np.hypot(wx, wy)

    # probe: the leading qualifying jet (AOJ jet, mSD; skim: away from the lepton), with at most max_n_ak8
    # counted AK8 jets ("an": the only counted AK8 jet)
    aoj_jet, counted, cand = probe_candidates(fj, c)
    out["n_aoj_jets"] = fj.count(aoj_jet)
    out["n_ak8"] = fj.count(counted)
    if c["probe_min_dr_lep"] > 0:
        fe = fj.evt
        with np.errstate(invalid="ignore"):
            cand = cand & (delta_r(fj["eta"], fj["phi"], lep["eta"][fe], lep["phi"][fe]) > c["probe_min_dr_lep"])
    ip = fj.first(cand)
    probe_ok = (out["n_ak8"] >= 1) & (out["n_ak8"] <= c["max_n_ak8"]) & (ip >= 0)
    ip = np.where(probe_ok, ip, fj.first(counted))         # otherwise: the leading counted jet, for bookkeeping
    other = np.abs(fj["eta"]) < c["fj_max_abs_eta"]        # leading other AK8 jet (any pt): offline veto scans
    other[ip[ip >= 0]] = False
    out["ak8_other_pt"] = fj.take("pt", fj.first(other))
    out["probe_idx"] = fj.local(ip)
    p_eta, p_phi = fj.take("eta", ip), fj.take("phi", ip)
    out["probe_pt"], out["probe_eta"], out["probe_phi"] = fj.take("pt", ip), p_eta, p_phi
    out["probe_msd"] = fj.take("msoftdrop", ip)
    tau1, tau2, tau3 = fj.take("tau1", ip), fj.take("tau2", ip), fj.take("tau3", ip)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["tau32"] = np.where(tau2 > 0, tau3 / tau2, np.nan)
        out["tau21"] = np.where(tau1 > 0, tau2 / tau1, np.nan)
    out["dphi_lep_probe"] = np.abs(wrap_phi(lep["phi"] - p_phi))
    out["dr_lep_probe"] = delta_r(lep["eta"], lep["phi"], p_eta, p_phi)

    # b-tagged AK4 jets separated from the probe and the lepton
    wp = E["btag_wp"][c["b_wp"]]
    good = ((jet["pt"] > c["b_min_pt"]) & (np.abs(jet["eta"]) < c["b_max_abs_eta"])
            & ((jet["jetId"].astype(np.int64) & 2) != 0) & (jet["btagDeepFlavB"] > wp))
    je = jet.evt
    dr_fj = delta_r(jet["eta"], jet["phi"], p_eta[je], p_phi[je])
    dr_l = delta_r(jet["eta"], jet["phi"], lep["eta"][je], lep["phi"][je])
    with np.errstate(invalid="ignore"):
        sep = good & (dr_fj > c["b_min_dr_fj"]) & (dr_l > c["b_min_dr_lep"])
    out["n_b"] = jet.count(sep)
    ib = jet.first(sep)
    out["b_idx"] = jet.local(ib)
    out["b_pt"], out["b_eta"], out["b_phi"] = jet.take("pt", ib), jet.take("eta", ib), jet.take("phi", ib)
    out["b_score"] = jet.take("btagDeepFlavB", ib)
    out["b_dr_probe"] = delta_r(out["b_eta"], out["b_phi"], p_eta, p_phi)
    out["b_dr_lep"] = delta_r(out["b_eta"], out["b_phi"], lep["eta"], lep["phi"])

    extra_outputs(ev, out, c, wp, lep, is_mu, is_el, imu, iel, ip)

    # cut flow
    with np.errstate(invalid="ignore"):
        met_ok = met > c["met_min"]
        tau_ok = out["tau32"] < c["tau32_max"]
        dphi_ok = out["dphi_lep_probe"] > c["min_dphi_lep_probe"] if c["min_dphi_lep_probe"] > 0 else probe_ok
    b_ok = out["n_b"] >= 1
    out["pass_tau32"] = tau_ok
    for ch, is_ch in (("mu", is_mu), ("el", is_el)):
        steps = np.stack([t_ch[ch], filt, is_ch, met_ok, probe_ok, dphi_ok, b_ok, tau_ok])
        out["stage_" + ch] = np.cumprod(steps, axis=0).sum(axis=0).astype(np.int8)
    last = len(CUTFLOW) - 1
    out["selected"] = (out["stage_mu"] == last) | (out["stage_el"] == last)
    out["stage"] = np.maximum(out["stage_mu"], out["stage_el"])       # channel-exclusive from one_lepton on
    return out


def _col(coll, name):
    """Flat column as float64, NaN where the input does not have it (OPTIONAL_COLUMNS)."""
    return coll[name].astype(np.float64) if name in coll else np.full(len(coll.evt), np.nan)


def _at(v, g):
    """v[g] per event, NaN where g < 0 (g = global rows; v may be empty)."""
    out = np.full(len(g), np.nan)
    ok = g >= 0
    if ok.any():
        out[ok] = v[g[ok]]
    return out


def _scalar(ev, name, n):
    return np.asarray(ev[name], np.float64) if name in ev else np.full(n, np.nan)


def extra_outputs(ev, out, c, wp, lep, is_mu, is_el, imu, iel, ip):
    """v3 bookkeeping outputs for offline cuts and corrections (nothing here is cut on).  Inputs the
    file does not have give NaN (Pileup_* and Jet_hadronFlavour in data, SubJet if absent, ...).
    imu / iel / ip = global rows of the chosen muon / electron / probe (-1 = none)."""
    mu, el, jet, fj = ev["Muon"], ev["Electron"], ev["Jet"], ev["FatJet"]
    n = mu.n
    bits = np.zeros(n, dtype=np.int64)
    for k, path in enumerate(TRIG_BITS):
        if path in ev:
            bits |= np.asarray(ev[path]).astype(np.int64) << k
    out["trig_bits"] = bits
    out["npv_good"] = _scalar(ev, "PV_npvsGood", n)
    out["pu_ntrueint"] = _scalar(ev, "Pileup_nTrueInt", n)
    out["puppimet_pt"], out["puppimet_phi"] = _scalar(ev, "PuppiMET_pt", n), _scalar(ev, "PuppiMET_phi", n)

    # tag lepton: supercluster eta (el; mu: eta), impact parameters, mini-isolation
    def pick(vm, ve):
        return np.where(is_mu, _at(vm, imu), np.where(is_el, _at(ve, iel), np.nan))
    out["lep_sc_eta"] = pick(_col(mu, "eta"), _col(el, "eta") + _col(el, "deltaEtaSC"))
    for k, col in (("lep_dxy", "dxy"), ("lep_dz", "dz"), ("lep_miniiso", "miniPFRelIso_all")):
        out[k] = pick(_col(mu, col), _col(el, col))

    # AK4 jets with tight ID, pT > 30, |eta| < 2.4, dR(tag lepton) > 0.4 (none removed without a lepton):
    # multiplicity, HT, medium b tags; hadron flavour of the leading separated b jet (MC)
    je = jet.evt
    dr_l = delta_r(jet["eta"], jet["phi"], lep["eta"][je], lep["phi"][je])
    with np.errstate(invalid="ignore"):
        j30 = ((jet["pt"] > 30) & (np.abs(jet["eta"]) < 2.4) & ((jet["jetId"].astype(np.int64) & 2) != 0)
               & ~(dr_l < 0.4))
    out["n_jet"] = jet.count(j30)
    out["ht"] = np.bincount(je[j30], weights=jet["pt"][j30], minlength=n)
    out["n_b_all"] = jet.count(j30 & (jet["btagDeepFlavB"] > wp))
    bl = np.asarray(out["b_idx"], np.int64)
    out["b_hflav"] = _at(_col(jet, "hadronFlavour"), np.where(bl >= 0, jet.offsets[:-1] + bl, -1))

    # 2D-isolation inputs: the AK4 jet nearest to the lepton among pT > 15, |eta| < 2.4 (dR, and the
    # lepton momentum transverse to that jet's axis), after taking the lepton out of the jet that contains it
    # (Muon_jetIdx / Electron_jetIdx): its momentum is subtracted from the raw jet, which is then scaled back by
    # the jet's JEC (Jet_rawFactor). AK4 jets are clustered with the leptons in them, so without this the nearest
    # jet is the lepton's own jet for ~98 % of isolated leptons (dR ~ 0, ptrel ~ 0).
    pt_j = np.asarray(jet["pt"], np.float64)
    jx, jy, jz = pt_j * np.cos(jet["phi"]), pt_j * np.sin(jet["phi"]), pt_j * np.sinh(jet["eta"])
    lx, ly, lz = lep["pt"] * np.cos(lep["phi"]), lep["pt"] * np.sin(lep["phi"]), lep["pt"] * np.sinh(lep["eta"])
    own_loc = pick(_col(mu, "jetIdx"), _col(el, "jetIdx"))           # the lepton's jet (local index), NaN / -1: none
    with np.errstate(invalid="ignore"):
        has_own = np.isfinite(own_loc) & (own_loc >= 0) & (own_loc < jet.counts)
    e_own = np.flatnonzero(has_own)
    g_own = jet.offsets[e_own] + own_loc[e_own].astype(np.int64)
    if len(g_own):
        r = 1.0 - _col(jet, "rawFactor")[g_own]                      # raw / corrected pT (no Jet_rawFactor: 1)
        r = np.where(np.isfinite(r) & (r > 0), r, 1.0)
        jx[g_own] = (jx[g_own] * r - lx[e_own]) / r
        jy[g_own] = (jy[g_own] * r - ly[e_own]) / r
        jz[g_own] = (jz[g_own] * r - lz[e_own]) / r
    cpt = np.hypot(jx, jy)
    with np.errstate(invalid="ignore", divide="ignore"):
        ceta = np.where(cpt > 0, np.arcsinh(jz / np.where(cpt > 0, cpt, 1.0)), 0.0)
        dr_c = delta_r(ceta, np.arctan2(jy, jx), lep["eta"][je], lep["phi"][je])
        key = np.where((cpt > 15) & (np.abs(ceta) < 2.4) & np.isfinite(dr_c), dr_c, np.inf)
    nearest = np.full(n, -1, dtype=np.int64)
    if len(key):
        o = np.lexsort((key, je))
        evs, pos = np.unique(je[o], return_index=True)
        cand = o[pos]
        good = np.isfinite(key[cand])
        nearest[evs[good]] = cand[good]
    out["lep_jet_dr"] = _at(key, nearest)
    nx, ny, nz = _at(jx, nearest), _at(jy, nearest), _at(jz, nearest)
    cx, cy, cz = ly * nz - lz * ny, lz * nx - lx * nz, lx * ny - ly * nx
    with np.errstate(invalid="ignore", divide="ignore"):
        out["lep_jet_ptrel"] = np.sqrt(cx * cx + cy * cy + cz * cz) / np.sqrt(nx * nx + ny * ny + nz * nz)

    # AK4 jets for b-tag event weights (the fixed-WP method needs every jet the b-tag requirement can use, tagged
    # or not): tight ID, pT > AK4_MIN_PT, |eta| < AK4_MAX_ABS_ETA, the first N_AK4 in NanoAOD (pT) order, with
    # their dR to the probe and to the lepton so that any b-tag selection can be redone offline; NaN padded
    p_eta, p_phi = fj.take("eta", ip), fj.take("phi", ip)
    with np.errstate(invalid="ignore"):
        a4 = ((jet["pt"] > AK4_MIN_PT) & (np.abs(jet["eta"]) < AK4_MAX_ABS_ETA)
              & ((jet["jetId"].astype(np.int64) & 2) != 0))
    ak4 = np.full((n, N_AK4, len(AK4_COLUMNS)), np.nan, dtype=np.float32)
    g = np.flatnonzero(a4)
    if len(g):
        e = je[g]
        start = np.r_[0, np.flatnonzero(np.diff(e)) + 1]
        rank = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
        k = rank < N_AK4
        g, e, rank = g[k], e[k], rank[k]
        cols = {"pt": jet["pt"][g], "eta": jet["eta"][g], "phi": jet["phi"][g],
                "btagDeepFlavB": jet["btagDeepFlavB"][g], "hadronFlavour": _col(jet, "hadronFlavour")[g],
                "dr_probe": delta_r(jet["eta"][g], jet["phi"][g], p_eta[e], p_phi[e]), "dr_lep": dr_l[g]}
        ak4[e, rank] = np.stack([np.asarray(cols[c], np.float64) for c in AK4_COLUMNS], axis=1)
    out["ak4"] = ak4

    # probe: ungroomed mass, N3, DeepAK8 top scores, the larger subjet DeepCSV score
    for k, col in (("probe_mass", "mass"), ("probe_n3b1", "n3b1"), ("probe_deepTag_TvsQCD", "deepTag_TvsQCD"),
                   ("probe_deepTagMD_TvsQCD", "deepTagMD_TvsQCD")):
        out[k] = _at(_col(fj, col), ip)
    sjb = np.full(n, np.nan)
    sj = ev.get("SubJet")
    if sj is not None and "btagDeepB" in sj and "subJetIdx1" in fj:
        for col in ("subJetIdx1", "subJetIdx2"):
            loc = np.nan_to_num(_at(_col(fj, col), ip), nan=-1).astype(np.int64)
            ok = (loc >= 0) & (loc < sj.counts)
            with np.errstate(invalid="ignore"):
                sjb = np.fmax(sjb, _at(_col(sj, "btagDeepB"), np.where(ok, sj.offsets[:-1] + loc, -1)))
    out["probe_sj_btag"] = sjb


def cutflow(sel, weights=None):
    """(len(CUTFLOW), 2) counts (or weight sums) per channel (mu, el); row k = passed CUTFLOW[0..k]."""
    n = len(sel["stage_mu"])
    w = np.ones(n) if weights is None else np.asarray(weights, np.float64)
    cf = np.zeros((len(CUTFLOW), len(CHANNELS)), dtype=np.float64)
    for j, ch in enumerate(CHANNELS):
        st = sel["stage_" + ch]
        for k in range(len(CUTFLOW)):
            cf[k, j] = w[st >= k].sum()
    return cf


# ---------------------------------------------------------------------------
# MC truth of the probe
# ---------------------------------------------------------------------------
def gen_truth(gen, probe_eta, probe_phi, dr=TRUTH_DR):
    """Classify each event's probe from GenPart (pdgId, genPartIdxMother, pt, eta, phi).

    Hadronic W = a W whose decay products (GenParticles with a W mother) are quarks. Its top =
    the first non-W ancestor if it is a top; that top's b = the b quark with the top as mother.
    Per W: t = q, q' and b inside dR; W = q, q' inside (b outside or no top); bq = b and exactly
    one of q, q'; other otherwise.  The event takes the best category over its hadronic Ws
    (t > W > bq > other).  Also returns the nearest hadronic top (pt, dR), the numbers of hadronic
    and leptonic W decays and the flavour (11/13/15) of the first leptonic W's charged lepton.
    """
    n = gen.n
    pdg = gen["pdgId"].astype(np.int64)
    absp = np.abs(pdg)
    mom = gen["genPartIdxMother"].astype(np.int64)
    gmom = np.where(mom >= 0, gen.offsets[gen.evt] + mom, -1)
    gm = np.maximum(gmom, 0)
    has_m = gmom >= 0
    ev_eta, ev_phi = np.asarray(probe_eta, np.float64), np.asarray(probe_phi, np.float64)
    with np.errstate(invalid="ignore"):
        inside = delta_r(gen["eta"], gen["phi"], ev_eta[gen.evt], ev_phi[gen.evt]) < dr

    w_child = has_m & (absp[gm] == 24) & (absp != 24)
    quark = w_child & (absp <= 5)
    lepton = w_child & ((absp == 11) | (absp == 13) | (absp == 15))
    out = {}

    # ancestor top of each W (walk up through W copies)
    k = gm.copy()
    for _ in range(64):
        up = gmom[k]
        step = (absp[k] == 24) & (up >= 0)
        if not step.any():
            break
        k = np.where(step, up, k)
    top_of = np.where(absp[k] == 6, k, -1)            # per particle: top above its W mother chain

    # b from each top
    b_quark = has_m & (absp == 5) & (absp[gm] == 6)
    b_in_top = np.zeros(len(pdg), dtype=bool)
    has_b_top = np.zeros(len(pdg), dtype=bool)
    b_in_top[gmom[b_quark]] |= inside[b_quark]
    has_b_top[gmom[b_quark]] = True

    # per hadronic W: number of quark daughters inside, number of quark daughters
    qi = np.nonzero(quark)[0]
    wid = gmom[qi]
    cat = np.zeros(n, dtype=np.int64)
    n_had = np.zeros(n, dtype=np.int64)
    top_pt = np.full(n, np.nan)
    top_dr = np.full(n, np.nan)
    if len(qi):
        uw, inv = np.unique(wid, return_inverse=True)
        nq = np.bincount(inv, minlength=len(uw))
        nin = np.bincount(inv, weights=inside[qi].astype(np.float64), minlength=len(uw)).astype(np.int64)
        hadronic = nq >= 2
        wtop = np.full(len(uw), -1, dtype=np.int64)
        wtop[inv] = top_of[qi]
        b_in = np.where(wtop >= 0, b_in_top[np.maximum(wtop, 0)] & has_b_top[np.maximum(wtop, 0)], False)
        wcat = np.zeros(len(uw), dtype=np.int64)
        wcat[(nin == 1) & b_in] = 1
        wcat[nin >= 2] = 2
        wcat[(nin >= 2) & b_in] = 3
        wcat[~hadronic] = 0
        wev = gen.evt[uw]
        np.maximum.at(cat, wev, wcat)
        np.add.at(n_had, wev, hadronic.astype(np.int64))
        # nearest hadronic top
        ht = hadronic & (wtop >= 0)
        if ht.any():
            t_idx = wtop[ht]
            t_ev = gen.evt[t_idx]
            with np.errstate(invalid="ignore"):
                t_dr = delta_r(gen["eta"][t_idx], gen["phi"][t_idx], ev_eta[t_ev], ev_phi[t_ev])
            key = np.where(np.isfinite(t_dr), t_dr, 1e9)
            o = np.lexsort((key, t_ev))
            t_ev_s = t_ev[o]
            firsts = np.unique(t_ev_s, return_index=True)[1]
            sel_t = o[firsts]
            top_pt[t_ev[sel_t]] = gen["pt"][t_idx[sel_t]]
            top_dr[t_ev[sel_t]] = t_dr[sel_t]
    out["category"] = cat
    out["n_had_W"] = n_had
    out["had_top_pt"] = top_pt
    out["had_top_dr"] = top_dr

    li = np.nonzero(lepton)[0]
    lw = np.unique(gmom[li]) if len(li) else np.zeros(0, np.int64)
    out["n_lep_W"] = np.bincount(gen.evt[lw], minlength=n).astype(np.int64) if len(lw) else np.zeros(n, np.int64)
    flav = np.zeros(n, dtype=np.int64)
    if len(li):
        ev_l, pos = np.unique(gen.evt[li], return_index=True)
        flav[ev_l] = absp[li[pos]]
    out["lep_flav"] = flav
    # last-copy top / antitop pT (top-pT reweighting); NaN without statusFlags or tops
    for k, sign in (("gen_top_pt", 6), ("gen_antitop_pt", -6)):
        v = np.full(n, np.nan)
        if "statusFlags" in gen:
            last = (pdg == sign) & ((gen["statusFlags"].astype(np.int64) >> 13) & 1).astype(bool)
            i = gen.first(last)
            v = gen.take("pt", i)
        out[k] = v
    return out


# read when present (NaN otherwise): only the skim cuts on Electron_pfRelIso03_all; the rest are v3 bookkeeping
OPTIONAL_COLUMNS = {"Electron": ["pfRelIso03_all", "dxy", "dz", "miniPFRelIso_all", "jetIdx"],
                    "Muon": ["dxy", "dz", "miniPFRelIso_all", "jetIdx"],
                    "Jet": ["hadronFlavour", "rawFactor"],
                    "FatJet": ["mass", "n3b1", "deepTag_TvsQCD", "deepTagMD_TvsQCD", "subJetIdx1", "subJetIdx2"],
                    "SubJet": ["btagDeepB"],
                    "GenPart": ["statusFlags"]}
OPTIONAL_COLLECTIONS = ["GenPart", "SubJet"]           # absent collection: warn and go on
OPTIONAL_SCALARS = ["PV_npvsGood", "Pileup_nTrueInt", "PuppiMET_pt", "PuppiMET_phi"] + TRIG_BITS


def branches(era=DEFAULT_ERA, cuts=None, is_mc=False):
    """Branches the selection reads: (scalars, {collection: [columns]})."""
    c = merged_cuts(cuts)
    T = ERAS[era]["triggers"][c["trigger_set"]]
    T_an = ERAS[era]["triggers"]["an"]
    paths = T["mu"] + T["el"] + [p for p in T_an["mu"] + T_an["el"] if p not in T["mu"] + T["el"]]
    scalars = (["run", "luminosityBlock", "event", "MET_pt", "MET_phi"] + paths
               + aoj.AOJ_TRIGGERS + aoj.AOJ_FILTERS + (["genWeight"] if is_mc else []))
    scalars += [b for b in OPTIONAL_SCALARS if b not in scalars]
    el_ids = [c["el_id"]] + ([AN_EL_ID] if AN_EL_ID != c["el_id"] else [])
    colls = {
        "Muon": ["pt", "eta", "phi", "charge", "tightId", "looseId", "pfRelIso04_all"],
        "Electron": ["pt", "eta", "phi", "charge", "deltaEtaSC", "cutBased"] + el_ids,
        "Jet": ["pt", "eta", "phi", "jetId", "btagDeepFlavB"],
        "FatJet": ["pt", "eta", "phi", "msoftdrop", "jetId"] + FATJET_TAGGING,
        "SubJet": [],
    }
    if is_mc:
        colls["GenPart"] = ["pt", "eta", "phi", "pdgId", "genPartIdxMother"]
    for name, cols in OPTIONAL_COLUMNS.items():
        if name in colls:
            colls[name] += [x for x in cols if x not in colls[name]]
    return scalars, colls
