# AOJ selection pieces used by ttbar_semilep: the AOJ HLT trigger / MET-filter lists, the golden-JSON
# lumi mask and the AOJ jet definition.  A copy, so that ttbar_semilep runs without event_level/ (which
# holds the original, event_level/aoj_selection.py).  The lists must stay VERBATIM (same order) as
# ../H5_maker.py (NanoReader: `filters`, `triggers`); tests/test_ttbar.py parses H5_maker.py and fails if
# they drift, and also compares with event_level/aoj_selection.py when that is present.
#
# Pure numpy, Python >= 3.6.
import json

import numpy as np

# MET filters (== H5_maker.py `filters`; an absent Flag_* branch is skipped)
AOJ_FILTERS = [
    "Flag_goodVertices",
    "Flag_globalSuperTightHalo2016Filter",
    "Flag_HBHENoiseFilter",
    "Flag_HBHENoiseIsoFilter",
    "Flag_EcalDeadCellTriggerPrimitiveFilter",
    "Flag_BadPFMuonFilter",
    "Flag_eeBadScFilter",
    "Flag_CSCTightHaloFilter",
]

# AOJ JetHT paths (== H5_maker.py `triggers`; an absent HLT_* branch counts as not fired)
AOJ_TRIGGERS = [
    "HLT_AK8DiPFJet250_200_TrimMass30_BTagCSV_p20",
    "HLT_AK8DiPFJet250_200_TrimMass30",
    "HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p087",
    "HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p20",
    "HLT_AK8DiPFJet280_200_TrimMass30",
    "HLT_AK8DiPFJet300_200_TrimMass30_BTagCSV_p087",
    "HLT_AK8DiPFJet300_200_TrimMass30_BTagCSV_p20",
    "HLT_AK8DiPFJet300_200_TrimMass30",
    "HLT_AK8PFHT600_TrimR0p1PT0p03Mass50_BTagCSV_p20",
    "HLT_AK8PFHT650_TrimR0p1PT0p03Mass50",
    "HLT_AK8PFHT700_TrimR0p1PT0p03Mass50",
    "HLT_AK8PFHT750_TrimMass50",
    "HLT_AK8PFHT800_TrimMass50",
    "HLT_AK8PFJet140",
    "HLT_AK8PFJet200",
    "HLT_AK8PFJet260",
    "HLT_AK8PFJet320",
    "HLT_AK8PFJet360_TrimMass30",
    "HLT_AK8PFJet400_TrimMass30",
    "HLT_AK8PFJet400",
    "HLT_AK8PFJet40",
    "HLT_AK8PFJet450",
    "HLT_AK8PFJet500",
    "HLT_AK8PFJet60",
    "HLT_AK8PFJet80",
    "HLT_CaloJet500_NoJetID",
    "HLT_DiCentralPFJet170_CFMax0p1",
    "HLT_DiCentralPFJet170",
    "HLT_DiCentralPFJet220_CFMax0p3",
    "HLT_DiCentralPFJet330_CFMax0p5",
    "HLT_DiCentralPFJet430",
    "HLT_DiPFJetAve100_HFJEC",
    "HLT_DiPFJetAve140",
    "HLT_DiPFJetAve160_HFJEC",
    "HLT_DiPFJetAve200",
    "HLT_DiPFJetAve220_HFJEC",
    "HLT_DiPFJetAve260",
    "HLT_DiPFJetAve300_HFJEC",
    "HLT_DiPFJetAve320",
    "HLT_DiPFJetAve400",
    "HLT_DiPFJetAve40",
    "HLT_DiPFJetAve500",
    "HLT_DiPFJetAve60_HFJEC",
    "HLT_DiPFJetAve60",
    "HLT_DiPFJetAve80_HFJEC",
    "HLT_DiPFJetAve80",
    "HLT_HT2000",
    "HLT_HT2500",
    "HLT_L1_TripleJet_VBF",
    "HLT_PFHT125",
    "HLT_PFHT200",
    "HLT_PFHT250",
    "HLT_PFHT300",
    "HLT_PFHT350",
    "HLT_PFHT400_SixJet30_DoubleBTagCSV_p056",
    "HLT_PFHT400_SixJet30",
    "HLT_PFHT400",
    "HLT_PFHT450_SixJet40_BTagCSV_p056",
    "HLT_PFHT450_SixJet40",
    "HLT_PFHT475",
    "HLT_PFHT550_4JetPt50",
    "HLT_PFHT600",
    "HLT_PFHT650_4JetPt50",
    "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5",
    "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5",
    "HLT_PFHT650",
    "HLT_PFHT750_4JetPt50",
    "HLT_PFHT750_4JetPt70",
    "HLT_PFHT750_4JetPt80",
    "HLT_PFHT800_4JetPt50",
    "HLT_PFHT800",
    "HLT_PFHT850_4JetPt50",
    "HLT_PFHT900",
    "HLT_PFJet140",
    "HLT_PFJet200",
    "HLT_PFJet260",
    "HLT_PFJet320",
    "HLT_PFJet400",
    "HLT_PFJet40",
    "HLT_PFJet450",
    "HLT_PFJet500",
    "HLT_PFJet60",
    "HLT_PFJet80",
    "HLT_QuadPFJet_VBF",
    "HLT_SingleCentralPFJet170_CFMax0p1",
]

# AOJ per-AK8-jet selection (H5_maker.py: pt > 300 and jetId == 6 and |eta| < 2.5)
AOJ_JET_MIN_PT = 300.
AOJ_JET_MAX_ABS_ETA = 2.5
AOJ_JET_ID = 6          # == tight (bit 1) + tightLepVeto (bit 2); exact equality, as H5_maker


# ---------------------------------------------------------------------------
# Golden JSON (inclusive [l1, l2] lumi ranges per run, == NanoAODTools preSkim)
# ---------------------------------------------------------------------------
class LumiMask(object):
    """Vectorised golden-JSON lookup.  JSON: {"run": [[l1, l2], ...], ...}."""

    def __init__(self, json_path=None, ranges=None):
        if json_path is not None:
            with open(json_path) as f:
                ranges = json.load(f)
        starts, ends = [], []
        for run, lumis in ranges.items():
            r = int(run)
            for l1, l2 in lumis:
                starts.append((r << 32) | int(l1))
                ends.append((r << 32) | int(l2))
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        o = np.argsort(starts, kind="stable")
        starts, ends = starts[o], ends[o]
        # merge overlapping / touching ranges so a single searchsorted is exact
        ms, me = [], []
        for s, e in zip(starts.tolist(), ends.tolist()):
            if ms and s <= me[-1] + 1 and (s >> 32) == (me[-1] >> 32):
                me[-1] = max(me[-1], e)
            else:
                ms.append(s)
                me.append(e)
        self.starts = np.asarray(ms, dtype=np.int64)
        self.ends = np.asarray(me, dtype=np.int64)
        self.n_runs = len(ranges)

    def __call__(self, run, lumi):
        """Boolean mask: (run, lumi) inside a certified range (inclusive)."""
        key = (np.asarray(run).astype(np.int64) << 32) | np.asarray(lumi).astype(np.int64)
        if len(self.starts) == 0:
            return np.zeros(key.shape, dtype=bool)
        i = np.searchsorted(self.starts, key, side="right") - 1
        ok = i >= 0
        ic = np.where(ok, i, 0)
        return ok & (key <= self.ends[ic])


def trigger_matrix(arrays, names, n):
    """(n, len(names)) bool; a path whose branch is absent is False (it cannot have fired).
    arrays: dict branch -> 1D numpy (only present branches)."""
    m = np.zeros((n, len(names)), dtype=bool)
    for j, t in enumerate(names):
        if t in arrays:
            m[:, j] = np.asarray(arrays[t]).astype(bool)
    return m


def filter_pass(arrays, names, n):
    """AND of the filters that are PRESENT (H5_maker semantics: an absent Flag_* branch is
    skipped, i.e. treated as passing).  Returns (pass (n,) bool, list of missing names)."""
    ok = np.ones(n, dtype=bool)
    missing = []
    for f in names:
        if f in arrays:
            ok &= np.asarray(arrays[f]).astype(bool)
        else:
            missing.append(f)
    return ok, missing


def aoj_jet_mask(pt, eta, jet_id, min_pt=AOJ_JET_MIN_PT, max_abs_eta=AOJ_JET_MAX_ABS_ETA, jet_id_value=AOJ_JET_ID):
    """Per-AK8-jet AOJ selection, strict inequalities and exact jetId as H5_maker.py."""
    return (np.asarray(pt) > min_pt) & (np.asarray(jet_id) == jet_id_value) & (np.abs(np.asarray(eta)) < max_abs_eta)
