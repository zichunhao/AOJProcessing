# cmsRun wrapper: the AOJ PFNano Stage-1 config plus a loose MINIAOD pre-filter for the
# semileptonic-ttbar tag-and-probe, so the (expensive) NanoAOD/PFNano producers run only on candidate events.
#
#   AOJ_TTSKIM_BASE=pfnano_data_2016UL_OpenData.py cmsRun ttbar_semilep/pfnano_ttskim.py inputFiles_load=... [nThreads=N]
#   (run from AOJProcessing/; the base config's own options are parsed as usual)
#
# Kept: >= 1 slimmedMuons/slimmedElectrons with pT > 20, |eta| < 2.6 AND >= 1 slimmedJetsAK8 with pT > 200,
# |eta| < 2.6.  This is looser than every ttbar_selection preset (lepton pT > 30, probe = AOJ jet with
# NanoAOD pT > 300, |eta| < 2.5), so the selected events are unchanged; only events that cannot pass are
# not written.  Consequences: the NanoAOD (and ttbar_maker's cutflow "all") counts filtered events only.
# MC: the gen-weight counter (genWeightsTable -> Runs/genEventSumw, genEventCount) runs on its own
# unfiltered path, so the normalisation still counts every input event.
import os
import FWCore.ParameterSet.Config as cms

_base = os.environ.get("AOJ_TTSKIM_BASE", "")
if not _base or not os.path.isfile(_base):
    raise RuntimeError("pfnano_ttskim.py: set AOJ_TTSKIM_BASE to the base PFNano config (got '%s')" % _base)
exec(compile(open(_base).read(), _base, "exec"))       # defines `process` (and parses the command line)

process.ttskimLeptons = cms.EDProducer("CandViewMerger",
                                       src=cms.VInputTag("slimmedMuons", "slimmedElectrons"))
process.ttskimLepton = cms.EDFilter("CandViewSelector", src=cms.InputTag("ttskimLeptons"),
                                    cut=cms.string("pt > 20 && abs(eta) < 2.6"), filter=cms.bool(True))
process.ttskimAK8 = cms.EDFilter("CandViewSelector", src=cms.InputTag("slimmedJetsAK8"),
                                 cut=cms.string("pt > 200 && abs(eta) < 2.6"), filter=cms.bool(True))
process.ttskimSequence = cms.Sequence(process.ttskimLeptons + process.ttskimLepton + process.ttskimAK8)
process.nanoAOD_step.insert(0, process.ttskimSequence)
process.NANOAODSIMoutput.SelectEvents = cms.untracked.PSet(SelectEvents=cms.vstring("nanoAOD_step"))

_mc = "genWeightsTable" in process.nanoAOD_step.moduleNames()   # MC config (nano_cff loads it for data too)
if _mc:
    process.ttskimGenWeights = cms.Path(process.genWeightsTable)
    process.schedule.insert(0, process.ttskimGenWeights)
print("pfnano_ttskim: base %s + ttbar pre-filter (lepton pT > 20, AK8 pT > 200)%s" % (
    _base, "; genWeightsTable unfiltered" if _mc else ""))
