#!/usr/bin/env python3
"""R&D job list: a few evenly spread NanoAODv9 files per sample (DAS), read remotely via FNAL xrootd.

    python3 make_jobs.py            # -> jobs.txt, samples.json (needs a grid proxy in ~/.x509)

Cross sections (pb, 13 TeV): ttbar / single top = CMS standard (sigma_tt = 831.76 pb); QCD HT and W+jets
(x 1.21 NNLO k-factor) = HH4b/src/HH4b/xsecs.json 13 TeV values; QCD MuEnriched = Run-2 XSDB sigma x filter
efficiency (quoted from memory: check before using beyond R&D).  Data: scaled to 2016G+H by event count.
"""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MC = "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17"
DAS = "/cvmfs/cms.cern.ch/common/dasgoclient"
REDIR = "root://cmsxrootd.fnal.gov/"
# name: (dataset, n files, xsec pb (MC) or None (data), group)
S = {
    "tt": ("/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/%s-v1/NANOAODSIM" % MC, 3, 365.34, "ttbar"),
    "tt2l": ("/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/%s-v1/NANOAODSIM" % MC, 3, 88.29, "ttbar"),
    "qcdmu170": ("/QCD_Pt-170To300_MuEnrichedPt5_TuneCP5_13TeV-pythia8/%s-v1/NANOAODSIM" % MC, 3, 8654.49, "qcd_mu"),
    "qcdmu300": ("/QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8/%s-v1/NANOAODSIM" % MC, 4, 797.35, "qcd_mu"),
    "qcdmu470": ("/QCD_Pt-470To600_MuEnrichedPt5_TuneCP5_13TeV-pythia8/%s-v1/NANOAODSIM" % MC, 4, 79.03, "qcd_mu"),
    "qcdmu600": ("/QCD_Pt-600To800_MuEnrichedPt5_TuneCP5_13TeV-pythia8/%s-v1/NANOAODSIM" % MC, 4, 25.10, "qcd_mu"),
    "qcdmu800": ("/QCD_Pt-800To1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8/%s-v1/NANOAODSIM" % MC, 3, 4.707, "qcd_mu"),
    "qcdmu1000": ("/QCD_Pt-1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8/%s-v1/NANOAODSIM" % MC, 3, 1.621, "qcd_mu"),
    "qcdht300": ("/QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/%s-v1/NANOAODSIM" % MC, 3, 347700., "qcd_ht"),
    "qcdht500": ("/QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/%s-v1/NANOAODSIM" % MC, 3, 30330., "qcd_ht"),
    "qcdht700": ("/QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/%s-v1/NANOAODSIM" % MC, 3, 6412., "qcd_ht"),
    "qcdht1000": ("/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/%s-v1/NANOAODSIM" % MC, 3, 1118., "qcd_ht"),
    "qcdht1500": ("/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/%s-v1/NANOAODSIM" % MC, 2, 108.5, "qcd_ht"),
    "qcdht2000": ("/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/%s-v1/NANOAODSIM" % MC, 2, 21.94, "qcd_ht"),
    "wj100": ("/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/%s-v1/NANOAODSIM" % MC, 2, 1270. * 1.21, "wjets"),
    "wj200": ("/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/%s-v1/NANOAODSIM" % MC, 3, 336.8 * 1.21, "wjets"),
    "wj400": ("/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/%s-v1/NANOAODSIM" % MC, 3, 44.64 * 1.21, "wjets"),
    "wj600": ("/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/%s-v1/NANOAODSIM" % MC, 3, 11.6 * 1.21, "wjets"),
    "wj800": ("/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/%s-v1/NANOAODSIM" % MC, 3, 4.898 * 1.21, "wjets"),
    "wj1200": ("/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/%s-v1/NANOAODSIM" % MC, 3, 1.142 * 1.21, "wjets"),
    "wj2500": ("/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/%s-v2/NANOAODSIM" % MC, 2, 0.02678 * 1.21, "wjets"),
    "sttw": ("/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/%s-v1/NANOAODSIM" % MC, 3, 19.47, "single_top"),
    "sttwbar": ("/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/%s-v1/NANOAODSIM" % MC, 3, 19.47, "single_top"),
    "sttch": ("/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/%s-v1/NANOAODSIM" % MC, 2, 136.02, "single_top"),
    "sttchbar": ("/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/%s-v1/NANOAODSIM" % MC, 2, 80.95, "single_top"),
    "smG": ("/SingleMuon/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD", 3, None, "data_mu"),
    "smH": ("/SingleMuon/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD", 3, None, "data_mu"),
    "seG": ("/SingleElectron/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD", 2, None, "data_el"),
    "seH": ("/SingleElectron/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD", 2, None, "data_el"),
}
CERT = os.path.join(HERE, "..", "..", "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")


def das(q):
    env = dict(os.environ, X509_USER_PROXY=os.path.expanduser("~/.x509/x509up_u%d" % os.getuid()))
    return subprocess.run([DAS, "-query=" + q], capture_output=True, text=True, env=env, timeout=180).stdout


def main():
    jobs, table = [], {}
    for name, (ds, n, xs, grp) in S.items():
        files = sorted(das("file dataset=%s" % ds).split())
        if not files:
            raise SystemExit("no files for %s" % ds)
        pick = [files[int(i * len(files) / n)] for i in range(min(n, len(files)))]
        n_ds = json.loads(das("summary dataset=%s" % ds))[0]["nevents"]
        args = ("--sample_type data -j %s" % os.path.abspath(CERT)) if xs is None else "--sample_type MC"
        for i, lfn in enumerate(pick):
            jobs.append("%s_%d -i %s%s %s --trigger lepton" % (name, i, REDIR, lfn, args))
        table[name] = {"dataset": ds, "xsec": xs, "group": grp, "n_files": len(files), "n_events_dataset": n_ds}
        print("%-10s %2d/%3d files  %11d events  %s" % (name, len(pick), len(files), n_ds, ds))
    open(os.path.join(HERE, "jobs.txt"), "w").write("\n".join(jobs) + "\n")
    json.dump(table, open(os.path.join(HERE, "samples.json"), "w"), indent=1)
    print("%d jobs -> jobs.txt" % len(jobs))


if __name__ == "__main__":
    main()
