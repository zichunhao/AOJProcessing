"""Validate PFNano NanoAOD ROOT files (run inside cmssw-el7 via PyROOT).

A cmsRun killed mid-write leaves a large but unclosed file that ROOT can only
"recover" -- so size is not proof of completeness. We treat a file as complete
only if it opens cleanly (not recovered/zombie) and has non-empty Events and
Runs trees.

Prints one TSV line per file:  <path>\t<OK|BAD>\t<n_events>\t<reason>
"""
import sys
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kError

for f in sys.argv[1:]:
    ok, n, reason = "BAD", -1, ""
    tf = ROOT.TFile.Open(f)
    if (not tf) or tf.IsZombie():
        reason = "zombie_or_unopenable"
    else:
        recovered = bool(tf.TestBit(ROOT.TFile.kRecovered))
        ev = tf.Get("Events")
        runs = tf.Get("Runs")
        n = int(ev.GetEntries()) if ev else -1
        if recovered:    reason = "recovered_truncated"
        elif not ev:     reason = "no_Events_tree"
        elif not runs:   reason = "no_Runs_tree"
        elif n <= 0:     reason = "empty"
        else:            ok, reason = "OK", "complete"
        tf.Close()
    print("%s\t%s\t%d\t%s" % (f, ok, n, reason))
