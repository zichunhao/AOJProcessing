#!/usr/bin/env python3
"""<store>/ttbar_skim (skim preset) vs <store>/ttbar (an preset), chunk by chunk, and a cache of the skim columns.

    python3 skim_vs_an.py CACHE_DIR NWORKERS

Per sample: identical n_entries / sum_genweight per chunk (same inputs), and 'an' recovered from the skim rows
(an_channel == channel, the 'an' lepton paths or JetHT, MET > 50, n_ak8 == 1) gives the same events as the 'an'
production from the dphi stage on, with identical probe pT and tau32. Writes CACHE_DIR/<label>.npz for
skim_variants.py.
"""
import sys, os, glob, re, json, numpy as np, h5py
from multiprocessing import Pool
S = "/ceph/cms/store/user/zichun/parcel/AOJ"
STAGES = None
def cols(d): return d.attrs["columns"].split(",")
def one(name):
    out = {"name": name}
    with h5py.File("%s/ttbar/%s" % (S, name), "r") as fa, h5py.File("%s/ttbar_skim/%s" % (S, name), "r") as fs:
        out["a_ne"], out["s_ne"] = int(fa.attrs["n_entries"]), int(fs.attrs["n_entries"])
        out["a_sw"], out["s_sw"] = float(fa.attrs["sum_genweight"]), float(fs.attrs["sum_genweight"])
        sel = json.loads(fs.attrs["selection"]); out["s_sel"] = (sel["preset"], sel["trigger"], sel["store"])
        out["s_ver"] = fs.attrs.get("versions", "")
        stages = fs["cutflow"].attrs["stages"].split(",")
        ca, cs = cols(fa["ttbar_tag"]), cols(fs["ttbar_tag"])
        A, T = fa["ttbar_tag"][:], fs["ttbar_tag"][:]
        aev, sev = fa["event_info"][:], fs["event_info"][:]
        apt, skin = fa["jet_kinematics"][:, 0], fs["jet_kinematics"][:]
        tc = cols(fs["truth"]); cat = fs["truth"][:, tc.index("category")] if len(T) else np.zeros(0)
        gw = fs["gen_weight"][:]
        out["s_cf_n"], out["a_cf_n"] = fs["cutflow/n"][:], fa["cutflow/n"][:]
    a = {k: A[:, i] for i, k in enumerate(ca)}; s = {k: T[:, i] for i, k in enumerate(cs)}
    out["cs"], out["stages"] = cs, stages
    # 'an' recovered from the skim rows (stored-row semantics of the an production: lepton(an) OR jetht trigger)
    if len(T):
        ch = s["channel"]
        trig_an = np.where(ch == 1, s["trig_mu_an"], np.where(ch == 2, s["trig_el_an"], 0)) > 0
        an_ok = (s["an_channel"] == ch) & (ch > 0) & (trig_an | (s["trig_jetht"] > 0)) & (s["met_pt"] > 50) & (s["n_ak8"] == 1)
    else:
        an_ok = np.zeros(0, bool)
    k = lambda ev: ev[:, 1].astype(np.int64) * (1 << 32) + ev[:, 2]
    res = {}
    for st in ("dphi", "btag", "tau32"):
        si = stages.index(st)
        ma = a["stage"] >= si if len(A) else np.zeros(0, bool)
        ms = an_ok & (s["stage"] >= si) if len(T) else np.zeros(0, bool)
        ka, ks = k(aev[ma]), k(sev[ms])
        common, ia, is_ = np.intersect1d(ka, ks, return_indices=True)
        pt_bad = int((np.abs(apt[ma][ia] - skin[ms][is_, 0]) > 1e-3).sum())
        tau_bad = int((np.abs(np.nan_to_num(a["tau32"][ma][ia]) - np.nan_to_num(s["tau32"][ms][is_])) > 1e-6).sum()) if len(ia) else 0
        res[st] = (len(ka), len(ks), len(common), pt_bad, tau_bad)
    out["cmp"] = res
    out["rows"] = len(T)
    out["data"] = dict(tag=T, kin=skin, cat=cat, gw=gw, ev=sev)
    return out
if __name__ == "__main__":
    cache = sys.argv[1]; os.makedirs(cache, exist_ok=True)
    names = sorted(os.path.basename(p) for p in glob.glob(S + "/ttbar_skim/tt_*.h5"))
    lab = lambda n: re.sub(r"_sub\d+\.h5$", "", n)[3:]
    per = {}
    with Pool(int(sys.argv[2])) as pool:
        for i, r in enumerate(pool.imap_unordered(one, names, chunksize=4)):
            per.setdefault(lab(r["name"]), []).append(r)
            if i % 1000 == 0: print(i, len(names), flush=True)
    print("\n%-18s %5s %6s %11s %4s %4s %9s | %s" % ("label", "files", "empty", "entries", "ne!=", "sw!=", "skim rows", "an recovered at dphi / btag / tau32: an, skim-an, common, pt/tau32 mismatches"))
    allsel = set()
    for l in sorted(per):
        R = sorted(per[l], key=lambda r: r["name"])
        ne_bad = sum(r["a_ne"] != r["s_ne"] for r in R); sw_bad = sum(abs(r["a_sw"] - r["s_sw"]) > 1e-6 * max(1, abs(r["a_sw"])) for r in R)
        empty = sum(r["rows"] == 0 for r in R); zero_ne = sum(r["s_ne"] == 0 for r in R)
        for r in R: allsel.add((r["s_sel"], r["s_ver"]))
        cmp_ = {st: tuple(sum(r["cmp"][st][i] for r in R) for i in range(5)) for st in ("dphi", "btag", "tau32")}
        print("%-18s %5d %6d %11d %4d %4d %9d | %s%s" % (l, len(R), empty, sum(r["s_ne"] for r in R), ne_bad, sw_bad, sum(r["rows"] for r in R),
              "  ".join("%d/%d/%d/%d/%d" % cmp_[st] for st in ("dphi", "btag", "tau32")), "  ZERO-ENTRY FILES: %d" % zero_ne if zero_ne else ""))
        D = [r["data"] for r in R]
        np.savez(os.path.join(cache, l + ".npz"), tag=np.concatenate([d["tag"] for d in D]), kin=np.concatenate([d["kin"] for d in D]),
                 cat=np.concatenate([d["cat"] for d in D]), gw=np.concatenate([d["gw"] for d in D]), ev=np.concatenate([d["ev"] for d in D]),
                 sumw=np.array([r["s_sw"] for r in R]), n_entries=np.array([r["s_ne"] for r in R]), cols=np.array(",".join(R[0]["cs"])),
                 stages=np.array(",".join(R[0]["stages"])), cf_n=sum(r["s_cf_n"] for r in R), a_cf_n=sum(r["a_cf_n"] for r in R))
    print("\ndistinct (preset, trigger, store), versions in the skim files:")
    for x in sorted(allsel, key=str): print("  ", x)
