# L1 pre-firing weights: ttbar_maker format 4 and the CR MC rerun

Instructions for the LPC session (production) and the UAF session (SF inputs), branch `ttbar-semilep-prefiring`
of `git@github.com:zichunhao/AOJProcessing.git`. Written 2026-11-21 on UAF. Ask the user before anything marked
*(ask first)*.

## Why
The SF inputs `sf_inputs_20261120_iso30_met20/` (UAF, `/ceph/cms/store/user/zichun/parcel/AOJ/`, README there)
show the L1 ECAL pre-firing in data: at `iso30_met20 & btag`, data / MC of the electron channel is 0.794 +- 0.031
for electrons at 2.0 < abs(SC eta) < 2.4 (muons at the same abs(eta): 0.996 +- 0.032), and forward probes are low
as well (probe abs(eta) 2.0-2.5: e 0.839, mu 0.946). In 2016 the L1 ECAL trigger fired a bunch crossing early for
electrons, photons and jets at 2 < abs(eta) < 3; the MC does not have it, and CMS corrects the MC with the
event weight `L1PreFiringWeight_Nom` (1 - pre-firing probability) of the NanoAOD. `ttbar_maker.py` (format 3) does
not store it, so the format-3 MC cannot be corrected.

## What this branch changes
| file | change |
|---|---|
| `ttbar_semilep/ttbar_selection.py` | reads the NanoAOD `L1PreFiringWeight_{Nom,Up,Dn,ECAL_Nom,Muon_Nom}` (optional branches: NaN where absent) |
| `ttbar_semilep/ttbar_maker.py` | `ttbar_tag` gets five appended columns `l1pf_nom, l1pf_up, l1pf_dn, l1pf_ecal_nom, l1pf_muon_nom`; `FORMAT_VERSION = 4` |
| `ttbar_semilep/tests/test_ttbar.py` | `test_real_mc_prefiring_weights`: the columns equal the NanoAOD branches, finite, in (0, 1], up <= nom <= dn |
| `condor/config.sh`, `condor/README.md` | default code tarball `ttbar_semilep_v4.tar.gz` |
| `finetune/make_sf_inputs.py`, `finetune/check_sf_inputs.py` | UAF: `sf_l1pf` (+ `_up`, `_down`) from format-4 MC, in the nominal weight (1 for format-3 inputs); MC and data may come from different productions (`--data-src`) |
| `ttbar_semilep/prefiring/check_prefiring.py` | compares format-4 outputs with the format-3 outputs of the same chunks |

Nothing else changes: every other dataset and column is bit-identical to format 3. Checked on UAF on 44,000
events of a real NanoAOD (the 2018 TTToSemiLeptonic PFNano of the tests, `--preset skim --store one_probe`): v4 vs
v3 `check_prefiring.py` ALL OK, mean `l1pf_nom` 0.998 there (2018: muon part only). `test_ttbar.py`: 17 passed.

Facts about the weights (CMSSW_10_6_30 `PhysicsTools/NanoAOD/python/triggerObjects_cff.py`): the NanoAOD of UL
2016 post-VFP computes them with `DataEraECAL = UL2016postVFP`, `DataEraMuon = 2016postVFP`, for MC and also for
data (where they mean nothing: apply them to MC only). `Up` = more pre-firing, i.e. a lower weight. In the AOJ
2016H data PFNano file the computed `L1PreFiringWeight_Nom` has mean 0.979 and minimum 0.64: a percent-level
effect, larger for forward objects. Only the CR **MC** needs the rerun; the format-3 data outputs stay.

## Rules (from the user)
- Only public CMS Open Data: every input is an eospublic path (`root://eospublic.cern.ch//eos/opendata/cms/`).
- Store writes and reads over xrootd only through `root://redir.t2.ucsd.edu:1095/`; never use
  `redirector.t2.ucsd.edu`, not even as a fallback (UCSD admin rule).
- `<store>/sf_samples_20261026_skim/` is read-only: never write, move or delete anything in it.
- Commits: no Claude attribution lines; never commit `CLAUDE.md`; stage named files only.
- Ask the user before submitting the full production.

## LPC: steps
1. **Code.** `git fetch origin && git checkout ttbar-semilep-prefiring` (it is `origin/ttbar-semilep` 5294f8e plus
   one commit).
2. **Tests** on a 2016 MC file, which UAF did not have (it checked a 2018 MC file and the 2016 data PFNano):
   ```bash
   AOJ_TT_TEST_MC=<a UL16 post-VFP MC NanoAOD/PFNano with GenPart> AOJ_TT_TEST_MC_ERA=2016postVFP \
       bash ttbar_semilep/py3env.sh ttbar_semilep/tests/test_ttbar.py
   ```
   Expect 0 failed (17 passed on UAF; `test_real_pfnano_pfcands_like_h5maker` skips without the AOJ data PFNano
   file, `AOJ_TT_TEST_PFNANO`) and `test_real_mc_prefiring_weights` PASS, not SKIP: that confirms the branches are
   in our 2016 PFNano MC. If it is skipped because the branches are missing, stop and tell the user.
3. **Pack** the v4 code once: `cd condor && ./pack_ttbar.sh` (packs the committed `ttbar_semilep/` to
   `<store>/cmssw/ttbar_semilep_v4.tar.gz`; never replaces an existing tarball).
4. **Smoke test: one chunk**, the same one as in the v3 production, in a test directory:
   ```bash
   cd condor
   python3 ../ttbar_semilep/samples.py --regen TTToSemiLeptonic > regen_lists.txt
   echo TTToSemiLeptonic_sub0000 > smoke_v4.txt
   AOJ_TTBAR=only AOJ_TTBAR_PRESET=skim AOJ_TTBAR_SUBDIR=ttbar_test_v4 AOJ_MISSING=smoke_v4.txt ./run.sh jobs
   ./run.sh submit
   ```
   No `AOJ_MAX_EVENTS`: the chunk must be complete to compare with v3. When it is done:
   ```bash
   bash ttbar_semilep/py3env.sh ttbar_semilep/prefiring/check_prefiring.py \
       <store>/ttbar_test_v4/tt_TTToSemiLeptonic_sub0000.h5 <store>/sf_samples_20261026_skim/tt_TTToSemiLeptonic_sub0000.h5
   ```
   Expect `ALL OK` (identical but for the five new columns) and a mean `l1pf_nom` of ~0.97-0.99.
5. **Production** *(ask first)*: every CR MC sample (53; not the data, not TTToHadronic), same settings as v3,
   FILES_PER_JOB = 3, into a new dated directory:
   ```bash
   cd condor
   python3 ../ttbar_semilep/samples.py --regen | grep -v '^Single' > regen_lists.txt      # 53 lines
   AOJ_TTBAR=only AOJ_TTBAR_PRESET=skim AOJ_TTBAR_SUBDIR=sf_samples_<YYYYMMDD>_skim ./run.sh jobs
   ./run.sh submit
   ```
   ~6,400 jobs, as the MC part of the v3 production. Use `AOJ_MISSING=<list>` for resubmissions, as before.
6. **Validate**: every chunk present and readable (your `ttbar_lpc/sf_validate.py`; 6,381 files of 53 samples,
   the same chunk names as in `sf_samples_20261026_skim/`), then the whole directory against v3:
   ```bash
   bash ttbar_semilep/py3env.sh ttbar_semilep/prefiring/check_prefiring.py \
       <store>/sf_samples_<YYYYMMDD>_skim <store>/sf_samples_20261026_skim
   ```
   Expect `6381 pairs compared; ALL OK`: every stored event and column the same as in v3, only the weights added.
   Report the directory name and the per-sample mean `l1pf_nom` (printed by the check) to the user.
7. **Merge**: after the user agrees, fast-forward `ttbar-semilep` to this branch and push.

## UAF: after the production
Rebuild the SF inputs with the format-4 MC and the format-3 data (a new directory; the current one stays until
the new one is checked):
```bash
bash ttbar_semilep/py3env.sh finetune/make_sf_inputs.py -o <store>/sf_inputs_<YYYYMMDD>_iso30_met20 \
    --src <store>/sf_samples_<YYYYMMDD>_skim --data-src <store>/sf_samples_20261026_skim \
    --lumi-csv pp_2016lumibyls.csv --el-trig-sf el_trig_sf.json
bash ttbar_semilep/py3env.sh finetune/check_sf_inputs.py <store>/sf_inputs_<YYYYMMDD>_iso30_met20
bash ttbar_semilep/py3env.sh finetune/sf_yields.py <store>/sf_inputs_<YYYYMMDD>_iso30_met20
bash ttbar_semilep/py3env.sh finetune/sf_plots.py <store>/sf_inputs_<YYYYMMDD>_iso30_met20
```
`pp_2016lumibyls.csv`: CERN Open Data record 1059; `el_trig_sf.json`: the one in `sf_inputs_20261120_iso30_met20/`
(or `finetune/etrig_sf.py`, which reads the data and the format-3 MC). The weight columns gain `sf_l1pf`,
`sf_l1pf_up`, `sf_l1pf_down`, and `weight` includes `sf_l1pf`; the build log prints, per sample, the rows that
have it (all MC rows, none of the data). Expected: the weights lower the MC by ~1-2 % (more for forward objects),
so data / MC rises by about that much, the forward-electron deficit (0.79) mostly goes, and the e and mu channels
come closer.
