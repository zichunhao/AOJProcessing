set -e
cmsenv

python H5_merge.py merged.h5 out_run_2016G_*.h5 out_run_2016H_*.h5
