set -e
cmsenv

CERT=Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt
PIDS=()

for rundir in run_2016G_* run_2016H_*; do
    [ -f "$rundir/nano_data2016.root" ] || continue
    outname="out_${rundir}.h5"
    python H5_maker.py -i "$rundir/nano_data2016.root" -o "$outname" --sample_type data -j $CERT &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} H5_maker jobs"

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "Job $pid failed"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "$FAILED job(s) failed"
    exit 1
fi

echo "All H5 jobs completed"
