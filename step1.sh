set -e
cmsenv

NJOBS=8           # parallel jobs per dataset (16 total)
NTHREADS=8        # threads per job (16 * 8 = 128 cores used)
BASEDIR=$(pwd)

# Split a file list into N chunks and launch parallel cmsRun jobs
# Usage: run_batches <dataset_name> <file_list>
# Creates directories: run_<dataset_name>_00, run_<dataset_name>_01, ...
run_batches() {
    local name=$1
    local filelist=$2
    local total
    total=$(wc -l < "$filelist")
    local per_job=$(( (total + NJOBS - 1) / NJOBS ))

    # Split file list into chunks
    local splitdir="$BASEDIR/split_${name}"
    mkdir -p "$splitdir"
    split -l "$per_job" -d -a 2 "$filelist" "$splitdir/chunk_"

    # Launch a cmsRun job per chunk
    local i=0
    for chunk in "$splitdir"/chunk_*; do
        local rundir="$BASEDIR/run_${name}_$(printf '%02d' $i)"
        mkdir -p "$rundir"
        (
            cd "$rundir"
            cmsRun "$BASEDIR/pfnano_data_2016UL_OpenData.py" \
                inputFiles_load="$chunk" \
                nThreads=$NTHREADS
        ) &
        PIDS+=($!)
        i=$((i + 1))
    done
}

PIDS=()

run_batches "2016G" "$BASEDIR/file_lists/2016G.txt"
run_batches "2016H" "$BASEDIR/file_lists/2016H.txt"

echo "Launched ${#PIDS[@]} parallel jobs"

# Wait for all and track failures
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

echo "All jobs completed successfully"
echo "Outputs in run_2016G_*/nano_data2016.root and run_2016H_*/nano_data2016.root"
