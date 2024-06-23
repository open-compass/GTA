# demo: tools/internal/cancel_job.sh OpenICL
regx=$1
jobnames="$(squeue -o "%.j" -u $USER | grep $regx )"
for jobname in $jobnames; do
    echo $jobname
    scancel -n $jobname
done
