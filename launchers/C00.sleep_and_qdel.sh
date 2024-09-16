#!/bin/zsh

hours=$1



qlist=$PROJECTS/qsub-launcher/scripts/qlist.sh

# sleep $hours hours
sleep "${hours}"h

num_jobs=`$qlist -f | grep run_causal_prover | wc -l`
echo $num_jobs
delete_jobs=$((num_jobs - 3))
$qlist -f | grep run_causal_prover | tail -n $delete_jobs | awk '{print \$1}' | xargs qdel
