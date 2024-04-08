#!/bin/bash
#SBATCH --mail-type=ALL		
#SBATCH --mail-user=shuhei.hara1@oist.jp
#SBATCH --array=1-21
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=45:00
#SBATCH -o log/output-%A-%a.txt
#SBATCH --job-name=glm_circular
#SBATCH --partition=compute
##### END OF JOB DEFINITION  #####

export STUDY=/home/s/shuhei-hara1

DIR_BIDS="/bucket/DoyaU/Shuhei/cat_fox/fMRI/heudiconv/BIDS"
DIR_PREP="/bucket/DoyaU/Shuhei/cat_fox/fMRI/fmriprep"
DIR_OUTPUT="/flash/DoyaU/shuhei/GLM/cpsy_tokyo"
DIR_SCRIPT="/home/s/shuhei-hara1/workspace/GLM2"


module load python/3.7.3

subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" /bucket/DoyaU/Shuhei/cat_fox/fMRI/heudiconv/BIDS/participants.tsv)
# subject='DI'

# DIR_BIDS="/bucket/DoyaU/Shuhei/cat_fox/fMRI/heudiconv/BIDS/sub-${subject}"
# DIR_PREP="/bucket/DoyaU/Shuhei/cat_fox/fMRI/fmriprep/sub-${subject}"

cmd='python3 ${DIR_SCRIPT}/run.py ${DIR_PREP} ${DIR_OUTPUT} run --bids-dir ${DIR_BIDS} --space template --fwhm 6.0 --participant-label ${subject}'
# python3 ${DIR_SCRIPT}/run_subject.py --participant-label ${subject}


# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

# Output results to a table
# echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
#       >> ${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode