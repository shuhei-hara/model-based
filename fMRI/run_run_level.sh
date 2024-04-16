#!/bin/bash
#SBATCH --mail-type=ALL		
#SBATCH --mail-user= # mail
#SBATCH --array= # number of subjects 
#SBATCH -n  # node
#SBATCH --cpus-per-task= # cpu
#SBATCH --mem-per-cpu= # memory
#SBATCH --time= # runtime
#SBATCH -o log/output-%A-%a.txt
#SBATCH --job-name= # job name
#SBATCH --partition= # short or compute
##### END OF JOB DEFINITION  #####

export STUDY= # current directory

DIR_BIDS= # BIDS directory
DIR_PREP= # fmriprep directory
DIR_OUTPUT= # output directory
DIR_SCRIPT= # code directory


module load python/3.7.3

subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" /{bids directory}/participants.tsv)


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
