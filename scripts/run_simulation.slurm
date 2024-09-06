#!/bin/bash
#SBATCH -J StabilityAnalysis
#SBATCH --mail-user npradeep96
#SBATCH -p sched_mit_arupc_long
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu 4000
cd "$SLURM_SUBMIT_DIR"
echo $PWD

run_program()
{
    source activate fipyenv
    python run_simulation.py --i $input_file --o $out_folder
    conda deactivate
}

cleanup_files()
{
    rm input_parameters_$SLURM_JOBID.txt
    echo "DONE"
}

run_program
cleanup_files