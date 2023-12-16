#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-19
#SBATCH --cpus-per-task=6
#SBATCH --time=1:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=amlds
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=log/amlds_%A_%a.out
#SBATCH --error=log/amlds_%A_%a.err
#SBATCH --gres=gpu:1

singularity exec --nv \
	    --overlay $SCRATCH/environments/pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh;
        python adam.py --job_index ${SLURM_ARRAY_TASK_ID}"