#!/bin/bash
#SBATCH --job-name="bddc_hts"
#SBATCH --workdir=/home/bsc21/bsc21910/Projects/HTS/bddc/src
#SBATCH --output=bddc.out
#SBATCH --error=bddc.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
module load python/3.8.2
python3 -u main_Alya3D.py
