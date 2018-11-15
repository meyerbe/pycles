#!/bin/sh
#
#SBATCH -p aegir
#SBATCH -A ocean
#SBATCH --job-name='pycles_triple'
#SBATCH --time=06:00:00
#SBATCH --constraint=v1
#SBATCH --nodes=2
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mymail@nbi.ku.dk #SBATCH --output=slurm.out
var0=$1
echo $var0 > out.txt
#srun --mpi=pmi2 --kill-on-bad-exit python main.py ${var0}
srun --mpi=pmi2 --kill-on-bad-exit python main_aegir2.py ${var0}
#srun --mpi=pmi2 --kill-on-bad-exit python main_aegir.py 
