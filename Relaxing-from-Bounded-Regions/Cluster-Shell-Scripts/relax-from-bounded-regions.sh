#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=62gb
#PBS -N Escape-Rate
#PBS -J 1-20000

module load anaconda3/personal
source activate personalpy3
date

python $PBS_O_WORKDIR/main.py $PBS_ARRAY_INDEX
