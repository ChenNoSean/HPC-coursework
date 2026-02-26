# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.
#!/bin/bash
module purge
module load PrgEnv-gnu


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-144}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_SCHEDULE=static
export OMP_DYNAMIC=false