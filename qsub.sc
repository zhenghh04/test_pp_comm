#!/bin/bash
#PBS -l walltime=0:10:00          
#PBS -l select=4                    
#PBS -A Aurora_deployment
#PBS -q lustre_scaling
#PBS -l filesystems=flare:home

cd ${PBS_O_WORKDIR}
PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
PPN=12
module load frameworks
echo "PBS_JOBSIZE: $PBS_JOBSIZE, PPN: $PPN"

echo "w/o initialization"
mpiexec -np $((PBS_JOBSIZE*PPN)) --ppn $PPN ./local_rank.sh python3 ./test_pp_comm.py --pp $PBS_JOBSIZE | tee -a comm.$PBS_JOBSIZE.log

echo "w/ Initialization"
mpiexec -np $((PBS_JOBSIZE*PPN)) --ppn $PPN ./local_rank.sh python3 ./test_pp_comm.py --pp $PBS_JOBSIZE --init | tee -a comm.$PBS_JOBSIZE.init.log
