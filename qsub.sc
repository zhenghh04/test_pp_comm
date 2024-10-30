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
OUTPUT=./outputs/n$PBS_JOBSIZE/$(date '+%Y-%m-%d-%H-%M-%S')
mkdir -p $OUTPUT

echo "w/o initial"
mpiexec -np $((PBS_JOBSIZE*PPN)) --ppn $PPN ./local_rank.sh python3 ./test_pp_comm.py --pp $PBS_JOBSIZE --output $OUTPUT/comm-wo-init.log --trace $OUTPUT/torch-wo-init

echo "w/ comm init"
mpiexec -np $((PBS_JOBSIZE*PPN)) --ppn $PPN ./local_rank.sh python3 ./test_pp_comm.py --pp $PBS_JOBSIZE --init-comm --output $OUTPUT/comm-w-init-comm.log --trace $OUTPUT/torch-w-init-comm

echo "w/ comm and comp init"
mpiexec -np $((PBS_JOBSIZE*PPN)) --ppn $PPN ./local_rank.sh python3 ./test_pp_comm.py --pp $PBS_JOBSIZE --init-comm --init-comp --output $OUTPUT/comm-w-init-comm-comp.log --trace $OUTPUT/torch-w-init-comm-comp
