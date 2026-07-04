#!/bin/sh
#PBS -N groebner
#PBS -e groebner.e
#PBS -o groebner.o
#PBS -l nodes=NODES

/usr/local/bin/pssh -h $PBS_NODEFILE mkdir -p /home/${USER} 1>&2
scp master_ubss1:/home/${USER}/gauss/groebner_mpi /home/${USER} 1>&2
/usr/local/bin/pscp -h $PBS_NODEFILE /home/${USER}/groebner_mpi /home/${USER} 1>&2

NP=$(cat $PBS_NODEFILE | wc -l)
/usr/local/bin/mpiexec -np $NP -machinefile $PBS_NODEFILE /home/${USER}/groebner_mpi /home/${USER}/gauss/groebner_test 2000 500 1000

rm /home/${USER}/groebner_mpi
