#!/bin/sh
#PBS -N mpi_gauss
#PBS -e mpi.e
#PBS -o mpi.o
#PBS -l nodes=8


# 在所有分配的计算节点上创建工作目录
/usr/local/bin/pssh -h $PBS_NODEFILE mkdir -p /home/${USER} 1>&2

# 从主节点拷贝可执行文件
scp master_ubss1:/home/${USER}/gauss/main_mpi /home/${USER} 1>&2

# 分发可执行文件到所有计算节点
/usr/local/bin/pscp -h $PBS_NODEFILE /home/${USER}/main_mpi /home/${USER} 1>&2

# 获取实际分配的进程数
NP=$(cat $PBS_NODEFILE | wc -l)

# MPI 运行
/usr/local/bin/mpiexec -np $NP -machinefile $PBS_NODEFILE /home/${USER}/main_mpi

# 清理
rm /home/${USER}/main_mpi
