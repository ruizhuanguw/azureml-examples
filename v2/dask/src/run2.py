print()
print("*"*100)
print()

print("importing mpi4py...")
from mpi4py import MPI 

print("getting stuff...")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print("printing stuff...")
print(comm)
print(rank)

print()
print("*"*100)
print()