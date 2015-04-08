branchAndBound
==============

How to compile
--------------

```
make [optimization-seq|optimization-openmp|optimization-mpi]
```

For mpi you need to change the value of `BINROOT` to the path of `mpic++`

How to launch
-------------

- Sequentcial : `./optimization-seq [function precision]`
- Parallel (OMP) : `./optimization-openmp [function precision]`
- Parallel (MPI) : `mpirun -H host1,host2 -n 4 ./optimization-mpi [function precision]`

If you run without parameter, you have access of help.
