**Programming Assignement 03 Documentation**
=================================

## Create/Access Files

### Create Parallel File

- Create Simpson.cpp File

```
emacs -nw Simpson.cpp
```

- Compiles the `Simpson.cpp` file using `g++` with OpenMP support and generates an executable named `Simpson`

```
g++ -fopenmp Simpson.cpp -o Simpson
```

- Remove the environment variable `OMP_NUM_THREADS` from the current shell session, if it was previously set. This ensures that any prior setting of the number of OpenMP threads is cleared. Then, set the environment variable `OMP_NUM_THREADS` to `#`, telling OpenMP that the program should run with only `#` number of threads. Finally, Run and Execute the Program using the following command:

```
(unset OMP_NUM_THREADS; export OMP_NUM_THREADS=(# Number of Threads); ./Simpson (# Number of Threads) 1e6)
```
