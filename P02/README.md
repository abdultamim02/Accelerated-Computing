**Programming Assignement 02 Documentation**
=================================

## Create/Access Files

### Create Parallel File

- Create HW1_omp.c File

```
emacs -nw HW2_omp.c
```

- Compiles the `HW2_ser.c` file using `GCC` with OpenMP support and generates an executable named `HW2_ser`

```
gcc -fopenmp HW2_omp.c -o HW2_omp -lm
```

- Remove the environment variable `OMP_NUM_THREADS` from the current shell session, if it was previously set. This ensures that any prior setting of the number of OpenMP threads is cleared. Then, set the environment variable `OMP_NUM_THREADS` to `#`, telling OpenMP that the program should run with only `#` number of threads. Run the following command:

```
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=#
```

- Run and Execute the Program

```
./HW2_omp #(Number of Threads) #(Number of Nodes)
```
