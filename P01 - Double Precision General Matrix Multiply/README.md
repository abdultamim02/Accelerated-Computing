**Programming Assignement 01 Documentation**
=================================

## Create/Access Files

### Create Serial File

- Create HW1_ser.c File

```
emacs -nw HW1_ser.c
```

- Remove the environment variable `OMP_NUM_THREADS` from the current shell session, if it was previously set. This ensures that any prior setting of the number of OpenMP threads is cleared. Then, set the environment variable `OMP_NUM_THREADS` to `1`, telling OpenMP that the program should run with only `1` thread. Run the following command:

```
unset OMP_NUM_THREADS; export OMP_NUM_THREADS=1
```

- Compiles the `HW1_ser.c` file using `GCC` with OpenMP support and generates an executable named `HW1_ser`

```
gcc -fopenmp HW1_ser.c -o HW1_ser
```

- Run and Execute the Program

```
./HW1_ser
```

*Expected/Possible Output*:
```
Order 1000 multiplication in ##.##### seconds
 ###.####### mflops

No errors (errsq = 0.0000000000000000e+00).
```

### Create Parallel File

- Create HW1_omp.c File

```
emacs -nw HW1_omp.c
```

- Remove the custom number of threads and allow OpenMP to decide the optimal number of threads based on system resources. Run the following command:
```
unset OMP_NUM_THREADS
```

- Compiles the `HW1_omp.c` file using `GCC` with OpenMP support and generates an executable named `HW1_omp`

```
gcc -fopenmp HW1_omp.c -o HW1_omp
```

Run and Execute the program with `#` threads to control the number of threads used in the OpenMP parallel region.

```
./HW1_omp 1
```

*Expected/Possible Output For the Above Command*

```
Order 1000 multiplication in ##.##### seconds
 set 1 threads
 ###.####### mflops

No errors (errsq = 0.0000000000000000e+00).
```

- Run and Execute the `HW1_omp` program with `#` thread and display the execution time. Run the following command:
```
./HW1_omp 1 | grep seconds
```

*Expected/Possible Output For the Above Command*

```
Order 1000 multiplication in #.##### seconds
```
