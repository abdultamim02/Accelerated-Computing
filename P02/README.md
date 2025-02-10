**Programming Assignement 01 Documentation**
=================================

## Create/Access Files

### Create Serial File

- Create HW1_ser.c File

```
emacs -nw HW2_ser.c
```

- Remove the environment variable `OMP_NUM_THREADS` from the current shell session, if it was previously set. This ensures that any prior setting of the number of OpenMP threads is cleared. Then, set the environment variable `OMP_NUM_THREADS` to `1`, telling OpenMP that the program should run with only `1` thread. Run the following command:

```
unset OMP_NUM_THREADS; export OMP_NUM_THREADS=1
```

- Compiles the `HW2_ser.c` file using `GCC` with OpenMP support and generates an executable named `HW2_ser`

```
gcc -fopenmp HW2_ser.c -o HW2_ser
```

- Run and Execute the Program

```
./HW2_ser
```