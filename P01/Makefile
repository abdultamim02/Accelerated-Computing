# Makefile Class Version

all: hi practice HW1_ser HW1_omp

#--------------------------------------------------------------------                                                                                                                                                                         
# Makefile for hi.c                                                                                                                                                                                                                           
hi: hi.o
        gcc -fopenmp hi.c -o hi
runHi: hi
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=4; ./hi)
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=8; ./hi)

#--------------------------------------------------------------------                                                                                                                                                                         
# Makefile for practice.c                                                                                                                                                                                                                     
practice: practice.o
        gcc -fopenmp practice.c -o practice
runPractice: practice
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=8; ./practice)

#--------------------------------------------------------------------                                                                                                                                                                         
# Makefile for HW1_ser.c                                                                                                                                                                                                                      
HW1_ser: HW1_ser.o
        gcc -fopenmp HW1_ser.c -o HW1_ser
runSer: HW1_ser
        (unset OMP_NUM_THREADS; export OMP_NUM_THREADS=1; ./HW1_ser)

#--------------------------------------------------------------------                                                                                                                                                                         
# Makefile for HW1_omp.c                                                                                                                                                                                                                      
HW1_omp: HW1_omp.o
        gcc -fopenmp HW1_omp.c -o HW1_omp
runOmp: HW1_omp
        unset OMP_NUM_THREADS
        ./HW1_omp 1
        ./HW1_omp 1 | grep seconds


# ===============================================

# Makefile for hi.c

# Compiler                                                                                                                                                                                                                                    
CC = gcc

# Compiler flags                                                                                                                                                                                                                              
CFLAGS = -Wall -g

# Target executable                                                                                                                                                                                                                           
TARGET = my_program

# Source files                                                                                                                                                                                                                                
SRCS = main.c
OBJS = $(SRCS:.c=.o)

# Rule to build the executable                                                                                                                                                                                                                
$(TARGET): $(OBJS)
        $(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile .c files into .o files                                                                                                                                                                                                      
%.o: %.c
        $(CC) $(CFLAGS) -c $< -o $@

# Clean up generated files                                                                                                                                                                                                                    
clean:
        rm -f $(TARGET) $(OBJS)

# ===============================================

# Makefile for HW1_ser.c

# Compiler                                                                                                                                                                                                                                    
CC = gcc

# Compiler flags (Enable OpenMP)                                                                                                                                                                                                              
CFLAGS = -Wall -g -fopenmp

# Target executable for HW1_ser                                                                                                                                                                                                               
TARGET = HW1_ser

# Source files for HW1_ser                                                                                                                                                                                                                    
SRCS = HW1_ser.c

# Object files for HW1_ser                                                                                                                                                                                                                    
OBJS = $(SRCS:.c=.o)

# Rule to build the executable (HW1_ser)                                                                                                                                                                                                      
$(TARGET): $(OBJS)
        $(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile .c files into .o files                                                                                                                                                                                                      
%.o: %.c
        $(CC) $(CFLAGS) -c $< -o $@

# Rule to set OMP_NUM_THREADS, compile and run HW1_ser                                                                                                                                                                                        
run_HW1_ser: $(TARGET)
        unset OMP_NUM_THREADS; export OMP_NUM_THREADS=1; ./$(TARGET)

# Clean up generated files                                                                                                                                                                                                                    
clean:
        rm -f $(TARGET) $(OBJS)
