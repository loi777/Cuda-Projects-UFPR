# Define the CUDA compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -arch=sm_61 -I/usr/include/c++/10 -I/usr/lib/cuda/include/ --std=c++11
# CUDA flags
# CUDAFLAGS = -arch=sm_50

# Source files
SRCS = simpleSort.cu bitonicM.cu histogramM.cu
# Object files
OBJS = $(SRCS:.cu=.o)
# Executable name
TARGET = simpleSort

# Default rule
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(CUDAFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile CUDA source files into object files
%.o: %.cu
	$(NVCC) $(CFLAGS) $(CUDAFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# default run call
run: all
	clear
	./simpleSort 1000000 9 10