CC = nvcc
CFLAGS = -arch=sm_61 -I/usr/include/c++/10 -I/usr/lib/cuda/include/ --std=c++11
LDLIBS = 

EXEC = bitonic



all: $(EXEC) clean

$(EXEC): $(EXEC).cu
	$(CC) $(CFLAGS) $(EXEC).cu -o $(EXEC) $(LDLIBS)

debug: CFLAGS += -lineinfo -I/usr/lib/nvidia-cuda-toolkit/compute-sanitizer -g
debug: all

clean:
	@rm -f *.o
purge: clean
	@rm -f $(EXEC)
