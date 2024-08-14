CC = nvcc
CFLAGS = -arch=sm_61 -I/usr/include/c++/10 -I/usr/lib/cuda/include/ --std=c++11
LDLIBS = 
EXE = simpleSort

SRC = $(wildcard ./*.c)
OBJ = $(SRC:.c=.o)
CUSRC = $(wildcard ./*.cu)
CUOBJ = $(SRC:.cu=.o)	

#--


all : $(EXE) clean


#--


purge :	clean
	@rm -f $(EXE)
	@echo "terminado purge"

clean :
	@rm -f $(OBJ)
	@rm -f $(CUOBJ)
	@echo "terminado clean"


#--


$(EXE) : $(OBJ) $(CUOBJ)


#--