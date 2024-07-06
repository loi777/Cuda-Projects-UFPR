// This file will be simply used for the generation and acquirement of user input
#ifndef __INPUT_HANDLER__
#define __INPUT_HANDLER__

#include "../cudaReduceMax.h"

#define FLOATMAX 1000000000.0

// Get the main parameters and saves them in global variables
// finishes generating the vector array to be used as input
void getUserInput(int argc, char** argv, float *vector, u_int *nTotalElements, u_int *nR);

// via parameter alloc space and fills an array of size 's'
// returns the generated vector
void generateArrayInput(float *vector, u_int *nTotalElements);

// free memory space and destroy vector
void destroyArrayInput(float *vector);

#endif // !__INPUT_HANDLER__
