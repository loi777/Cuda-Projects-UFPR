#include "../main.h"

//===============================

float* arrayToReduce;

//===============================



// via parameter alloc space and fills an array of size 's'
// save it in the global variable arrayToReduce.
void generateArrayInput(int s) {
    arrayToReduce = malloc(s * sizeof(float));

    for (int x = 0; x < s; x++) {
        arrayToReduce[x] = ((float)rand()/(float)(RAND_MAX)) * FLOATMAX;
    }
}