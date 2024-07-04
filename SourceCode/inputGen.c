#include "../main.h"

//===============================

float* arrayToReduce;

//===============================


// Gets from the user the size of the array to be reduced
// and generates it in global variable arrayToReduce
void getUserInput() {
    int s;
    printf("Write the size of the array to be generated:\n");
    scanf("%d", &s);

    generateArrayInput(s);
}



// via parameter alloc space and fills an array of size 's'
// save it in the global variable arrayToReduce.
void generateArrayInput(int s) {
    arrayToReduce = malloc(s * sizeof(float));

    for (int x = 0; x < s; x++) {
        arrayToReduce[x] = ((float)rand()/(float)(RAND_MAX)) * FLOATMAX;
    }
}